#!/usr/bin/env python3
"""
Head Emotion & Face Tracking Controller
Drives pan/tilt servos via PCA9685 to follow faces detected by trackingeyes.py.
Can be imported and started as a thread, or run standalone.

Wiring (PCA9685 default I2C 0x40):
  Channel 0 → Pan servo  (horizontal, left/right)
  Channel 1 → Tilt servo (vertical,   up/down)
"""

import time
import math
import random
import threading
import sys

try:
    from simple_pid import PID
except ImportError:
    print("Error: simple-pid not found. pip3 install simple-pid")
    sys.exit(1)

try:
    from adafruit_servokit import ServoKit
except ImportError:
    print("Error: adafruit-circuitpython-servokit not found.")
    print("pip3 install adafruit-circuitpython-servokit")
    sys.exit(1)


# ─── Servo Configuration ────────────────────────────────────────────────────

# PCA9685 channel assignments
PAN_CHANNEL  = 0   # horizontal (left ↔ right)
TILT_CHANNEL = 1   # vertical   (up ↕ down)

# Servo physical limits (degrees)
PAN_MIN   = 30
PAN_MAX   = 150
TILT_MIN  = 50
TILT_MAX  = 130

# Center position (neutral / resting pose)
PAN_CENTER  = 90
TILT_CENTER = 90

# Servo deadband – ignore movements smaller than this (degrees) to avoid jitter
DEADBAND = 0.4

# PCA9685 actuation range (µs), adjust to match your servos
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500


# ─── PID Gains ───────────────────────────────────────────────────────────────
# Positive Kp moves servo in the direction that reduces error.
# Tune these on the physical robot; start conservatively.

PAN_KP  = 28.0
PAN_KI  = 0.05
PAN_KD  = 3.5

TILT_KP = 22.0
TILT_KI = 0.04
TILT_KD = 2.8

PID_SAMPLE_TIME = 0.04   # 25 Hz PID update rate


# ─── Tracking Smoothing ──────────────────────────────────────────────────────
# EMA alpha for raw face-position input (lower = smoother but laggier)
TRACK_SMOOTH_ALPHA = 0.18

# Dead zone: ignore face offsets smaller than this fraction of MAX offset
TRACK_DEAD_ZONE = 0.06

# Scale from normalised face offset (±1) to servo degrees from centre
PAN_SCALE  = 45.0   # max ±45° pan  from centre when face is at edge of frame
TILT_SCALE = 30.0   # max ±30° tilt from centre


# ─── Gesture Config ──────────────────────────────────────────────────────────
NOD_AMPLITUDE  = 12.0   # degrees
NOD_DURATION   = 0.55   # seconds per half-cycle
NOD_REPS       = 2

SHAKE_AMPLITUDE = 14.0
SHAKE_DURATION  = 0.40
SHAKE_REPS      = 3

TILT_AMPLITUDE  = 15.0
TILT_DURATION   = 0.8

RETURN_SPEED    = 0.12   # fraction per update (EMA towards target)

# Idle wander (gentle micro-movements when no face is detected)
IDLE_WANDER_AMPLITUDE = 5.0    # degrees
IDLE_WANDER_PERIOD    = 8.0    # seconds

# How long (s) to wait without a face before returning to centre
NO_FACE_RETURN_DELAY = 2.5


# ─── Controller loop rate ────────────────────────────────────────────────────
CONTROLLER_FPS = 30


# ─────────────────────────────────────────────────────────────────────────────
class HeadController:
    """
    Manages pan/tilt servos and expressive head gestures.
    Call start() to run in a background thread.
    """

    def __init__(self):
        self.kit = ServoKit(channels=16)

        # Configure actuation range
        for ch in (PAN_CHANNEL, TILT_CHANNEL):
            self.kit.servo[ch].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)

        # Current servo positions (degrees)
        self._pan_pos  = float(PAN_CENTER)
        self._tilt_pos = float(TILT_CENTER)

        # Smoothed face-tracking targets (degrees, relative to centre)
        self._smooth_x = 0.0
        self._smooth_y = 0.0

        # Time of last face detection
        self._last_face_time = 0.0

        # PID controllers (setpoint = 0 error, output = degree correction)
        self._pan_pid = PID(
            PAN_KP, PAN_KI, PAN_KD,
            setpoint=0.0,
            output_limits=(-PAN_SCALE, PAN_SCALE),
            sample_time=PID_SAMPLE_TIME,
        )
        self._tilt_pid = PID(
            TILT_KP, TILT_KI, TILT_KD,
            setpoint=0.0,
            output_limits=(-TILT_SCALE, TILT_SCALE),
            sample_time=PID_SAMPLE_TIME,
        )

        # Gesture state ("idle", "nod", "shake", "tilt")
        self._gesture        = "idle"
        self._gesture_lock   = threading.Lock()
        self._gesture_done   = threading.Event()

        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None

        # Move to centre on init
        self._write_servos(PAN_CENTER, TILT_CENTER)
        print("HeadController: servos centred.")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the background control loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="HeadCtrl")
        self._thread.start()
        print("HeadController: started.")

    def stop(self):
        """Stop the loop and centre the head."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._write_servos(PAN_CENTER, TILT_CENTER)
        print("HeadController: stopped.")

    def nod(self, reps: int = NOD_REPS):
        """Trigger a nodding gesture (non-blocking)."""
        self._queue_gesture("nod", reps=reps)

    def shake(self, reps: int = SHAKE_REPS):
        """Trigger a head-shake gesture (non-blocking)."""
        self._queue_gesture("shake", reps=reps)

    def tilt(self, direction: str = "right"):
        """Tilt the head left or right (non-blocking)."""
        self._queue_gesture("tilt", direction=direction)

    def update_face(self, norm_x: float, norm_y: float):
        """
        Feed a normalised face position.
        norm_x: −1.0 (far left) … +1.0 (far right)
        norm_y: −1.0 (top)      … +1.0 (bottom)
        Call with (0, 0) or simply don't call when no face is present.
        """
        self._last_face_time = time.time()
        self._smooth_x += (norm_x - self._smooth_x) * TRACK_SMOOTH_ALPHA
        self._smooth_y += (norm_y - self._smooth_y) * TRACK_SMOOTH_ALPHA

    # ── Internal ──────────────────────────────────────────────────────────────

    def _queue_gesture(self, name, **kwargs):
        with self._gesture_lock:
            self._gesture_kwargs = kwargs
            self._gesture = name

    def _clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def _write_servos(self, pan: float, tilt: float):
        pan  = self._clamp(pan,  PAN_MIN,  PAN_MAX)
        tilt = self._clamp(tilt, TILT_MIN, TILT_MAX)

        if abs(pan  - self._pan_pos)  > DEADBAND:
            self.kit.servo[PAN_CHANNEL].angle  = pan
            self._pan_pos  = pan
        if abs(tilt - self._tilt_pos) > DEADBAND:
            self.kit.servo[TILT_CHANNEL].angle = tilt
            self._tilt_pos  = tilt

    # ── Gesture runners (blocking, called from loop) ───────────────────────

    def _run_nod(self, reps=NOD_REPS):
        base_tilt = self._tilt_pos
        for _ in range(reps):
            self._smooth_to(self._pan_pos, base_tilt + NOD_AMPLITUDE, NOD_DURATION)
            self._smooth_to(self._pan_pos, base_tilt - NOD_AMPLITUDE * 0.4, NOD_DURATION * 0.7)
        self._smooth_to(self._pan_pos, base_tilt, NOD_DURATION * 0.5)

    def _run_shake(self, reps=SHAKE_REPS):
        base_pan = self._pan_pos
        for _ in range(reps):
            self._smooth_to(base_pan + SHAKE_AMPLITUDE, self._tilt_pos, SHAKE_DURATION)
            self._smooth_to(base_pan - SHAKE_AMPLITUDE, self._tilt_pos, SHAKE_DURATION)
        self._smooth_to(base_pan, self._tilt_pos, SHAKE_DURATION * 0.6)

    def _run_tilt(self, direction="right"):
        base_pan = self._pan_pos
        delta = TILT_AMPLITUDE if direction == "right" else -TILT_AMPLITUDE
        # "Tilt" = pan rotates so head cocks to the side
        self._smooth_to(base_pan + delta, self._tilt_pos, TILT_DURATION)
        time.sleep(0.4)
        self._smooth_to(base_pan, self._tilt_pos, TILT_DURATION * 0.8)

    def _smooth_to(self, target_pan: float, target_tilt: float, duration: float):
        """Interpolate servos from current position to target over `duration` seconds."""
        steps = max(1, int(duration * CONTROLLER_FPS))
        start_pan  = self._pan_pos
        start_tilt = self._tilt_pos
        for i in range(1, steps + 1):
            if not self._running:
                return
            t = i / steps
            # Ease in/out
            t_e = t * t * (3 - 2 * t)
            pan  = start_pan  + (target_pan  - start_pan)  * t_e
            tilt = start_tilt + (target_tilt - start_tilt) * t_e
            self._write_servos(pan, tilt)
            time.sleep(1.0 / CONTROLLER_FPS)

    # ── Main control loop ─────────────────────────────────────────────────────

    def _loop(self):
        interval = 1.0 / CONTROLLER_FPS
        next_tick = time.perf_counter()

        while self._running:
            try:
                # Check for queued gesture
                with self._gesture_lock:
                    gesture = self._gesture
                    kwargs  = getattr(self, "_gesture_kwargs", {})
                    if gesture != "idle":
                        self._gesture = "idle"

                if gesture == "nod":
                    self._run_nod(**{k: v for k, v in kwargs.items() if k == "reps"})
                elif gesture == "shake":
                    self._run_shake(**{k: v for k, v in kwargs.items() if k == "reps"})
                elif gesture == "tilt":
                    self._run_tilt(**{k: v for k, v in kwargs.items() if k == "direction"})
                else:
                    self._track()

            except Exception as e:
                print(f"HeadController loop error: {e}")

            next_tick += interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    def _track(self):
        """PID tracking update – called every loop iteration when no gesture is active."""
        now = time.time()
        face_age = now - self._last_face_time

        if face_age > NO_FACE_RETURN_DELAY:
            # No face: idle wander + slowly drift back to centre
            wander = math.sin(now * (2 * math.pi / IDLE_WANDER_PERIOD)) * IDLE_WANDER_AMPLITUDE
            target_pan  = PAN_CENTER  + wander
            target_tilt = TILT_CENTER + wander * 0.3
            self._pan_pos  += (target_pan  - self._pan_pos)  * RETURN_SPEED
            self._tilt_pos += (target_tilt - self._tilt_pos) * RETURN_SPEED
            self._write_servos(self._pan_pos, self._tilt_pos)
            # Reset PID integrators so there's no windup accumulation
            self._pan_pid.reset()
            self._tilt_pid.reset()
            self._smooth_x *= 0.92
            self._smooth_y *= 0.92
            return

        # Apply dead zone
        sx = self._smooth_x if abs(self._smooth_x) > TRACK_DEAD_ZONE else 0.0
        sy = self._smooth_y if abs(self._smooth_y) > TRACK_DEAD_ZONE else 0.0

        # Error = how far the face is from centre of frame (in degrees)
        pan_error  = sx * PAN_SCALE   # positive → face is to the right → pan right
        tilt_error = sy * TILT_SCALE  # positive → face is low in frame  → tilt down

        pan_correction  = self._pan_pid(-pan_error)   # negate: servo moves to reduce error
        tilt_correction = self._tilt_pid(-tilt_error)

        new_pan  = self._clamp(PAN_CENTER  + pan_correction,  PAN_MIN,  PAN_MAX)
        new_tilt = self._clamp(TILT_CENTER + tilt_correction, TILT_MIN, TILT_MAX)

        self._write_servos(new_pan, new_tilt)


# ─── Integration helper ───────────────────────────────────────────────────────

_controller: HeadController | None = None
_controller_lock = threading.Lock()


def get_controller() -> HeadController:
    """Return the singleton HeadController, creating it if needed."""
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = HeadController()
        return _controller


def start_head_tracking():
    """
    Convenience: create and start the head controller.
    Call this from trackingeyes.py after the camera is ready.
    """
    ctrl = get_controller()
    ctrl.start()
    return ctrl


def feed_face(norm_x: float, norm_y: float):
    """
    Feed the latest normalised face position to the controller.
    norm_x:  −1 = left edge,  +1 = right edge of camera frame
    norm_y:  −1 = top edge,   +1 = bottom edge
    Designed to be called from the vision_worker thread in trackingeyes.py.
    """
    global _controller
    if _controller is not None:
        _controller.update_face(norm_x, norm_y)


# ─── Standalone execution ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal

    print("Running HeadController standalone (no camera – free-run idle wander).")
    ctrl = start_head_tracking()

    def _sigint(sig, frame):
        print("\nShutting down...")
        ctrl.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    # Demo: cycle through gestures
    time.sleep(3)
    print("Nodding...")
    ctrl.nod()
    time.sleep(3)
    print("Shaking...")
    ctrl.shake()
    time.sleep(3)
    print("Tilting right...")
    ctrl.tilt("right")
    time.sleep(3)
    print("Tilting left...")
    ctrl.tilt("left")
    time.sleep(3)

    print("Idle wander. Ctrl-C to exit.")
    signal.pause()