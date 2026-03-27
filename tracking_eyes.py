#!/usr/bin/env python3
"""
Face Tracking Eyes for Dual SPI Displays (Picamera2)
Combines face tracking (YuNet) with dual SPI display output (ST7735).
"""

import time
import math
import random
import sys
import io
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import cv2
from pathlib import Path

# Hardware / Display Imports
import board
import busio
import digitalio
from PIL import Image, ImageDraw
try:
    from adafruit_rgb_display import st7735
except ImportError:
    print("Error: adafruit-circuitpython-rgb-display not found.")
    print("pip3 install adafruit-circuitpython-rgb-display")
    sys.exit(1)

# Camera Import
try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 not found. Please install with: sudo apt install python3-picamera2")
    sys.exit(1)


# --- Configuration ---
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 160
EYE_COLOR = (255, 255, 255) # White
BG_COLOR = (0, 0, 0)       # Black
EYE_SIZE = 120              # Base size
FLOOR_Y = SCREEN_HEIGHT - 5

# Camera / Face Tracking Config
FACE_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
# We'll capture at low res for faster processing
CAMERA_RES = (320, 240)
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
# Camera adjustments
CAMERA_ROTATE_180 = True
# If stream colors look wrong, swap R/B for MJPEG output
STREAM_SWAP_RB = True

# Eye Interaction Config
MAX_X_OFFSET = 50
MAX_Y_OFFSET = 35
FACE_ROLL_MULT = 0.75
FACE_ROLL_MAX_DEG = 10.0

# Blink Speed (Higher = Faster)
BLINK_SPEED_MIN = 3.5
BLINK_SPEED_MAX = 5.0

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle":       {"scale_w": 1.0, "scale_h": 1.0,  "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle":   0.0, "mirror_angle": True},
    "happy":      {"scale_w": 1.2, "scale_h": 0.65, "top_lid": 0.0,  "bottom_lid": 0.55, "lid_angle": -12.0, "mirror_angle": True},
    "sad":        {"scale_w": 1.1, "scale_h": 1.1,  "top_lid": 0.35, "bottom_lid": 0.0,  "lid_angle":  15.0, "mirror_angle": True},
    "angry":      {"scale_w": 1.0, "scale_h": 0.9,  "top_lid": 0.35, "bottom_lid": 0.0,  "lid_angle": -20.0, "mirror_angle": True},
    "surprised":  {"scale_w": 0.9, "scale_h": 1.3,  "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle":   0.0, "mirror_angle": True},
    "suspicious": {"scale_w": 1.1, "scale_h": 0.6,  "top_lid": 0.4,  "bottom_lid": 0.4,  "lid_angle":   0.0, "mirror_angle": True},
    "sleepy":     {"scale_w": 1.1, "scale_h": 1.0,  "top_lid": 0.6,  "bottom_lid": 0.0,  "lid_angle":   0.0, "mirror_angle": True},
    "looking":    {"scale_w": 1.0, "scale_h": 0.9,  "top_lid": 0.25, "bottom_lid": 0.0,  "lid_angle":  -8.0, "mirror_angle": False},
}

SPECIAL_EMOTIONS = ["happy", "suspicious", "sleepy"]

# MJPEG Stream Config (for headless SSH viewing)
STREAM_ENABLED = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8080
STREAM_FPS = 12
STREAM_JPEG_QUALITY = 80


# --- BlockyEye Class (PIL Version with emotion controls) ---
class BlockyEye:
    def __init__(self, x, y, scale=1.0, is_left=True):
        self.base_x, self.base_y = x, y
        self.current_pos = [float(x), float(y)]
        self.target_pos = [float(x), float(y)]

        self.vel_x = 0.0
        self.vel_y = 0.0

        self.base_w = EYE_SIZE * scale
        self.base_h = EYE_SIZE * scale

        self.current_w = self.base_w
        self.current_h = self.base_h
        self.target_w = self.base_w
        self.target_h = self.base_h

        self.vel_w = 0.0
        self.vel_h = 0.0

        self.w = self.base_w
        self.h = self.base_h

        self.current_rotation = 0.0
        self.target_rotation = 0.0
        self.rot_sensitivity = random.uniform(0.3, 0.5)
        self.rot_speed = random.uniform(0.15, 0.25)

        self.is_left = is_left
        self.blink_state = "IDLE"
        self.vy = 0
        self.blink_speed_mult = 1.0

        self.target_scale_w = 1.0
        self.target_scale_h = 1.0
        self.scale_w = 1.0
        self.scale_h = 1.0
        self.scale_w_vel = 0.0
        self.scale_h_vel = 0.0
        self.top_lid = 0.0
        self.bottom_lid = 0.0
        self.lid_angle = 0.0
        self.top_lid_vel = 0.0
        self.bottom_lid_vel = 0.0
        self.lid_angle_vel = 0.0
        self.target_top_lid = 0.0
        self.target_bottom_lid = 0.0
        self.target_lid_angle = 0.0
        self.current_emotion = "idle"
        self.happy_phase = random.uniform(0.0, math.pi * 2)
        self.happy_burst_until = 0.0

        self.noise_t = random.uniform(0, 100)

    def start_blink(self, speed_mult=None):
        if self.blink_state == "IDLE":
            self.blink_state = "DROPPING"
            if speed_mult is not None:
                self.blink_speed_mult = speed_mult
            else:
                self.blink_speed_mult = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            self.vy = 40 * self.blink_speed_mult

    def set_emotion(self, emotion_name: str, intensity: float = 1.0):
        if emotion_name not in EMOTION_PRESETS:
            return

        if emotion_name == "happy" and self.current_emotion != "happy":
            self.happy_burst_until = time.time() + 0.35

        self.current_emotion = emotion_name
        preset = EMOTION_PRESETS[emotion_name]
        idle = EMOTION_PRESETS["idle"]

        intensity = max(0.0, min(1.0, intensity))
        scale_w    = idle["scale_w"]    + (preset["scale_w"]    - idle["scale_w"])    * intensity
        scale_h    = idle["scale_h"]    + (preset["scale_h"]    - idle["scale_h"])    * intensity
        top_lid    = idle["top_lid"]    + (preset["top_lid"]    - idle["top_lid"])    * intensity
        bottom_lid = idle["bottom_lid"] + (preset["bottom_lid"] - idle["bottom_lid"]) * intensity
        lid_angle  = idle["lid_angle"]  + (preset["lid_angle"]  - idle["lid_angle"])  * intensity

        self.target_scale_w    = scale_w
        self.target_scale_h    = scale_h
        self.target_top_lid    = top_lid
        self.target_bottom_lid = bottom_lid

        if preset.get("mirror_angle", True) and not self.is_left and abs(lid_angle) > 0:
            lid_angle = -lid_angle
        self.target_lid_angle = lid_angle

    def update(self):
        if self.blink_state == "IDLE":
            t = time.time() + self.noise_t
            noise_x = (math.sin(t * 1.3) * 0.2 + math.sin(t * 0.7) * 0.1)
            noise_y = (math.cos(t * 1.1) * 0.2 + math.cos(t * 0.9) * 0.1)

            target_x_phys = self.target_pos[0] + noise_x
            target_y_phys = self.target_pos[1] + noise_y

            burst_active = time.time() < self.happy_burst_until
            if burst_active:
                target_y_phys -= 12.0
                self.target_top_lid    = max(self.target_top_lid,    0.95)
                self.target_bottom_lid = max(self.target_bottom_lid, 0.95)
                self.target_lid_angle  = 0.0

            if self.current_emotion == "happy":
                ht = time.time() * 6.0 + self.happy_phase
                target_y_phys -= 4.0 + math.sin(ht) * 3.5
                target_x_phys += math.sin(ht * 1.7) * 2.0

            dx = target_x_phys - self.current_pos[0]
            dy = target_y_phys - self.current_pos[1]

            speed_x = 0.20
            speed_y = 0.22
            if dy < -1.0:
                speed_y = 0.14
            elif dy > 1.0:
                speed_y = 0.38

            self.current_pos[0] += dx * speed_x
            self.current_pos[1] += dy * speed_y

            self.vel_x = dx * speed_x
            self.vel_y = dy * speed_y

            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            look_rot = (rel_x * 0.5 + rel_y * 0.8) * self.rot_sensitivity
            if self.current_emotion == "happy":
                look_rot += math.sin(time.time() * 8.0 + self.happy_phase) * 1.2
            final_target_rot = look_rot + self.target_rotation
            self.current_rotation += (final_target_rot - self.current_rotation) * self.rot_speed

            t = time.time()
            breath_w = (math.sin(t * 1.5 + self.base_x) * 1.5 + math.sin(t * 0.5) * 1.0)
            breath_h = (math.cos(t * 1.8 + self.base_y) * 1.5 + math.cos(t * 0.6) * 1.0)

            move_stretch_x = (dx * speed_x) * 2.5
            move_stretch_y = (dy * speed_y) * 2.5

            k = 0.22
            d = 0.55
            self.scale_w_vel   = (self.scale_w_vel   + (self.target_scale_w   - self.scale_w)   * k) * d
            self.scale_h_vel   = (self.scale_h_vel   + (self.target_scale_h   - self.scale_h)   * k) * d
            self.scale_w      += self.scale_w_vel
            self.scale_h      += self.scale_h_vel

            self.top_lid_vel    = (self.top_lid_vel    + (self.target_top_lid    - self.top_lid)    * k) * d
            self.bottom_lid_vel = (self.bottom_lid_vel + (self.target_bottom_lid - self.bottom_lid) * k) * d
            self.lid_angle_vel  = (self.lid_angle_vel  + (self.target_lid_angle  - self.lid_angle)  * k) * d

            self.top_lid    += self.top_lid_vel
            self.bottom_lid += self.bottom_lid_vel
            self.lid_angle  += self.lid_angle_vel

            self.target_w = (self.base_w * self.scale_w) + breath_w + (move_stretch_x * 0.5)
            self.target_h = (self.base_h * self.scale_h) + breath_h - (move_stretch_y * 0.2)

        elif self.blink_state == "DROPPING":
            self.vy += 10 * self.blink_speed_mult
            self.current_pos[1] += self.vy
            self.current_w = self.base_w - 10
            self.current_h = self.base_h + 20
            self.target_w = self.current_w
            self.target_h = self.current_h

            if self.current_pos[1] + self.current_h // 2 >= FLOOR_Y:
                self.current_pos[1] = FLOOR_Y - self.current_h // 2
                self.blink_state = "SQUASHING"
                self.velocity = [0.0, 0.0]

        elif self.blink_state == "SQUASHING":
            squeeze_speed = 65 * self.blink_speed_mult
            spread_speed  = 40 * self.blink_speed_mult
            self.current_h -= squeeze_speed
            self.current_w += spread_speed
            self.current_pos[1] = FLOOR_Y - self.current_h // 2

            if self.current_h <= 22:
                self.current_h = 22
                self.blink_state = "JUMPING"

        elif self.blink_state == "JUMPING":
            recovery_speed = max(0.15, min(0.95, 0.85 * self.blink_speed_mult))
            self.current_h += (self.base_h - self.current_h) * recovery_speed
            self.current_w += (self.base_w - self.current_w) * recovery_speed

            self.vel_x = (self.vel_x + (self.target_pos[0] - self.current_pos[0]) * 0.1) * 0.8
            self.current_pos[0] += self.vel_x

            target_y = self.target_pos[1]
            self.current_pos[1] += (target_y - self.current_pos[1]) * 0.8

            if abs(self.current_h - self.base_h) < 5 and abs(self.current_pos[1] - target_y) < 5:
                self.current_h = self.base_h
                self.current_w = self.base_w
                self.blink_state = "IDLE"
                self.vy = 0
                self.vel_x = 0
                self.vel_y = 0

        if self.blink_state == "IDLE":
            k = 0.08
            d = 0.90
            force_w = (self.target_w - self.current_w) * k
            self.vel_w = (self.vel_w + force_w) * d
            self.current_w += self.vel_w

            force_h = (self.target_h - self.current_h) * k
            self.vel_h = (self.vel_h + force_h) * d
            self.current_h += self.vel_h
        else:
            self.vel_w = 0
            self.vel_h = 0

        self.w = self.current_w
        self.h = self.current_h

    def draw_radial_rect(self, draw, x, y, w, h, color, radius, pupil_offset=(0, 0)):
        center_x = x + w / 2
        center_y = y + h / 2
        steps = 4
        for i in range(steps):
            size_factor = 1.0 - (i / steps)
            current_w = w * size_factor
            current_h = h * size_factor
            if current_w <= 0 or current_h <= 0:
                continue
            b_factor = 0.85 + 0.15 * (i / steps)
            cur_color = (
                int(color[0] * b_factor),
                int(color[1] * b_factor),
                int(color[2] * b_factor),
            )
            shift_x = pupil_offset[0] * (1.0 - size_factor) * 15
            shift_y = pupil_offset[1] * (1.0 - size_factor) * 15
            cx = center_x + shift_x
            cy = center_y + shift_y
            x0 = cx - current_w / 2
            y0 = cy - current_h / 2
            x1 = cx + current_w / 2
            y1 = cy + current_h / 2
            base_radius = int(radius)
            cur_radius = min(base_radius, int(min(current_w, current_h) / 2))
            draw.rounded_rectangle([x0, y0, x1, y1], radius=cur_radius, fill=cur_color)

    def draw_eyelids(self, eye_img, rect):
        x0, y0, x1, y1 = rect
        w = int(x1 - x0)
        h = int(y1 - y0)
        lid_color = BG_COLOR

        if self.top_lid > 0.01:
            lid_h = int(h * self.top_lid)
            lid_src = Image.new("RGBA", (int(w + 20), int(lid_h + 20)), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y0 - 10)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            lid_src = Image.new("RGBA", (int(w + 20), int(lid_h + 20)), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y1 + 10 - lid_src.height)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

    def draw(self, bg_image):
        draw_w = max(4, int(self.w))
        draw_h = max(4, int(self.h))

        eye_img_size = int(max(self.base_w, self.base_h) * 2.5)
        eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0, 0, 0, 0))
        eye_draw = ImageDraw.Draw(eye_img)

        base_radius   = int(min(self.base_w, self.base_h) * 0.25)
        corner_radius = min(base_radius, int(min(draw_w, draw_h) / 2))
        off_x = max(-1, min(1, (self.current_pos[0] - self.base_x) / 30.0))
        off_y = max(-1, min(1, (self.current_pos[1] - self.base_y) / 20.0))

        cx, cy = eye_img_size / 2, eye_img_size / 2
        x0 = cx - draw_w / 2
        y0 = cy - draw_h / 2
        x1 = cx + draw_w / 2
        y1 = cy + draw_h / 2

        self.draw_radial_rect(eye_draw, x0, y0, draw_w, draw_h, EYE_COLOR, corner_radius, (off_x, off_y))
        self.draw_eyelids(eye_img, (x0, y0, x1, y1))

        rotated = eye_img.rotate(self.current_rotation, resample=Image.BICUBIC, expand=False)

        paste_x = int(self.current_pos[0] - eye_img_size / 2)
        paste_y = int(self.current_pos[1] - eye_img_size / 2)
        bg_image.alpha_composite(rotated, (paste_x, paste_y))


# --- MJPEG Streaming Server ---
latest_frame = None
frame_lock = threading.Lock()
stream_server = None


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                with frame_lock:
                    frame = None if latest_frame is None else latest_frame.copy()

                if frame is None:
                    time.sleep(0.05)
                    continue

                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=STREAM_JPEG_QUALITY)
                jpg = buf.getvalue()

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / max(1, STREAM_FPS))
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format, *args):
        return


def start_stream_server():
    global stream_server
    stream_server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), MJPEGHandler)
    thread = threading.Thread(target=stream_server.serve_forever, daemon=True)
    thread.start()
    print(f"MJPEG stream started: http://{STREAM_HOST}:{STREAM_PORT}/stream")


# --- Display Setup ---
print("Initializing Displays (Dual SPI)...")
disp_l = None
disp_r = None

# SPI 0 (Left Screen)
try:
    spi0 = board.SPI()
    disp_l = st7735.ST7735R(
        spi0,
        rotation=0,
        baudrate=24000000,
        bgr=True,
        cs=digitalio.DigitalInOut(board.CE1),
        dc=digitalio.DigitalInOut(board.D24),
        rst=digitalio.DigitalInOut(board.D25),
    )
except Exception as e:
    print(f"Error init Left Display (SPI0): {e}")

# SPI 1 (Right Screen)
try:
    spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19)
    disp_r = st7735.ST7735R(
        spi1,
        rotation=0,
        baudrate=24000000,
        bgr=True,
        cs=digitalio.DigitalInOut(board.D18),
        dc=digitalio.DigitalInOut(board.D23),
        rst=digitalio.DigitalInOut(board.D27),
    )
except Exception as e:
    print(f"Error init Right Display (SPI1): {e}")


# --- Camera & Face Detector Setup ---
print("Initializing Picamera2...")
picam2 = None
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": CAMERA_RES}
    )
    picam2.configure(config)
    picam2.start()
    print(f"Camera started: {CAMERA_RES}")
except Exception as e:
    print(f"Error starting Picamera2: {e}")
    sys.exit(1)

print("Initializing YuNet Face Detector...")
try:
    if not Path(FACE_MODEL_PATH).exists():
        print(f"Error: Face model not found at {FACE_MODEL_PATH}")
        sys.exit(1)

    detector = cv2.FaceDetectorYN.create(
        model=FACE_MODEL_PATH,
        config="",
        input_size=CAMERA_RES,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU,
    )
    print("YuNet initialized.")
except Exception as e:
    print(f"Error initializing detector: {e}")
    sys.exit(1)


# --- MJPEG Stream ---
if STREAM_ENABLED:
    try:
        start_stream_server()
    except Exception as e:
        print(f"Error starting MJPEG stream: {e}")


# --- Eye Objects ---
center_x = SCREEN_WIDTH / 2
center_y = SCREEN_HEIGHT / 2

left_eye  = BlockyEye(center_x, center_y, scale=1.0, is_left=True)
right_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=False)
left_eye.set_emotion("idle",  0.45)
right_eye.set_emotion("idle", 0.45)

# Animation Loop State
running          = True
next_blink_time  = time.time() + random.uniform(3, 6)
last_blink_time  = time.time()
frame_skip_counter = 0
last_face_time   = time.time()
search_phase     = random.uniform(0.0, math.pi * 2)
search_active_until = 0.0
next_search_time = time.time() + random.uniform(6.0, 12.0)
next_special_time = time.time() + random.uniform(12.0, 20.0)
special_active_until = 0.0
special_emotion  = None
smoothed_x_off   = 0.0
smoothed_y_off   = 0.0
smoothed_rotation = 0.0
search_emotion_active = False
search_style     = "plain"

print("Starting Tracking Loop...")
time.sleep(1.0)  # Warmup

try:
    while running:
        start_time = time.time()

        target_x_off  = 0
        target_y_off  = 0
        target_rotation = 0.0

        if frame_skip_counter % 2 == 0:
            try:
                frame = picam2.capture_array()

                if CAMERA_ROTATE_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                if frame is not None and frame.size > 0:
                    if STREAM_ENABLED:
                        stream_frame = frame
                        if STREAM_SWAP_RB:
                            stream_frame = cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB)
                        with frame_lock:
                            latest_frame = stream_frame

                    detector.setInputSize((frame.shape[1], frame.shape[0]))
                    faces = detector.detect(frame)

                    if faces[1] is not None:
                        detected_faces = faces[1]
                        largest_face = max(detected_faces, key=lambda f: f[2] * f[3])

                        fx, fy, fw, fh = largest_face[0:4]
                        re_x, re_y = largest_face[4], largest_face[5]
                        le_x, le_y = largest_face[6], largest_face[7]

                        face_cx = (fx + fw / 2) / CAMERA_RES[0]
                        face_cy = (fy + fh / 2) / CAMERA_RES[1]

                        norm_x = (face_cx - 0.5) * 2.0
                        norm_y = (face_cy - 0.5) * 2.0
                        norm_x = -norm_x
                        target_x_off = norm_x * MAX_X_OFFSET
                        target_y_off = norm_y * MAX_Y_OFFSET
                        target_x_off = max(-MAX_X_OFFSET, min(MAX_X_OFFSET, target_x_off))
                        target_y_off = max(-MAX_Y_OFFSET, min(MAX_Y_OFFSET, target_y_off))

                        dx = re_x - le_x
                        dy = re_y - le_y
                        if dx != 0:
                            angle_rad = math.atan2(dy, dx)
                            angle_deg = math.degrees(angle_rad)
                            target_rotation = max(
                                -FACE_ROLL_MAX_DEG,
                                min(FACE_ROLL_MAX_DEG, -angle_deg * FACE_ROLL_MULT),
                            )

                        last_face_time = time.time()

                        left_eye.set_emotion("idle",  0.45)
                        right_eye.set_emotion("idle", 0.45)
                        special_emotion = None
                        search_emotion_active = False

            except Exception as e:
                print(f"Capture/Detect Error: {e}")

        frame_skip_counter += 1
        now = time.time()

        # No-face idle behaviours: looking around + special emotions
        if now - last_face_time > 0.8:
            if now > next_search_time:
                search_active_until = now + random.uniform(1.2, 2.2)
                next_search_time    = now + random.uniform(6.0, 12.0)
                search_style = "looking" if random.random() < 0.5 else "plain"

            if now < search_active_until:
                t = now * 2.2 + search_phase
                target_x_off   = math.sin(t)       * (MAX_X_OFFSET * 0.75)
                target_y_off   = math.sin(t * 1.2) * (MAX_Y_OFFSET * 0.22)
                target_rotation = math.sin(t * 1.4) * 0.6
                if special_emotion is None and search_style == "looking":
                    search_emotion_active = True
                    left_eye.set_emotion("looking",  intensity=0.45)
                    right_eye.set_emotion("looking", intensity=0.45)
            elif search_emotion_active and special_emotion is None:
                search_emotion_active = False
                left_eye.set_emotion("idle",  0.45)
                right_eye.set_emotion("idle", 0.45)

            if now > next_special_time and special_emotion is None:
                special_active_until = now + random.uniform(1.2, 2.2)
                special_emotion = random.choice(SPECIAL_EMOTIONS)
                left_eye.set_emotion(special_emotion,  intensity=0.55)
                right_eye.set_emotion(special_emotion, intensity=0.55)
                for eye in (left_eye, right_eye):
                    eye.blink_state = "IDLE"
                    eye.vy = 0
                next_special_time = now + random.uniform(12.0, 20.0)

            if special_emotion and now > special_active_until:
                special_emotion = None
                left_eye.set_emotion("idle",  0.45)
                right_eye.set_emotion("idle", 0.45)

        # Smooth tracking to reduce jitter
        smooth_alpha      = 0.15
        smoothed_x_off    += (target_x_off    - smoothed_x_off)    * smooth_alpha
        smoothed_y_off    += (target_y_off    - smoothed_y_off)    * smooth_alpha
        smoothed_rotation += (target_rotation - smoothed_rotation) * smooth_alpha

        # Update Eye Targets
        left_eye.target_pos[0]  = left_eye.base_x  + smoothed_x_off
        left_eye.target_pos[1]  = left_eye.base_y  + smoothed_y_off
        right_eye.target_pos[0] = right_eye.base_x + smoothed_x_off
        right_eye.target_pos[1] = right_eye.base_y + smoothed_y_off
        left_eye.target_rotation  = smoothed_rotation
        right_eye.target_rotation = smoothed_rotation

        # Blink Logic
        if time.time() > next_blink_time:
            blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            for eye in (left_eye, right_eye):
                eye.blink_state = "IDLE"
                eye.vy = 0
            left_eye.start_blink(blink_speed)
            right_eye.start_blink(blink_speed)
            last_blink_time = time.time()
            next_blink_time = time.time() + random.uniform(3.5, 7.0)

        # Tiny idle wander when nothing detected
        if target_x_off == 0 and target_y_off == 0:
            if random.random() < 0.05:
                left_eye.target_pos[0]  += random.uniform(-2, 2)
                left_eye.target_pos[1]  += random.uniform(-2, 2)
                right_eye.target_pos[0] += random.uniform(-2, 2)
                right_eye.target_pos[1] += random.uniform(-2, 2)

        # Physics Update
        left_eye.update()
        right_eye.update()

        # Render to displays
        if disp_l:
            img_l = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
            left_eye.draw(img_l)
            disp_l.image(img_l.convert("RGB"))

        if disp_r:
            img_r = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
            right_eye.draw(img_r)
            disp_r.image(img_r.convert("RGB"))

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    try:
        if picam2:
            picam2.stop()
            picam2.close()
            print("Camera closed.")
    except Exception as e:
        print(e)

    if stream_server:
        try:
            stream_server.shutdown()
            stream_server.server_close()
            print("MJPEG stream stopped.")
        except Exception as e:
            print(e)

    black = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    if disp_l:
        disp_l.image(black)
    if disp_r:
        disp_r.image(black)
    print("Displays cleared.")
