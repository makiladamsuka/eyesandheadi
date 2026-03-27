# NEma — Tracking Eyes (Pi)

Face-tracking animated eyes on dual ST7735 SPI displays driven by a Raspberry Pi and Picamera2.

## Features
- YuNet face detection (OpenCV) for real-time face tracking
- Dual SPI ST7735 (128×160) display output — one eye per screen
- Expressive eye animations: blink, emotions (happy, sad, angry, surprised, suspicious, sleepy)
- MJPEG stream on port `8080` for headless SSH monitoring (`http://<pi-ip>:8080/stream`)

## Hardware
| Pin | Function |
|-----|----------|
| CE1 / D24 / D25 | Left display CS / DC / RST (SPI0) |
| D21 / D20 / D19 | Right display CLK / MOSI / MISO (SPI1) |
| D18 / D23 / D27 | Right display CS / DC / RST |

## Setup (on Pi)

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-pil python3-numpy
pip3 install adafruit-circuitpython-rgb-display adafruit-blinka
```

Download the YuNet face model:
```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Run

```bash
python3 tracking_eyes.py
```

## Configuration

Edit the top of `tracking_eyes.py` to adjust:
- `CAMERA_ROTATE_180` — flip camera if mounted upside-down
- `STREAM_PORT` — MJPEG stream port (default `8080`)
- `MAX_X_OFFSET` / `MAX_Y_OFFSET` — how far eyes move to follow a face
- `FACE_ROLL_MULT` — eye tilt sensitivity to face roll
