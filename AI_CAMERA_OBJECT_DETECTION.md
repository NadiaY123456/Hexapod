# Raspberry Pi AI Camera Object Detection

This repo includes `ai_camera_object_detection.py`, a Picamera2/IMX500 object
detection runner for the Raspberry Pi AI Camera.

## Install on the Raspberry Pi

```bash
sudo apt update
sudo apt full-upgrade
sudo apt install imx500-all python3-opencv python3-munkres
```

The `imx500-all` package installs the IMX500 camera firmware and example neural
network model files under `/usr/share/imx500-models/`.

## Run

```bash
python ai_camera_object_detection.py
```

For a terminal-only run:

```bash
python ai_camera_object_detection.py --headless --no-draw
```

Only print a specific class, such as people:

```bash
python ai_camera_object_detection.py --target person
```

Each terminal update is JSON, including object label, confidence, bounding box,
bounding-box width in pixels, center point, and normalized center offset from the camera frame. The offset is
useful later if you want the hexapod to turn toward an object:

```json
{"detections":[{"label":"person","confidence":0.82,"box":[100,80,220,300],"width_px":220,"center":[210,230],"offset":[-0.34,-0.12]}]}
```

## Estimate human distance

Run the human-specific entry point:

```bash
python3 human_distance.py
```

It uses the AI Camera's 4.74 mm focal length, 1.55 micrometre pixel pitch,
and 4056-pixel sensor width from the
[Raspberry Pi AI Camera product brief](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf).
The default assumed human width is 0.45 m:

```text
distance = assumed_width_m * focal_length_px / bounding_box_width_px
```

Override the assumed width after measuring the intended target person:

```bash
python3 human_distance.py --object-width-m 0.50
```

The estimate assumes the whole body width is visible and facing the camera.
Side-on poses, loose clothing, raised arms, partial occlusion, and detector-box
jitter can cause substantial error, so calibrate against known distances before
using the result for robot motion.

## Model

By default, the script uses:

```text
/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
```

You can pass a different IMX500 object detection model:

```bash
python ai_camera_object_detection.py --model /path/to/model.rpk
```
