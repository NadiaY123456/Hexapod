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
center point, and normalized center offset from the camera frame. The offset is
useful later if you want the hexapod to turn toward an object:

```json
{"detections":[{"label":"person","confidence":0.82,"box":[100,80,220,300],"center":[210,230],"offset":[-0.34,-0.12]}]}
```

## Model

By default, the script uses:

```text
/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
```

You can pass a different IMX500 object detection model:

```bash
python ai_camera_object_detection.py --model /path/to/model.rpk
```
