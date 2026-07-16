# Hexapod Camera Web Stream

`camera_web_stream.py` publishes the Raspberry Pi camera as an MJPEG stream.
Any browser on the same local network can view it without installing an app.

## Pi setup

Picamera2 is normally installed by the Raspberry Pi AI Camera packages. If it
is missing, install the Raspberry Pi OS package:

```bash
sudo apt update
sudo apt install python3-picamera2
```

Run the server from the repository:

```bash
python3 camera_web_stream.py
```

The terminal prints an address similar to:

```text
Camera stream: http://192.168.1.42:8000/
```

Open that address on a computer or phone connected to the same network. The
Pi and viewing device do not need internet access, only a shared LAN.

Useful options:

```bash
python3 camera_web_stream.py --width 640 --height 480 --fps 15
python3 camera_web_stream.py --port 8080
python3 camera_web_stream.py --hflip --vflip
```

Only one process can control the camera at a time. Stop `follow_person.py`,
`follow_cat.py`, or another Picamera2 program before starting the web stream.

The stream has no login protection. Run it only on a trusted local network.

## Troubleshooting

Find the Pi's address manually with:

```bash
hostname -I
```

If the page does not open, verify that both devices are on the same Wi-Fi or
Ethernet network and that client isolation is disabled on the router. Test the
server locally on the Pi with:

```bash
curl -I http://127.0.0.1:8000/index.html
```

If Picamera2 says the camera is busy, stop the other process using it or reboot
the Pi after confirming no camera program is configured to start automatically.
