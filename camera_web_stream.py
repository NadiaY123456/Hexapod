#!/usr/bin/env python3
"""Serve the Raspberry Pi camera as an MJPEG stream on the local network."""

import argparse
import io
import socket
import socketserver
import sys
import threading
from http import server
from urllib.parse import urlsplit


INDEX_PAGE = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hexapod Camera</title>
  <style>
    html, body { height: 100%; margin: 0; background: #101214; color: #f4f5f6; }
    body { display: grid; grid-template-rows: auto 1fr; font-family: Arial, sans-serif; }
    header { padding: 10px 14px; background: #1c2024; border-bottom: 1px solid #343a40; }
    h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 0; }
    main { min-height: 0; display: grid; place-items: center; overflow: hidden; }
    img { display: block; width: 100%; height: 100%; object-fit: contain; }
  </style>
</head>
<body>
  <header><h1>Hexapod Camera</h1></header>
  <main><img src="/stream.mjpg" alt="Live camera stream"></main>
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    """Keep the newest encoded JPEG and notify every connected browser."""

    def __init__(self):
        super().__init__()
        self.frame = None
        self.sequence = 0
        self.condition = threading.Condition()

    def writable(self):
        return True

    def write(self, data):
        frame = bytes(data)
        with self.condition:
            self.frame = frame
            self.sequence += 1
            self.condition.notify_all()
        return len(frame)


class StreamingHandler(server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
            return

        if path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(INDEX_PAGE)))
            self.end_headers()
            self.wfile.write(INDEX_PAGE)
            return

        if path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=FRAME",
        )
        self.end_headers()

        output = self.server.output
        last_sequence = -1
        try:
            while True:
                with output.condition:
                    output.condition.wait_for(
                        lambda: output.sequence != last_sequence,
                        timeout=10.0,
                    )
                    if output.sequence == last_sequence or output.frame is None:
                        continue
                    frame = output.frame
                    last_sequence = output.sequence

                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, message, *args):
        print(f"Browser {self.client_address[0]}: {message % args}")


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="View the Raspberry Pi camera from a browser on the same network."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Address to listen on.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--camera", type=int, default=0, help="Picamera2 camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Stream width.")
    parser.add_argument("--height", type=int, default=720, help="Stream height.")
    parser.add_argument("--fps", type=float, default=20.0, help="Stream frame rate.")
    parser.add_argument("--hflip", action="store_true", help="Mirror horizontally.")
    parser.add_argument("--vflip", action="store_true", help="Mirror vertically.")
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.camera < 0:
        parser.error("camera must be zero or greater")
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if args.width % 2 or args.height % 2:
        parser.error("width and height must be even numbers")
    if not 1.0 <= args.fps <= 60.0:
        parser.error("fps must be between 1 and 60")
    return args


def local_ip():
    """Return the address selected for normal outbound LAN traffic."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "<pi-ip-address>"


def import_camera_stack():
    try:
        from libcamera import Transform
        from picamera2 import Picamera2
        from picamera2.encoders import MJPEGEncoder
        from picamera2.outputs import FileOutput
    except ImportError as error:
        print(
            "Picamera2 is not installed. On Raspberry Pi OS run:\n"
            "  sudo apt update\n"
            "  sudo apt install python3-picamera2\n\n"
            f"Original import error: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return Transform, Picamera2, MJPEGEncoder, FileOutput


def main():
    args = parse_args()
    Transform, Picamera2, MJPEGEncoder, FileOutput = import_camera_stack()

    camera = Picamera2(args.camera)
    transform = Transform(hflip=args.hflip, vflip=args.vflip)
    configuration = camera.create_video_configuration(
        main={"size": (args.width, args.height)},
        controls={"FrameRate": args.fps},
        transform=transform,
        buffer_count=6,
    )
    camera.configure(configuration)

    output = StreamingOutput()
    recording = False
    try:
        camera.start_recording(MJPEGEncoder(), FileOutput(output))
        recording = True

        with StreamingServer((args.host, args.port), StreamingHandler) as httpd:
            httpd.output = output
            display_host = local_ip() if args.host == "0.0.0.0" else args.host
            print(f"Camera stream: http://{display_host}:{args.port}/")
            print("Press Ctrl+C to stop.")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nStopping camera stream.")
    finally:
        if recording:
            camera.stop_recording()
        camera.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
