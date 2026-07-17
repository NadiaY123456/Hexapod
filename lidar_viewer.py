#!/usr/bin/env python3
"""RPLIDAR C1 console monitor and live browser view.

Run this file directly on the Raspberry Pi from the Python environment where
the C1-specific ``rplidarc1`` driver is installed. No ROS, Flask, or changes
to the robot code are needed.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


BAUDRATE = 460800
DEFAULT_WEB_PORT = 8000


def ensure_driver() -> None:
    """Confirm that the active Python environment contains the C1 driver."""
    try:
        from scanner import RPLidar  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "The RPLIDAR C1 driver is not installed in the active Python environment.\n"
            "Activate your virtual environment, then run:\n"
            f"  {sys.executable} -m pip install rplidarc1==0.1.3"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print RPLIDAR C1 readings and display a live 360-degree view."
    )
    parser.add_argument(
        "--device",
        help="Serial device (auto-detects /dev/ttyUSB* and /dev/ttyACM* by default).",
    )
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--max-range-m", type=float, default=12.0)
    parser.add_argument(
        "--print-every-point",
        action="store_true",
        help="Print every sample (very noisy); default output is a twice-per-second summary.",
    )
    args = parser.parse_args()
    if not 1 <= args.web_port <= 65535:
        parser.error("--web-port must be between 1 and 65535")
    if args.max_range_m <= 0:
        parser.error("--max-range-m must be positive")
    return args


def find_device(requested: str | None) -> str:
    if requested:
        candidates = [requested]
    else:
        candidates = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))

    if not candidates:
        raise SystemExit(
            "No USB serial device was found. Connect the lidar and run `ls /dev/ttyUSB*`."
        )
    if len(candidates) > 1 and not requested:
        choices = ", ".join(candidates)
        raise SystemExit(
            f"More than one serial device was found: {choices}\n"
            "Run again with --device /dev/ttyUSB0 (using the correct device)."
        )

    device = candidates[0]
    if not os.access(device, os.R_OK | os.W_OK):
        raise SystemExit(
            f"Your user cannot access {device}. Run:\n"
            "  sudo usermod -aG dialout $USER\n"
            "Then log out and back in (or reboot) before trying again."
        )
    return device


class ScanState:
    def __init__(self, max_range_m: float) -> None:
        self.lock = threading.Lock()
        self.points: dict[int, dict[str, float | int]] = {}
        self.samples = 0
        self.started = time.monotonic()
        self.last_sample = 0.0
        self.error: str | None = None
        self.device = ""
        self.max_range_mm = max_range_m * 1000.0

    def add(self, angle: float, distance: float, quality: int) -> None:
        if not (0.0 < distance <= self.max_range_mm):
            return
        bucket = round(angle) % 360
        now = time.monotonic()
        with self.lock:
            self.points[bucket] = {
                "angle": round(angle % 360.0, 2),
                "distance_mm": round(distance, 1),
                "quality": quality,
            }
            self.samples += 1
            self.last_sample = now

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            points = list(self.points.values())
            samples = self.samples
            last_sample = self.last_sample
            error = self.error
        distances = [float(point["distance_mm"]) for point in points]
        return {
            "device": self.device,
            "baudrate": BAUDRATE,
            "connected": bool(last_sample and time.monotonic() - last_sample < 2.0),
            "points": points,
            "point_count": len(points),
            "samples": samples,
            "samples_per_second": round(samples / max(time.monotonic() - self.started, 0.001)),
            "nearest_mm": min(distances) if distances else None,
            "max_range_mm": self.max_range_mm,
            "error": error,
        }


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>RPLIDAR C1 Live View</title>
<style>
body{margin:0;background:#091017;color:#dbeafe;font:15px system-ui,sans-serif;text-align:center}
header{padding:16px} h1{margin:0 0 6px;font-size:22px}.ok{color:#4ade80}.bad{color:#fb7185}
canvas{width:min(86vw,86vh);height:min(86vw,86vh);max-width:760px;max-height:760px;background:#07131c;border:1px solid #284154;border-radius:50%}
#stats{color:#93c5fd;margin:8px}.hint{color:#7890a5;font-size:13px}
</style></head><body><header><h1>RPLIDAR C1</h1><div id="status">Waiting for scan data…</div>
<div id="stats"></div><div class="hint">Forward is up · distances are shown in metres</div></header>
<canvas id="view" width="800" height="800"></canvas>
<script>
const c=document.querySelector('#view'),x=c.getContext('2d'),statusEl=document.querySelector('#status'),stats=document.querySelector('#stats');
function draw(data){const w=c.width,h=c.height,cx=w/2,cy=h/2,r=w*.46,max=data.max_range_mm;x.clearRect(0,0,w,h);x.strokeStyle='#1d394b';x.fillStyle='#7c93a5';x.font='13px system-ui';x.textAlign='center';
for(let i=1;i<=4;i++){x.beginPath();x.arc(cx,cy,r*i/4,0,Math.PI*2);x.stroke();x.fillText((max*i/4000)+'m',cx+4,cy-r*i/4+15)}
x.strokeStyle='#284154';x.beginPath();x.moveTo(cx-r,cy);x.lineTo(cx+r,cy);x.moveTo(cx,cy-r);x.lineTo(cx,cy+r);x.stroke();
x.fillStyle='#22d3ee';for(const p of data.points){const a=p.angle*Math.PI/180,d=Math.min(p.distance_mm/max,1)*r;x.beginPath();x.arc(cx+Math.sin(a)*d,cy-Math.cos(a)*d,2.5,0,Math.PI*2);x.fill()}
x.fillStyle='#fbbf24';x.beginPath();x.arc(cx,cy,7,0,Math.PI*2);x.fill();
statusEl.className=data.connected?'ok':'bad';statusEl.textContent=data.connected?'● Live scan':'● Waiting for lidar data';
stats.textContent=`${data.device} · ${data.point_count} directions · ${data.samples_per_second} samples/s · nearest ${data.nearest_mm==null?'—':(data.nearest_mm/1000).toFixed(2)+' m'}`;if(data.error)statusEl.textContent=data.error}
async function update(){try{const response=await fetch('/scan',{cache:'no-store'});draw(await response.json())}catch(e){statusEl.className='bad';statusEl.textContent='Viewer lost connection'}}setInterval(update,100);update();
</script></body></html>"""


def make_handler(state: ScanState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif self.path.split("?", 1)[0] == "/scan":
                body = json.dumps(state.snapshot(), separators=(",", ":")).encode()
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def local_ip() -> str:
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("10.255.255.255", 1))
        return connection.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        connection.close()


async def print_summaries(state: ScanState) -> None:
    while True:
        await asyncio.sleep(0.5)
        data = state.snapshot()
        nearest = data["nearest_mm"]
        nearest_text = "none" if nearest is None else f"{float(nearest) / 1000:.2f} m"
        print(
            f"LIDAR: {data['point_count']} directions, "
            f"{data['samples_per_second']} samples/s, nearest={nearest_text}",
            flush=True,
        )


async def run_lidar(device: str, state: ScanState, print_every_point: bool) -> None:
    from scanner import RPLidar

    lidar = RPLidar(device, BAUDRATE)
    scan_task = asyncio.create_task(lidar.simple_scan(make_return_dict=False))
    try:
        while True:
            item = await asyncio.wait_for(lidar.output_queue.get(), timeout=3.0)
            angle = float(item["a_deg"])
            distance = float(item["d_mm"])
            quality = int(item["q"])
            state.add(angle, distance, quality)
            if print_every_point:
                print(
                    f"angle={angle:7.2f}°  distance={distance:8.1f} mm  quality={quality}",
                    flush=True,
                )
    except asyncio.TimeoutError as error:
        state.error = "No scan data received for 3 seconds"
        raise RuntimeError(
            "The serial port opened, but no scan data arrived. Check that this is the "
            "correct device and that the lidar has adequate USB power."
        ) from error
    finally:
        lidar.stop_event.set()
        try:
            await asyncio.wait_for(scan_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            scan_task.cancel()
        try:
            lidar.shutdown()
        except Exception:
            lidar.reset()


async def async_main(args: argparse.Namespace, state: ScanState) -> None:
    tasks = [asyncio.create_task(run_lidar(state.device, state, args.print_every_point))]
    if not args.print_every_point:
        tasks.append(asyncio.create_task(print_summaries(state)))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> int:
    ensure_driver()
    args = parse_args()
    state = ScanState(args.max_range_m)
    state.device = find_device(args.device)

    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.web_port), make_handler(state))
    except OSError as error:
        raise SystemExit(f"Could not start web viewer on port {args.web_port}: {error}") from error
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Connecting to RPLIDAR C1 on {state.device} at {BAUDRATE} baud")
    print(f"Live view on this Pi: http://127.0.0.1:{args.web_port}")
    print(f"Live view from another device: http://{local_ip()}:{args.web_port}")
    print("Press Ctrl+C to stop the scanner and viewer.")
    try:
        asyncio.run(async_main(args, state))
    except KeyboardInterrupt:
        print("\nStopping lidar...")
    except Exception as error:
        print(f"LIDAR ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
