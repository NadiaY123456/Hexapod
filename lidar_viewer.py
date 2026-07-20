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


def direction_label(angle: float) -> str:
    """Return an eight-way robot-relative label for a lidar angle."""
    directions = (
        "forward",
        "front-right",
        "right",
        "rear-right",
        "behind",
        "rear-left",
        "left",
        "front-left",
    )
    return directions[int(((angle % 360.0) + 22.5) // 45.0) % len(directions)]


def nearest_sector_mm(
    points: list[dict[str, float | int | str]],
    center_angle: float,
    half_width: float = 22.5,
) -> float | None:
    """Return the nearest range in one robot-relative angular sector."""
    distances = [
        float(point["distance_mm"])
        for point in points
        if abs(((float(point["angle"]) - center_angle + 180.0) % 360.0) - 180.0)
        <= half_width
    ]
    return min(distances, default=None)


def ensure_driver() -> None:
    """Confirm that the active Python environment contains the C1 driver."""
    try:
        from rplidarc1 import RPLidar  # noqa: F401
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
        self.points: dict[int, dict[str, float | int | str]] = {}
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
                "direction": direction_label(angle),
                "seen_at": now,
            }
            self.samples += 1
            self.last_sample = now

    def snapshot(self) -> dict[str, object]:
        now = time.monotonic()
        with self.lock:
            stale = [bucket for bucket, point in self.points.items() if now - float(point["seen_at"]) > 1.0]
            for bucket in stale:
                del self.points[bucket]
            points = [
                {key: value for key, value in point.items() if key != "seen_at"}
                for point in self.points.values()
            ]
            samples = self.samples
            last_sample = self.last_sample
            error = self.error
        nearest = min(points, key=lambda point: float(point["distance_mm"]), default=None)
        return {
            "device": self.device,
            "baudrate": BAUDRATE,
            "connected": bool(last_sample and time.monotonic() - last_sample < 2.0),
            "points": points,
            "point_count": len(points),
            "samples": samples,
            "samples_per_second": round(samples / max(now - self.started, 0.001)),
            "nearest_mm": nearest["distance_mm"] if nearest else None,
            "nearest_angle": nearest["angle"] if nearest else None,
            "nearest_direction": nearest["direction"] if nearest else None,
            "front_left_mm": nearest_sector_mm(points, 315.0),
            "forward_mm": nearest_sector_mm(points, 0.0),
            "front_right_mm": nearest_sector_mm(points, 45.0),
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
stats.textContent=`${data.device} · ${data.point_count} directions · ${data.samples_per_second} samples/s · nearest ${data.nearest_mm==null?'—':(data.nearest_mm/1000).toFixed(2)+' m '+data.nearest_direction+' ('+Number(data.nearest_angle).toFixed(1)+'°)'}`;if(data.error)statusEl.textContent=data.error}
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


async def print_summaries(
    state: ScanState,
    stop_event: threading.Event | None = None,
    interval: float = 0.5,
) -> None:
    while stop_event is None or not stop_event.is_set():
        await asyncio.sleep(interval)
        data = state.snapshot()
        nearest = data["nearest_mm"]
        nearest_text = (
            "none"
            if nearest is None
            else (
                f"{float(nearest) / 1000:.2f} m "
                f"{data['nearest_direction']} ({float(data['nearest_angle']):.1f}°)"
            )
        )
        def format_range(key: str) -> str:
            distance = data[key]
            return "--" if distance is None else f"{float(distance) / 1000:.2f}m"

        print(
            f"LIDAR: FL={format_range('front_left_mm')} "
            f"F={format_range('forward_mm')} "
            f"FR={format_range('front_right_mm')} | "
            f"{data['point_count']} directions, "
            f"{data['samples_per_second']} samples/s, nearest={nearest_text}",
            flush=True,
        )


async def run_lidar(
    device: str,
    state: ScanState,
    print_every_point: bool,
    stop_event: threading.Event | None = None,
) -> None:
    from rplidarc1 import RPLidar

    lidar = RPLidar(device, BAUDRATE)
    scan_task = asyncio.create_task(lidar.simple_scan(make_return_dict=False))
    last_item_time = time.monotonic()
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                item = await asyncio.wait_for(
                    lidar.output_queue.get(),
                    timeout=0.25 if stop_event is not None else 3.0,
                )
            except asyncio.TimeoutError:
                if stop_event is not None and stop_event.is_set():
                    break
                if time.monotonic() - last_item_time < 3.0:
                    continue
                raise
            last_item_time = time.monotonic()
            # The C1 driver represents a zero/invalid range as None. Those
            # samples are normal (for example, when no laser return is seen)
            # and should not stop the scan.
            try:
                angle = float(item["a_deg"])
                distance_value = item["d_mm"]
                quality = int(item["q"])
                if distance_value is None:
                    continue
                distance = float(distance_value)
            except (KeyError, TypeError, ValueError):
                continue
            state.add(angle, distance, quality)
            if print_every_point:
                print(
                    f"direction={direction_label(angle):>11}  angle={angle:7.2f}°  "
                    f"distance={distance:8.1f} mm  quality={quality}",
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


class LidarWebViewer:
    """Run the C1 scanner, terminal telemetry, and browser view in the background."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        device: str | None = None,
        max_range_m: float = 12.0,
        print_interval: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.requested_device = device
        self.max_range_m = max_range_m
        self.print_interval = print_interval
        self.state: ScanState | None = None
        self.server: ThreadingHTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self.scan_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    @property
    def url(self) -> str:
        display_host = local_ip() if self.host == "0.0.0.0" else self.host
        return f"http://{display_host}:{self.port}/"

    def start(self) -> str:
        if self.scan_thread is not None:
            return self.url

        try:
            ensure_driver()
            device = find_device(self.requested_device)
        except SystemExit as error:
            raise RuntimeError(str(error)) from error

        self.state = ScanState(self.max_range_m)
        self.state.device = device
        try:
            self.server = ThreadingHTTPServer(
                (self.host, self.port),
                make_handler(self.state),
            )
        except OSError as error:
            self.server = None
            raise RuntimeError(
                f"Could not start lidar viewer on port {self.port}: {error}"
            ) from error

        self.stop_event.clear()
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            name="lidar-web-viewer",
            daemon=True,
        )
        self.scan_thread = threading.Thread(
            target=self._run_scan,
            name="lidar-c1-reader",
            daemon=True,
        )
        self.server_thread.start()
        self.scan_thread.start()
        print(f"Connecting to RPLIDAR C1 on {device} at {BAUDRATE} baud")
        return self.url

    def _run_scan(self) -> None:
        try:
            asyncio.run(self._scan_main())
        except Exception as error:
            if self.state is not None:
                self.state.error = str(error)
            print(f"LIDAR ERROR: {error}", file=sys.stderr, flush=True)

    async def _scan_main(self) -> None:
        tasks = [
            asyncio.create_task(
                run_lidar(
                    self.state.device,
                    self.state,
                    False,
                    self.stop_event,
                )
            )
        ]
        if self.print_interval > 0.0:
            tasks.append(
                asyncio.create_task(
                    print_summaries(
                        self.state,
                        self.stop_event,
                        self.print_interval,
                    )
                )
            )
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread is not None:
            self.server_thread.join(timeout=3.0)
        if self.scan_thread is not None:
            self.scan_thread.join(timeout=4.0)

        self.server = None
        self.server_thread = None
        self.scan_thread = None


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
