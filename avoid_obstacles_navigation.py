"""Navigate around obstacles using the AI Camera and RPLIDAR C1.

The robot stands up when the program starts, but does not walk until W is
pressed. W starts/resumes forward walking, S stops all walking while holding
the standing pose, P disengages the servos, and O stands up again without
walking. Camera monitoring continues throughout. Ctrl+C exits the whole
program.

The camera identifies objects while the RPLIDAR C1 scores body-width corridors
in front of the robot. The gait steers forward through a safe offset corridor
and turns in place only when no safe forward corridor exists or an obstacle is
inside the emergency range. A browser dashboard shows both sensors live.
"""

import argparse
import asyncio
import json
import math
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_camera_object_detection import (
    AI_CAMERA_FOCAL_LENGTH_MM,
    AI_CAMERA_PIXEL_PITCH_UM,
    AI_CAMERA_SENSOR_WIDTH_PX,
    DEFAULT_HUMAN_WIDTH_M,
    DEFAULT_MODEL,
    Detection,
    estimate_distance_m,
    import_camera_stack,
    load_labels,
    rectangle_to_box,
)


DEFAULT_CAMERA_BUFFER_COUNT = 6
DEFAULT_WEB_PORT = 8000
DEFAULT_BODY_HALF_WIDTH_M = 0.20
DEFAULT_NAVIGATION_LOOKAHEAD_M = 2.0
DEFAULT_EMERGENCY_DISTANCE_M = 0.38
DEFAULT_PATH_MARGIN_M = 0.12
CAMERA_READING_MAX_AGE = 1.0

# Approximate real widths improve monocular distance estimates. Unknown labels
# still use the human-width fallback, and --object-width-m overrides this map.
OBJECT_WIDTHS_M = {
    "person": DEFAULT_HUMAN_WIDTH_M,
    "bicycle": 0.65,
    "car": 1.80,
    "motorcycle": 0.80,
    "bus": 2.50,
    "truck": 2.50,
    "cat": 0.25,
    "dog": 0.35,
    "chair": 0.50,
    "couch": 1.80,
    "potted plant": 0.45,
    "dining table": 1.20,
}


def signed_lidar_angle(angle):
    """Convert 0..360 degrees to -180..180, with forward at zero."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


def path_clearance_mm(points, heading_deg, body_half_width_mm, lookahead_mm):
    """Return unobstructed distance along a body-width corridor."""
    clearance = lookahead_mm
    for point in points:
        relative = math.radians(point["relative_angle_deg"] - heading_deg)
        distance = point["distance_mm"]
        forward = distance * math.cos(relative)
        lateral = distance * math.sin(relative)
        if 0.0 < forward <= lookahead_mm and abs(lateral) <= body_half_width_mm:
            clearance = min(clearance, forward)
    return clearance


def choose_lidar_path(
    points,
    stop_distance_mm,
    forward_angle_deg,
    body_half_width_mm,
    lookahead_mm,
    emergency_distance_mm,
    path_margin_mm,
):
    """Choose a clear heading, or request an in-place turn when none fits."""
    heading_step = 5.0
    headings = []
    heading = -forward_angle_deg
    while heading <= forward_angle_deg + 0.001:
        headings.append(round(heading, 3))
        heading += heading_step
    if 0.0 not in headings:
        headings.append(0.0)

    clearances = {
        candidate: path_clearance_mm(
            points,
            candidate,
            body_half_width_mm,
            lookahead_mm,
        )
        for candidate in headings
    }
    center_clearance = clearances[0.0]
    center_penalty_per_degree = path_margin_mm / max(forward_angle_deg, 1.0)
    best_heading = max(
        headings,
        key=lambda candidate: (
            clearances[candidate] - abs(candidate) * center_penalty_per_degree,
            -abs(candidate),
        ),
    )
    best_clearance = clearances[best_heading]

    emergency = any(
        point["distance_mm"] <= emergency_distance_mm
        and abs(point["relative_angle_deg"]) <= forward_angle_deg
        for point in points
    )
    center_blocked = center_clearance <= stop_distance_mm
    path_available = best_clearance > stop_distance_mm + path_margin_mm
    worthwhile_detour = best_clearance > center_clearance + path_margin_mm
    steer_forward = (
        not emergency
        and path_available
        and abs(best_heading) >= heading_step
        and (center_blocked or worthwhile_detour)
    )
    must_turn = emergency or (center_blocked and not steer_forward)
    steering = max(-1.0, min(1.0, best_heading / forward_angle_deg))

    return {
        "center_clearance_mm": center_clearance,
        "best_clearance_mm": best_clearance,
        "target_heading_deg": best_heading,
        "steering": steering if steer_forward else 0.0,
        "steer_forward": steer_forward,
        "must_turn": must_turn,
        "emergency": emergency,
    }


class LidarObstacleMonitor:
    """Continuously read the C1 in a background thread without delaying gait."""

    def __init__(
        self,
        device,
        stop_distance_m,
        forward_angle_deg,
        forward_offset_deg,
        body_half_width_m,
        lookahead_m,
        emergency_distance_m,
        path_margin_m,
    ):
        self.device = device
        self.stop_distance_mm = stop_distance_m * 1000.0
        self.forward_angle_deg = forward_angle_deg
        self.forward_offset_deg = forward_offset_deg
        self.body_half_width_mm = body_half_width_m * 1000.0
        self.lookahead_mm = lookahead_m * 1000.0
        self.emergency_distance_mm = emergency_distance_m * 1000.0
        self.path_margin_mm = path_margin_m * 1000.0
        self.points = {}
        self.lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.data_ready = threading.Event()
        self.thread = None
        self.last_packet = 0.0
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        if not self.data_ready.wait(timeout=8.0):
            raise RuntimeError(
                f"RPLIDAR C1 on {self.device} did not produce data within 8 seconds."
            )
        if self.error is not None:
            raise RuntimeError(f"RPLIDAR C1 failed: {self.error}") from self.error

    def stop(self):
        self.stop_requested.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def _thread_main(self):
        try:
            asyncio.run(self._scan())
        except Exception as error:
            self.error = error
            self.data_ready.set()

    async def _scan(self):
        from rplidarc1 import RPLidar

        lidar = RPLidar(self.device, 460800)
        scan_task = asyncio.create_task(lidar.simple_scan(make_return_dict=False))
        try:
            while not self.stop_requested.is_set():
                try:
                    item = await asyncio.wait_for(lidar.output_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                now = time.monotonic()
                self.last_packet = now
                self.data_ready.set()
                try:
                    angle = float(item["a_deg"]) % 360.0
                    distance_value = item["d_mm"]
                    if distance_value is None:
                        continue
                    distance_mm = float(distance_value)
                    quality = int(item["q"])
                except (KeyError, TypeError, ValueError):
                    continue
                if distance_mm <= 0.0:
                    continue
                with self.lock:
                    self.points[round(angle) % 360] = {
                        "angle": angle,
                        "distance_mm": distance_mm,
                        "quality": quality,
                        "seen_at": now,
                    }
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

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            points = [
                point.copy()
                for point in self.points.values()
                if now - point["seen_at"] <= 1.0
            ]
        live = self.error is None and now - self.last_packet <= 1.5

        def relative_angle(point):
            return signed_lidar_angle(point["angle"] - self.forward_offset_deg)

        relative_points = [
            {
                "angle": round(point["angle"], 2),
                "relative_angle_deg": round(relative_angle(point), 2),
                "distance_mm": round(point["distance_mm"], 1),
                "quality": point["quality"],
            }
            for point in points
        ]
        forward = [
            point
            for point in relative_points
            if abs(point["relative_angle_deg"]) <= self.forward_angle_deg
        ]
        blocking_points = [
            point
            for point in forward
            if point["distance_mm"] <= self.stop_distance_mm
        ]
        nearest = min(blocking_points, key=lambda point: point["distance_mm"], default=None)

        path = choose_lidar_path(
            relative_points,
            self.stop_distance_mm,
            self.forward_angle_deg,
            self.body_half_width_mm,
            self.lookahead_mm,
            self.emergency_distance_mm,
            self.path_margin_mm,
        )

        def side_clearance(side):
            if side == "left":
                side_points = [
                    point for point in relative_points
                    if -120.0 <= point["relative_angle_deg"] <= -15.0
                ]
            else:
                side_points = [
                    point for point in relative_points
                    if 15.0 <= point["relative_angle_deg"] <= 120.0
                ]
            return min(
                (point["distance_mm"] for point in side_points),
                default=12000.0,
            )

        turn_direction = None
        if path["must_turn"] or path["steer_forward"]:
            target_heading = path["target_heading_deg"]
            if target_heading < -5.0:
                turn_direction = "left"
            elif target_heading > 5.0:
                turn_direction = "right"
            else:
                left_clearance = side_clearance("left")
                right_clearance = side_clearance("right")
                if left_clearance > right_clearance:
                    turn_direction = "left"
                elif right_clearance > left_clearance:
                    turn_direction = "right"

        return {
            "live": live,
            "blocking": path["center_clearance_mm"] <= self.stop_distance_mm,
            "must_turn": path["must_turn"],
            "steer_forward": path["steer_forward"],
            "steering": round(path["steering"], 3),
            "target_heading_deg": round(path["target_heading_deg"], 1),
            "center_clearance_m": round(path["center_clearance_mm"] / 1000.0, 3),
            "best_clearance_m": round(path["best_clearance_mm"] / 1000.0, 3),
            "emergency": path["emergency"],
            "nearest_m": (
                round(nearest["distance_mm"] / 1000.0, 3)
                if nearest is not None else None
            ),
            "nearest_angle_deg": (
                round(nearest["relative_angle_deg"], 1)
                if nearest is not None else None
            ),
            "turn_direction": turn_direction,
            "forward_half_angle_deg": self.forward_angle_deg,
            "body_half_width_m": round(self.body_half_width_mm / 1000.0, 3),
            "lookahead_m": round(self.lookahead_mm / 1000.0, 3),
            "points": relative_points,
            "error": str(self.error) if self.error is not None else None,
        }


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hexapod Navigation</title>
<style>
body{margin:0;background:#0b1116;color:#e5edf3;font:14px system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid #293640;display:flex;gap:16px;align-items:center}
h1{font-size:18px;margin:0}.status{color:#8bd5a1}main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}
section{min-width:0}h2{font-size:14px;margin:0 0 8px;color:#a9bac7}img,canvas{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#05090c;border:1px solid #293640}
pre{white-space:pre-wrap;color:#b8c7d1;margin:10px 0 0}@media(max-width:800px){main{grid-template-columns:1fr}}
</style></head><body><header><h1>Hexapod Navigation</h1><div id="status" class="status">Starting</div></header>
<main><section><h2>AI Camera</h2><img id="camera" alt="Annotated camera view"></section>
<section><h2>RPLIDAR C1</h2><canvas id="lidar" width="800" height="600"></canvas><pre id="details"></pre></section></main>
<script>
const camera=document.querySelector('#camera'),canvas=document.querySelector('#lidar'),ctx=canvas.getContext('2d');
const statusEl=document.querySelector('#status'),details=document.querySelector('#details');
function draw(data){const lidar=data.lidar||{},pts=lidar.points||[],w=canvas.width,h=canvas.height,cx=w/2,cy=h/2,r=Math.min(w,h)*.44,max=(lidar.lookahead_m||2)*1000;
ctx.clearRect(0,0,w,h);ctx.strokeStyle='#29404f';ctx.fillStyle='#8aa0ae';ctx.font='13px system-ui';ctx.textAlign='center';
for(let i=1;i<=4;i++){ctx.beginPath();ctx.arc(cx,cy,r*i/4,0,Math.PI*2);ctx.stroke();ctx.fillText((max*i/4000).toFixed(1)+'m',cx+4,cy-r*i/4+15)}
ctx.beginPath();ctx.moveTo(cx-r,cy);ctx.lineTo(cx+r,cy);ctx.moveTo(cx,cy-r);ctx.lineTo(cx,cy+r);ctx.stroke();
ctx.fillStyle='#44c7e8';for(const p of pts){const a=p.relative_angle_deg*Math.PI/180,d=Math.min(p.distance_mm/max,1)*r,x=cx+Math.sin(a)*d,y=cy-Math.cos(a)*d;ctx.beginPath();ctx.arc(x,y,2.7,0,Math.PI*2);ctx.fill()}
const target=(lidar.target_heading_deg||0)*Math.PI/180;ctx.strokeStyle=lidar.must_turn?'#fb7185':'#fbbf24';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.sin(target)*r*.8,cy-Math.cos(target)*r*.8);ctx.stroke();ctx.lineWidth=1;
statusEl.textContent=(data.motion||'unknown')+' | camera '+(data.camera_live?'live':'stale')+' | lidar '+(lidar.live?'live':'stale');
details.textContent=`motion: ${data.motion||'unknown'}\nsteering: ${Number(data.steering||0).toFixed(2)}\ntarget: ${Number(lidar.target_heading_deg||0).toFixed(1)} deg\ncenter clearance: ${lidar.center_clearance_m==null?'--':lidar.center_clearance_m+' m'}\nbest clearance: ${lidar.best_clearance_m==null?'--':lidar.best_clearance_m+' m'}\ncamera detections: ${(data.detections||[]).length}`;}
async function update(){try{const response=await fetch('/state',{cache:'no-store'});draw(await response.json());camera.src='/camera.jpg?t='+Date.now()}catch(error){statusEl.textContent='Dashboard disconnected'}}
setInterval(update,200);update();
</script></body></html>"""


def local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        try:
            connection.connect(("10.255.255.255", 1))
            return connection.getsockname()[0]
        except OSError:
            return "127.0.0.1"


class AvoidanceDashboard:
    """Serve the latest annotated camera frame and navigation state."""

    def __init__(self, port):
        self.port = port
        self.lock = threading.Lock()
        self.state = {"motion": "starting", "camera_live": False, "lidar": {}}
        self.camera_frame = None
        self.server = None
        self.thread = None

    def start(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    body = DASHBOARD_HTML.encode("utf-8")
                    content_type = "text/html; charset=utf-8"
                elif path == "/state":
                    with dashboard.lock:
                        body = json.dumps(dashboard.state, separators=(",", ":")).encode()
                    content_type = "application/json"
                elif path == "/camera.jpg":
                    with dashboard.lock:
                        body = dashboard.camera_frame
                    if body is None:
                        self.send_error(503, "Camera frame is not ready")
                        return
                    content_type = "image/jpeg"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://{local_ip()}:{self.port}"

    def update_state(self, state):
        with self.lock:
            self.state = state

    def update_camera(self, frame):
        with self.lock:
            self.camera_frame = frame

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.server = None
        self.thread = None


class KeyboardControls:
    """Read terminal keys without blocking and restore the terminal on exit."""

    def __init__(self):
        self.fd = None
        self.original_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.original_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        else:
            print("Warning: stdin is not a terminal; W/S/P/O controls are disabled.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_settings)

    def read_keys(self):
        keys = []
        if self.fd is None:
            return keys
        while select.select([sys.stdin], [], [], 0)[0]:
            keys.append(os.read(self.fd, 1).lower())
        return keys


class ObstacleStatusPrinter:
    """Print camera and lidar obstacle state as JSON."""

    def __init__(self, interval):
        self.interval = interval
        self.last_print = 0.0

    def print(self, detections, frame_size, args, blocking, motion, lidar_status):
        now = time.monotonic()
        if now - self.last_print < self.interval:
            return
        self.last_print = now
        frame_w, frame_h = frame_size
        items = []
        for detection in detections:
            distance_m = obstacle_distance(detection, frame_w, args)
            center_x, center_y = detection.center
            in_forward_view = camera_detection_is_forward(detection, frame_w, args)
            item = {
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "box": [int(value) for value in detection.box],
                "center": [int(center_x), int(center_y)],
                "offset": [
                    round((center_x - frame_w / 2) / (frame_w / 2), 3),
                    round((center_y - frame_h / 2) / (frame_h / 2), 3),
                ],
                "in_forward_view": in_forward_view,
                "nearby": in_forward_view
                and distance_m is not None
                and distance_m <= args.stop_distance_m,
            }
            if distance_m is not None:
                item["distance"] = {
                    "meters": round(distance_m, 2),
                    "feet": round(distance_m * 3.28084, 2),
                    "assumed_width_m": assumed_object_width(detection, args),
                }
            items.append(item)
        payload = {
            "sees_anything": bool(items),
            "nearby_obstacle": blocking,
            "motion": motion,
            "detections": items,
            "lidar": {
                key: value
                for key, value in lidar_status.items()
                if key != "points"
            },
        }
        print(json.dumps(payload, separators=(",", ":")))


def get_args():
    parser = argparse.ArgumentParser(
        description="Walk forward and turn until no detected nearby object remains."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--labels", help="Optional labels file, one label per line.")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--fps", type=int, help="Override camera inference frame rate.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--object-width-m",
        type=float,
        help=(
            "Override the assumed width for every camera detection. By default, "
            "known labels use per-class widths and unknown labels use 0.45m."
        ),
    )
    parser.add_argument(
        "--stop-distance-m",
        type=float,
        default=0.75,
        help="Avoid any detected object at or below this distance (default: 0.75m).",
    )
    parser.add_argument(
        "--camera-forward-half-width",
        type=float,
        default=0.45,
        help="Camera center corridor as a fraction of half-frame width (default: 0.45).",
    )
    parser.add_argument(
        "--lidar-device",
        default="/dev/ttyUSB0",
        help="RPLIDAR C1 serial device (default: /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--lidar-stop-distance-m",
        type=float,
        default=0.75,
        help="Stop for a forward lidar return at or below this range (default: 0.75m).",
    )
    parser.add_argument(
        "--lidar-forward-angle-deg",
        type=float,
        default=35.0,
        help="Half-width of the forward lidar corridor in degrees (default: 35).",
    )
    parser.add_argument(
        "--lidar-forward-offset-deg",
        type=float,
        default=0.0,
        help="Raw lidar angle that points forward on the mounted robot (default: 0).",
    )
    parser.add_argument(
        "--body-half-width-m",
        type=float,
        default=DEFAULT_BODY_HALF_WIDTH_M,
        help="Robot half-width including clearance margin (default: 0.20m).",
    )
    parser.add_argument(
        "--navigation-lookahead-m",
        type=float,
        default=DEFAULT_NAVIGATION_LOOKAHEAD_M,
        help="Distance used to score candidate paths (default: 2.0m).",
    )
    parser.add_argument(
        "--emergency-distance-m",
        type=float,
        default=DEFAULT_EMERGENCY_DISTANCE_M,
        help="Always turn in place inside this range (default: 0.38m).",
    )
    parser.add_argument(
        "--path-margin-m",
        type=float,
        default=DEFAULT_PATH_MARGIN_M,
        help="Extra clearance required before choosing a detour (default: 0.12m).",
    )
    parser.add_argument(
        "--clear-frames",
        type=int,
        default=3,
        help="Consecutive clear frames required before walking forward again (default: 3).",
    )
    parser.add_argument(
        "--turn-direction",
        choices=("left", "right"),
        default="left",
        help="Side used to go around obstacles (default: left).",
    )
    parser.add_argument("--turn-scale", type=float, default=0.50)
    parser.add_argument("--walk-steps", type=int, default=7)
    parser.add_argument("--walk-frame-delay", type=float, default=0.022)
    parser.add_argument("--print-interval", type=float, default=0.25)
    parser.add_argument(
        "--camera-buffers",
        type=int,
        default=DEFAULT_CAMERA_BUFFER_COUNT,
        help="Picamera2 request buffers (default: 6).",
    )
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--focal-length-mm", type=float, default=AI_CAMERA_FOCAL_LENGTH_MM)
    parser.add_argument("--pixel-pitch-um", type=float, default=AI_CAMERA_PIXEL_PITCH_UM)
    parser.add_argument("--sensor-width-px", type=int, default=AI_CAMERA_SENSOR_WIDTH_PX)
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bbox-order", choices=("yx", "xy"))
    parser.add_argument("--postprocess", choices=("", "nanodet"), default=None)
    parser.add_argument("--preserve-aspect-ratio", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if (
        args.stop_distance_m <= 0.0
        or args.lidar_stop_distance_m <= 0.0
        or args.body_half_width_m <= 0.0
        or args.navigation_lookahead_m <= 0.0
        or args.emergency_distance_m <= 0.0
        or args.path_margin_m <= 0.0
    ):
        parser.error("camera, lidar, and path measurements must be positive")
    if args.object_width_m is not None and args.object_width_m <= 0.0:
        parser.error("object-width-m must be positive")
    if not 0.0 < args.camera_forward_half_width <= 1.0:
        parser.error("camera-forward-half-width must be within 0..1")
    if not 0.0 < args.lidar_forward_angle_deg < 90.0:
        parser.error("lidar-forward-angle-deg must be between 0 and 90")
    if args.clear_frames < 1 or args.walk_steps < 1:
        parser.error("clear-frames and walk-steps must be at least 1")
    if args.walk_frame_delay < 0.0 or args.print_interval < 0.0:
        parser.error("walk-frame-delay and print-interval cannot be negative")
    if not 0.0 < args.turn_scale <= 1.0:
        parser.error("turn-scale must be within 0..1")
    if args.focal_length_mm <= 0.0 or args.pixel_pitch_um <= 0.0:
        parser.error("camera measurements must be positive")
    if args.sensor_width_px <= 0:
        parser.error("sensor-width-px must be positive")
    if args.camera_buffers < 3:
        parser.error("camera-buffers must be at least 3")
    if not 1 <= args.web_port <= 65535:
        parser.error("web-port must be between 1 and 65535")
    if args.emergency_distance_m >= args.lidar_stop_distance_m:
        parser.error("emergency-distance-m must be less than lidar-stop-distance-m")
    return args


def assumed_object_width(detection, args):
    if args.object_width_m is not None:
        return args.object_width_m
    return OBJECT_WIDTHS_M.get(detection.label.lower(), DEFAULT_HUMAN_WIDTH_M)


def obstacle_distance(detection, frame_width, args):
    return estimate_distance_m(
        assumed_object_width(detection, args),
        detection.box[2],
        frame_width,
        args.focal_length_mm,
        args.pixel_pitch_um,
        args.sensor_width_px,
    )


def camera_detection_is_forward(detection, frame_width, args):
    """Return whether a camera detection overlaps the forward image corridor."""
    corridor_half_width = frame_width / 2 * args.camera_forward_half_width
    corridor_left = frame_width / 2 - corridor_half_width
    corridor_right = frame_width / 2 + corridor_half_width
    box_left = detection.box[0]
    box_right = box_left + detection.box[2]
    return box_right >= corridor_left and box_left <= corridor_right


def select_obstacle(detections, frame_size, args):
    """Return the nearest detection and whether any detection is nearby."""
    candidates = []
    for detection in detections:
        if not camera_detection_is_forward(detection, frame_size[0], args):
            continue
        distance_m = obstacle_distance(detection, frame_size[0], args)
        if distance_m is not None:
            candidates.append((distance_m, detection))
    if not candidates:
        return None, None, False

    blocking_candidates = [
        (distance, detection)
        for distance, detection in candidates
        if distance <= args.stop_distance_m
    ]
    blocking = bool(blocking_candidates)
    distance_m, nearest = min(
        blocking_candidates if blocking else candidates,
        key=lambda item: item[0],
    )
    return nearest, distance_m, blocking


def camera_avoidance_steering(detection, frame_width):
    """Return forward-steering input that moves away from a camera obstacle."""
    if detection is None or frame_width <= 0:
        return 0.0
    center_offset = (detection.center[0] - frame_width / 2) / (frame_width / 2)
    if abs(center_offset) < 0.12:
        return 0.0
    return max(-1.0, min(1.0, -center_offset * 1.35))


def interruptible_walk_half_cycle(
    walk,
    home_pose,
    swing_tripod,
    stance_tripod,
    direction,
    steering,
    hip_swing_scale,
    interpolation_steps,
    frame_delay,
    attitude_provider,
    frame_callback,
):
    """Run one gait half-cycle while allowing a sensor safety interruption."""
    if interpolation_steps < 1:
        raise ValueError("interpolation_steps must be at least 1")
    if frame_delay < 0.0:
        raise ValueError("frame_delay cannot be negative")

    for step in range(interpolation_steps + 1):
        if frame_callback() is False:
            return False
        walk.set_walk_frame(
            home_pose,
            swing_tripod,
            stance_tripod,
            step / interpolation_steps,
            direction=direction,
            steering=steering,
            attitude=attitude_provider(),
            hip_swing_scale=hip_swing_scale,
        )
        time.sleep(frame_delay)
    return True


def main():
    args = get_args()
    cv2, MappedArray, Picamera2, IMX500, NetworkIntrinsics, nanodet = (
        import_camera_stack()
    )

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    if intrinsics.task and intrinsics.task != "object detection":
        print(f"Model task is {intrinsics.task!r}, not object detection.", file=sys.stderr)
        return 2
    intrinsics.task = "object detection"
    labels = load_labels(args.labels)
    if labels is not None:
        intrinsics.labels = labels
    for key, value in vars(args).items():
        if value is not None and hasattr(intrinsics, key):
            setattr(intrinsics, key, value)
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": args.fps or intrinsics.inference_rate},
        buffer_count=args.camera_buffers,
    )
    detections = []
    detections_lock = threading.Lock()
    camera_seen_at = 0.0
    blocking = False
    motion = "disengaged"
    dashboard = None
    if not args.no_web:
        dashboard = AvoidanceDashboard(args.web_port)
        try:
            dashboard_url = dashboard.start()
        except OSError as error:
            raise RuntimeError(
                f"Unable to start navigation dashboard on port {args.web_port}: {error}"
            ) from error
        print(f"Live camera and lidar dashboard: {dashboard_url}")

    @lru_cache
    def model_labels():
        result = intrinsics.labels or []
        if intrinsics.ignore_dash_labels:
            result = [label for label in result if label and label != "-"]
        return result

    def label_for(category):
        category = int(category)
        labels_now = model_labels()
        return labels_now[category] if 0 <= category < len(labels_now) else f"class_{category}"

    def parse_detections(metadata):
        nonlocal detections, camera_seen_at
        now = time.monotonic()
        outputs = imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            with detections_lock:
                camera_seen_at = now
                return list(detections)
        input_w, input_h = imx500.get_input_size()
        if intrinsics.postprocess == "nanodet":
            boxes, scores, classes = nanodet(
                outputs=outputs[0], conf=args.threshold, iou_thres=args.iou,
                max_out_dets=args.max_detections,
            )[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
            if intrinsics.bbox_normalization:
                boxes = boxes / input_h
            if intrinsics.bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]
        parsed = []
        for coords, score, category in zip(boxes, scores, classes):
            if score < args.threshold:
                continue
            x, y, width, height = rectangle_to_box(
                imx500.convert_inference_coords(coords, metadata, picam2)
            )
            parsed.append(Detection(label_for(category), int(category), float(score),
                                    (x, y, width, height),
                                    (x + width / 2, y + height / 2)))
        with detections_lock:
            detections = parsed[:args.max_detections]
            camera_seen_at = now
            return list(detections)

    def camera_snapshot():
        with detections_lock:
            current = list(detections)
            seen_at = camera_seen_at
        return current, bool(seen_at and time.monotonic() - seen_at <= CAMERA_READING_MAX_AGE)

    last_dashboard_frame = 0.0

    def draw_overlay(request, stream="main"):
        nonlocal last_dashboard_frame
        current = parse_detections(request.get_metadata())
        with MappedArray(request, stream) as mapped:
            height, width = mapped.array.shape[:2]
            color = (0, 0, 255) if blocking else (0, 255, 0)
            cv2.putText(mapped.array, motion, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2)
            for detection in current:
                x, y, box_w, box_h = detection.box
                distance_m = obstacle_distance(detection, width, args)
                nearby = (
                    camera_detection_is_forward(detection, width, args)
                    and distance_m is not None
                    and distance_m <= args.stop_distance_m
                )
                box_color = (0, 0, 255) if nearby else (0, 255, 0)
                cv2.rectangle(
                    mapped.array, (x, y), (x + box_w, y + box_h), box_color, 3
                )
                label = detection.label
                if distance_m is not None:
                    label += f" {distance_m:.2f}m"
                cv2.putText(mapped.array, label, (x + 4, max(50, y + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            now = time.monotonic()
            if dashboard is not None and now - last_dashboard_frame >= 0.15:
                image = mapped.array
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75],
                )
                if encoded:
                    dashboard.update_camera(jpeg.tobytes())
                    last_dashboard_frame = now

    imx500.show_network_fw_progress_bar()
    picam2.pre_callback = draw_overlay
    try:
        picam2.start(config, show_preview=not args.headless)
    except Exception:
        if dashboard is not None:
            dashboard.stop()
        raise
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    frame_size = tuple(picam2.camera_configuration()["main"]["size"])
    camera_deadline = time.monotonic() + 5.0
    while time.monotonic() < camera_deadline:
        if camera_snapshot()[1]:
            break
        time.sleep(0.05)
    else:
        picam2.stop()
        if dashboard is not None:
            dashboard.stop()
        raise RuntimeError(
            "AI Camera started but produced no frames within 5 seconds. "
            "Run rpicam-hello to check the IMX500/libcamera pipeline."
        )

    walk = None
    home_pose = None
    leveler = None
    walking_enabled = False
    avoiding = False
    avoid_turn_direction = args.turn_direction
    clear_frames = 0
    next_swing = None
    status_printer = ObstacleStatusPrinter(args.print_interval)
    lidar_monitor = LidarObstacleMonitor(
        args.lidar_device,
        args.lidar_stop_distance_m,
        args.lidar_forward_angle_deg,
        args.lidar_forward_offset_deg,
        args.body_half_width_m,
        args.navigation_lookahead_m,
        args.emergency_distance_m,
        args.path_margin_m,
    )

    def stand_robot():
        nonlocal walk, home_pose, leveler, next_swing
        if home_pose is not None:
            return
        import controller_walk as walk_module
        walk = walk_module
        print("Standing up; keep clear of the robot.")
        walk.validate_ik_constants()
        home_pose = walk.run_stand_up_sequence()
        walk.hold_standing_pose(home_pose)
        leveler = walk.LevelingController()
        next_swing = walk.TRIPOD_A

    def neutral_attitude():
        leveler.attitude()
        leveler.clear_correction()
        return {"roll": 0.0, "pitch": 0.0}

    def autonomous_walk_attitude(direction):
        return walk.tripod_level_attitude(direction, leveler, force=True)

    def navigation_snapshot():
        current, camera_live = camera_snapshot()
        nearest_camera, camera_distance_m, camera_blocking = select_obstacle(
            current,
            frame_size,
            args,
        )
        camera_steering = camera_avoidance_steering(
            nearest_camera,
            frame_size[0],
        )
        lidar_status = lidar_monitor.snapshot()
        lidar_unavailable = not lidar_status["live"]

        if lidar_unavailable:
            mode = "stopped_lidar_unavailable"
            steering = 0.0
        elif not camera_live:
            mode = "stopped_camera_unavailable"
            steering = 0.0
        elif lidar_status["must_turn"]:
            mode = "turn"
            steering = 0.0
        elif lidar_status["steer_forward"]:
            mode = "steer"
            steering = lidar_status["steering"]
        elif camera_blocking and camera_steering:
            mode = "steer"
            steering = camera_steering
        elif camera_blocking:
            mode = "turn"
            steering = 0.0
        else:
            mode = "forward"
            steering = 0.0

        turn_direction = lidar_status["turn_direction"]
        if turn_direction is None and nearest_camera is not None:
            camera_offset = nearest_camera.center[0] - frame_size[0] / 2
            if camera_offset > frame_size[0] * 0.06:
                turn_direction = "left"
            elif camera_offset < -frame_size[0] * 0.06:
                turn_direction = "right"
        if turn_direction is None:
            turn_direction = args.turn_direction

        return {
            "mode": mode,
            "steering": steering,
            "turn_direction": turn_direction,
            "camera_live": camera_live,
            "camera_blocking": camera_blocking,
            "camera_distance_m": camera_distance_m,
            "detections": current,
            "lidar": lidar_status,
            "blocking": (
                camera_blocking
                or lidar_status["blocking"]
                or lidar_unavailable
                or not camera_live
            ),
        }

    def publish_dashboard(navigation, displayed_motion):
        if dashboard is None:
            return
        detection_items = []
        for detection in navigation["detections"]:
            distance_m = obstacle_distance(detection, frame_size[0], args)
            detection_items.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": [int(value) for value in detection.box],
                    "distance_m": round(distance_m, 2) if distance_m is not None else None,
                    "assumed_width_m": assumed_object_width(detection, args),
                }
            )
        dashboard.update_state(
            {
                "motion": displayed_motion,
                "steering": round(navigation["steering"], 3),
                "camera_live": navigation["camera_live"],
                "camera_blocking": navigation["camera_blocking"],
                "detections": detection_items,
                "lidar": navigation["lidar"],
            }
        )

    try:
        print(f"Starting RPLIDAR C1 on {args.lidar_device} at 460800 baud.")
        lidar_monitor.start()
        print("Lidar ready; forward path monitoring is active.")
        stand_robot()
        motion = "standing"
        print("Camera ready. Robot is standing still and waiting for W.")
        print("W=start/resume, S=stop, P=disengage, O=stand only, Ctrl+C=quit.")
        with KeyboardControls() as keyboard:
            while True:
                keys = keyboard.read_keys()
                # Safety keys win if several buffered keys arrive together.
                disengage_requested = b"p" in keys
                if disengage_requested:
                    print("P pressed: disengaging.")
                    walking_enabled = False
                    avoiding = False
                    clear_frames = 0
                    motion = "disengaged"
                    if walk is not None:
                        walk.release_all()
                    home_pose = None
                    leveler = None
                    next_swing = None
                elif b"s" in keys:
                    walking_enabled = False
                    avoiding = False
                    clear_frames = 0
                    motion = "stopped"
                    if home_pose is not None:
                        walk.hold_standing_pose(home_pose, neutral_attitude())
                    print("S pressed: walking stopped.")
                elif b"o" in keys:
                    walking_enabled = False
                    avoiding = False
                    clear_frames = 0
                    stand_robot()
                    walk.hold_standing_pose(home_pose, neutral_attitude())
                    motion = "standing"
                    print("O pressed: standing still; walking remains disabled.")
                elif b"w" in keys:
                    stand_robot()
                    walking_enabled = True
                    motion = "walking_forward"
                    print("W pressed: forward walking enabled.")

                navigation = navigation_snapshot()
                requested_mode = navigation["mode"]
                if walking_enabled and requested_mode == "turn":
                    if not avoiding:
                        avoid_turn_direction = navigation["turn_direction"]
                    avoiding = True
                    clear_frames = 0
                elif walking_enabled and avoiding and not requested_mode.startswith("stopped"):
                    clear_frames += 1
                    if clear_frames < args.clear_frames:
                        requested_mode = "turn"
                    else:
                        avoiding = False
                        clear_frames = 0

                blocking = navigation["blocking"]
                if not walking_enabled:
                    motion = "stopped" if home_pose is not None else "disengaged"
                elif requested_mode.startswith("stopped_"):
                    motion = requested_mode
                elif requested_mode == "turn":
                    motion = f"turning_{avoid_turn_direction}"
                elif requested_mode == "steer":
                    steer_name = "right" if navigation["steering"] > 0.0 else "left"
                    motion = f"steering_{steer_name}"
                else:
                    motion = "walking_forward"
                status_printer.print(
                    navigation["detections"],
                    frame_size,
                    args,
                    blocking,
                    motion,
                    navigation["lidar"],
                )
                publish_dashboard(navigation, motion)

                if not walking_enabled:
                    time.sleep(0.03)
                    continue
                if requested_mode.startswith("stopped_"):
                    walk.hold_standing_pose(home_pose, neutral_attitude())
                    time.sleep(0.05)
                    continue

                if requested_mode == "turn":
                    # controller_walk uses +3 for left/CCW and -3 for right/CW.
                    direction = 3 if avoid_turn_direction == "left" else -3
                    steering = 0.0
                elif requested_mode == "steer":
                    direction = 4
                    steering = navigation["steering"]
                else:
                    direction = 1
                    steering = 0.0
                stance_tripod = (
                    walk.TRIPOD_B if next_swing == walk.TRIPOD_A else walk.TRIPOD_A
                )

                def frame_is_safe():
                    latest = navigation_snapshot()
                    if latest["mode"].startswith("stopped_"):
                        return False
                    if direction in (1, 4) and latest["mode"] == "turn":
                        return False
                    return True

                walk_attitude = autonomous_walk_attitude(direction)
                completed = interruptible_walk_half_cycle(
                    walk,
                    home_pose,
                    next_swing,
                    stance_tripod,
                    direction,
                    steering,
                    args.turn_scale if requested_mode == "turn" else 1.0,
                    args.walk_steps,
                    args.walk_frame_delay,
                    lambda: walk_attitude,
                    frame_is_safe,
                )
                if completed:
                    next_swing = stance_tripod
                    leveler.clear_correction()
                else:
                    walk.hold_standing_pose(home_pose, neutral_attitude())
    except KeyboardInterrupt:
        print("\nCtrl+C: quitting and disengaging.")
        return 0
    finally:
        try:
            lidar_monitor.stop()
        except Exception as error:
            print(f"Lidar shutdown failed: {error}", file=sys.stderr)
        try:
            picam2.stop()
        except Exception as error:
            print(f"Camera shutdown failed: {error}", file=sys.stderr)
        try:
            if dashboard is not None:
                dashboard.stop()
        finally:
            if walk is not None:
                walk.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
