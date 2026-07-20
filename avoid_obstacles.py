"""Walk forward and turn around obstacles seen by the AI Camera and lidar.

The robot stands up when the program starts, but does not walk until W is
pressed. W starts/resumes forward walking, S stops all walking while holding
the standing pose, P disengages the servos, and O stands up again without
walking. Camera monitoring continues throughout. Ctrl+C exits the whole
program.

The camera identifies objects while the RPLIDAR C1 measures the physical path
in front of the robot. The robot veers toward the clearer side, follows the
obstacle at a target distance, and then counter-steers back to its original
travel heading.
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
from statistics import median

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


LIDAR_POINT_MAX_AGE_S = 0.35
LIDAR_BAUDRATE = 460800
LIDAR_START_ATTEMPTS = 3
LIDAR_RETRY_DELAY_S = 0.75
LIDAR_STOP_COMMAND = b"\xA5\x25"
DEFAULT_CAMERA_BUFFER_COUNT = 6
DEFAULT_WEB_PORT = 8000
CAMERA_START_TIMEOUT_S = 5.0
SIDE_TRACK_MIN_POINTS = 3
SIDE_TRACK_SAMPLE_COUNT = 7
SIDE_TRACK_MAX_RANGE_M = 2.5
SIDE_TRACK_MIN_ANGLE_DEG = 30.0
SIDE_TRACK_MAX_ANGLE_DEG = 150.0
SIDE_DISTANCE_FILTER_ALPHA = 0.55
SIDE_DISTANCE_DEADBAND_M = 0.04
SIDE_ACQUIRE_MAX_CYCLES = 10
SIDE_PASSED_ANGLE_DEG = 112.0
SIDE_PASSED_CYCLES = 2
SIDE_MISSING_CYCLES = 4
FOLLOW_HEADING_DIVISOR_DEG = 20.0
FOLLOW_HEADING_MAX_FRACTION = 0.60
MPU_YAW_SIGN_CALIBRATION_DEG = 1.0
MPU_HEADING_SETTLE_CYCLES = 3
TRACK_CLUSTER_MAX_ANGLE_GAP_DEG = 7.0
TRACK_CLUSTER_MAX_POINT_GAP_M = 0.28
TRACK_ASSOCIATION_MAX_DISTANCE_M = 0.45
TRACK_ASSOCIATION_MAX_ANGLE_DEG = 32.0
TRACK_ASSOCIATION_MAX_MISSING_DISTANCE_M = 0.70
TRACK_ASSOCIATION_MAX_MISSING_ANGLE_DEG = 65.0
TRACK_MEMORY_CYCLES = 24
TRACK_FILTER_ALPHA = 0.72
TRACK_SIDE_MIN_ANGLE_DEG = 15.0
WRAP_COUNTERSTEER_CYCLES = 3
WRAP_COUNTERSTEER_FRACTION = 0.55


def signed_lidar_angle(angle):
    """Convert 0..360 degrees to -180..180, with forward at zero."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


def polar_xy(distance_m, angle_deg):
    """Convert lidar polar coordinates to right/forward robot coordinates."""
    angle_rad = math.radians(angle_deg)
    return distance_m * math.sin(angle_rad), distance_m * math.cos(angle_rad)


def cluster_lidar_points(points):
    """Group adjacent lidar returns into physical obstacle surfaces."""
    samples = sorted(
        (
            {
                "angle_deg": float(point["relative_angle_deg"]),
                "distance_m": float(point["distance_mm"]) / 1000.0,
            }
            for point in points
            if point.get("distance_mm", 0.0) > 0.0
        ),
        key=lambda sample: sample["angle_deg"],
    )
    groups = []
    for sample in samples:
        sample["x_m"], sample["y_m"] = polar_xy(
            sample["distance_m"], sample["angle_deg"]
        )
        if not groups:
            groups.append([sample])
            continue
        previous = groups[-1][-1]
        angle_gap = sample["angle_deg"] - previous["angle_deg"]
        point_gap = math.hypot(
            sample["x_m"] - previous["x_m"],
            sample["y_m"] - previous["y_m"],
        )
        if (
            angle_gap <= TRACK_CLUSTER_MAX_ANGLE_GAP_DEG
            and point_gap <= TRACK_CLUSTER_MAX_POINT_GAP_M
        ):
            groups[-1].append(sample)
        else:
            groups.append([sample])

    clusters = []
    for group in groups:
        # The closest few points represent the surface the body must clear;
        # farther returns in the same angular run should not pull the track
        # through or behind the obstacle.
        surface = sorted(group, key=lambda sample: sample["distance_m"])[
            :SIDE_TRACK_SAMPLE_COUNT
        ]
        angle_deg = median(sample["angle_deg"] for sample in surface)
        distance_m = median(sample["distance_m"] for sample in surface)
        x_m, y_m = polar_xy(distance_m, angle_deg)
        lateral_m = median(abs(sample["x_m"]) for sample in surface)
        surface_width_m = math.hypot(
            group[-1]["x_m"] - group[0]["x_m"],
            group[-1]["y_m"] - group[0]["y_m"],
        )
        clusters.append(
            {
                "angle_deg": angle_deg,
                "distance_m": distance_m,
                "x_m": x_m,
                "y_m": y_m,
                "lateral_m": lateral_m,
                "point_count": len(group),
                "surface_width_m": surface_width_m,
                "points": [
                    {
                        "relative_angle_deg": round(sample["angle_deg"], 2),
                        "distance_mm": round(sample["distance_m"] * 1000.0, 1),
                    }
                    for sample in group
                ],
            }
        )
    return clusters


class LockedObstacleTracker:
    """Keep the lidar surface that triggered a bypass locked across turns."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.locked = False
        self.angle_deg = None
        self.distance_m = None
        self.lateral_m = None
        self.missing_cycles = 0
        self.observation_cycles = 0
        self.last_commanded_heading_deg = 0.0
        self.point_count = 0
        self.surface_width_m = None
        self.tracked_points = []
        self.association_error_m = None

    def lock(self, lidar_status, commanded_heading_deg=0.0):
        """Seal the nearest forward obstacle as the bypass target."""
        angle_deg = lidar_status.get("nearest_angle_deg")
        distance_m = lidar_status.get("nearest_m")
        if angle_deg is None or distance_m is None:
            angle_deg = lidar_status.get("track_candidate_angle_deg")
            distance_m = lidar_status.get("track_candidate_m")
        if angle_deg is None or distance_m is None:
            return False
        target_x, target_y = polar_xy(distance_m, angle_deg)
        clusters = cluster_lidar_points(lidar_status.get("points", []))
        cluster = min(
            clusters,
            key=lambda item: math.hypot(
                item["x_m"] - target_x, item["y_m"] - target_y
            ),
            default=None,
        )
        if cluster is None:
            cluster = {
                "angle_deg": angle_deg,
                "distance_m": distance_m,
                "lateral_m": abs(target_x),
                "point_count": 1,
                "surface_width_m": 0.0,
                "points": [],
            }
        self.locked = True
        self.angle_deg = cluster["angle_deg"]
        self.distance_m = cluster["distance_m"]
        self.lateral_m = cluster["lateral_m"]
        self.missing_cycles = 0
        self.observation_cycles = 1
        self.last_commanded_heading_deg = commanded_heading_deg
        self.point_count = cluster["point_count"]
        self.surface_width_m = cluster["surface_width_m"]
        self.tracked_points = cluster["points"]
        self.association_error_m = 0.0
        return True

    def update(self, lidar_status, commanded_heading_deg):
        """Predict the locked bearing, then associate it with the next scan."""
        if not self.locked:
            return False
        heading_delta = signed_lidar_angle(
            commanded_heading_deg - self.last_commanded_heading_deg
        )
        predicted_angle = signed_lidar_angle(self.angle_deg - heading_delta)
        predicted_x, predicted_y = polar_xy(self.distance_m, predicted_angle)
        self.angle_deg = predicted_angle
        self.last_commanded_heading_deg = commanded_heading_deg

        max_distance = min(
            TRACK_ASSOCIATION_MAX_MISSING_DISTANCE_M,
            TRACK_ASSOCIATION_MAX_DISTANCE_M + 0.03 * self.missing_cycles,
        )
        max_angle = min(
            TRACK_ASSOCIATION_MAX_MISSING_ANGLE_DEG,
            TRACK_ASSOCIATION_MAX_ANGLE_DEG + 2.0 * self.missing_cycles,
        )
        candidates = []
        for cluster in cluster_lidar_points(lidar_status.get("points", [])):
            angle_error = abs(
                signed_lidar_angle(cluster["angle_deg"] - predicted_angle)
            )
            position_error = math.hypot(
                cluster["x_m"] - predicted_x,
                cluster["y_m"] - predicted_y,
            )
            if angle_error <= max_angle and position_error <= max_distance:
                width_error = abs(
                    cluster["surface_width_m"] - self.surface_width_m
                )
                score = position_error + angle_error * 0.010 + width_error * 0.30
                candidates.append((score, position_error, cluster))

        if not candidates:
            self.missing_cycles += 1
            self.tracked_points = []
            self.association_error_m = None
            if self.missing_cycles > TRACK_MEMORY_CYCLES:
                self.locked = False
            return False

        _, position_error, cluster = min(candidates, key=lambda item: item[0])
        angle_error = signed_lidar_angle(cluster["angle_deg"] - predicted_angle)
        self.angle_deg = signed_lidar_angle(
            predicted_angle + TRACK_FILTER_ALPHA * angle_error
        )
        self.distance_m = (
            TRACK_FILTER_ALPHA * cluster["distance_m"]
            + (1.0 - TRACK_FILTER_ALPHA) * self.distance_m
        )
        # Lateral clearance changes quickly while the obstacle moves from the
        # front to the side. Keep this measurement current; the wall-following
        # controller already applies its own distance filter and deadband.
        self.lateral_m = cluster["lateral_m"]
        self.missing_cycles = 0
        self.observation_cycles += 1
        self.point_count = cluster["point_count"]
        self.surface_width_m = cluster["surface_width_m"]
        self.tracked_points = cluster["points"]
        self.association_error_m = position_error
        return True

    def side_distance(self, bypass_side):
        if not self.locked or abs(self.angle_deg) < TRACK_SIDE_MIN_ANGLE_DEG:
            return None
        expected_sign = 1.0 if bypass_side == "left" else -1.0
        if self.angle_deg * expected_sign <= 0.0:
            return None
        return self.lateral_m

    def status(self):
        return {
            "locked": self.locked,
            "angle_deg": (
                round(self.angle_deg, 1) if self.angle_deg is not None else None
            ),
            "distance_m": (
                round(self.distance_m, 3) if self.distance_m is not None else None
            ),
            "lateral_m": (
                round(self.lateral_m, 3) if self.lateral_m is not None else None
            ),
            "missing_cycles": self.missing_cycles,
            "observation_cycles": self.observation_cycles,
            "point_count": self.point_count,
            "surface_width_m": (
                round(self.surface_width_m, 3)
                if self.surface_width_m is not None else None
            ),
            "points": self.tracked_points,
            "association_error_m": (
                round(self.association_error_m, 3)
                if self.association_error_m is not None else None
            ),
        }


def ensure_lidar_driver():
    """Fail early with an install command for the active Python environment."""
    try:
        from rplidarc1 import RPLidar  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "The RPLIDAR C1 driver is not installed in the Python environment "
            "running this program.\n"
            "Install it with this exact interpreter, then run the program again:\n"
            f"  {sys.executable} -m pip install rplidarc1==0.1.3\n"
            f"Active interpreter: {sys.executable}"
        ) from error


def stop_stale_lidar_stream(device, baudrate=LIDAR_BAUDRATE):
    """Stop an old scan and discard bytes left in the USB serial input queue."""
    import serial

    try:
        with serial.Serial(
            device,
            baudrate,
            timeout=0.2,
            write_timeout=0.2,
            exclusive=True,
        ) as port:
            port.write(LIDAR_STOP_COMMAND)
            port.flush()
            time.sleep(0.10)
            port.reset_input_buffer()
    except (OSError, serial.SerialException) as error:
        raise RuntimeError(
            f"Unable to prepare RPLIDAR serial device {device}: {error}. "
            "Confirm the device path and stop any other process using the port."
        ) from error


def open_lidar_with_recovery(device, baudrate=LIDAR_BAUDRATE):
    """Open the C1, retrying health-check sync after clearing stale scan bytes."""
    from rplidarc1 import RPLidar

    for attempt in range(1, LIDAR_START_ATTEMPTS + 1):
        stop_stale_lidar_stream(device, baudrate)
        try:
            return RPLidar(device, baudrate)
        except ValueError as error:
            if attempt == LIDAR_START_ATTEMPTS:
                raise RuntimeError(
                    "RPLIDAR returned an invalid health-response descriptor after "
                    f"{LIDAR_START_ATTEMPTS} attempts. Check that --lidar-device "
                    "selects the C1, no other process has the serial port open, and "
                    "the USB cable and 5V supply are stable."
                ) from error
            print(
                "RPLIDAR response was out of sync; clearing serial input and "
                f"retrying ({attempt}/{LIDAR_START_ATTEMPTS}).",
                file=sys.stderr,
            )
            time.sleep(LIDAR_RETRY_DELAY_S)

    raise AssertionError("unreachable")


class LidarObstacleMonitor:
    """Continuously read the C1 in a background thread without delaying gait."""

    def __init__(self, device, stop_distance_m, forward_angle_deg, forward_offset_deg):
        self.device = device
        self.stop_distance_mm = stop_distance_m * 1000.0
        self.forward_angle_deg = forward_angle_deg
        self.forward_offset_deg = forward_offset_deg
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
        lidar = open_lidar_with_recovery(self.device, LIDAR_BAUDRATE)
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
                if now - point["seen_at"] <= LIDAR_POINT_MAX_AGE_S
            ]
        live = self.error is None and now - self.last_packet <= 1.5

        def relative_angle(point):
            return signed_lidar_angle(point["angle"] - self.forward_offset_deg)

        relative_points = [
            {
                "relative_angle_deg": round(relative_angle(point), 2),
                "distance_mm": round(point["distance_mm"], 1),
                "quality": point["quality"],
            }
            for point in points
        ]

        forward = [
            point
            for point in points
            if abs(relative_angle(point)) <= self.forward_angle_deg
        ]
        blocking_points = [
            point
            for point in forward
            if point["distance_mm"] <= self.stop_distance_mm
        ]
        nearest = min(blocking_points, key=lambda point: point["distance_mm"], default=None)
        track_candidate = min(
            forward,
            key=lambda point: point["distance_mm"],
            default=None,
        )

        def side_clearance(side):
            if side == "left":
                side_points = [
                    point for point in points
                    if -120.0 <= relative_angle(point) <= -15.0
                ]
            else:
                side_points = [
                    point for point in points
                    if 15.0 <= relative_angle(point) <= 120.0
                ]
            return min(
                (point["distance_mm"] for point in side_points),
                default=12000.0,
            )

        def side_track(side):
            if side == "left":
                side_points = [
                    point for point in points
                    if (
                        -SIDE_TRACK_MAX_ANGLE_DEG
                        <= relative_angle(point)
                        <= -SIDE_TRACK_MIN_ANGLE_DEG
                    )
                ]
            else:
                side_points = [
                    point for point in points
                    if (
                        SIDE_TRACK_MIN_ANGLE_DEG
                        <= relative_angle(point)
                        <= SIDE_TRACK_MAX_ANGLE_DEG
                    )
                ]
            samples = []
            for point in side_points:
                distance_m = point["distance_mm"] / 1000.0
                if distance_m > SIDE_TRACK_MAX_RANGE_M:
                    continue
                angle_deg = relative_angle(point)
                angle_rad = math.radians(angle_deg)
                samples.append(
                    {
                        "angle_deg": angle_deg,
                        "lateral_m": abs(distance_m * math.sin(angle_rad)),
                        "forward_m": distance_m * math.cos(angle_rad),
                    }
                )
            if len(samples) < SIDE_TRACK_MIN_POINTS:
                return {"distance_m": None, "angle_deg": None, "forward_m": None}

            # Use the closest surface samples and measure perpendicular
            # clearance, not polar/slant range. This keeps the target distance
            # stable as the same obstacle moves from front-side to rear-side.
            tracked = sorted(samples, key=lambda sample: sample["lateral_m"])[
                :SIDE_TRACK_SAMPLE_COUNT
            ]
            return {
                "distance_m": median(sample["lateral_m"] for sample in tracked),
                "angle_deg": median(sample["angle_deg"] for sample in tracked),
                "forward_m": median(sample["forward_m"] for sample in tracked),
            }

        left_track = side_track("left")
        right_track = side_track("right")

        turn_direction = None
        direction_target = nearest or track_candidate
        if direction_target is not None:
            obstacle_angle = relative_angle(direction_target)
            if obstacle_angle > 5.0:
                turn_direction = "left"
            elif obstacle_angle < -5.0:
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
            "blocking": bool(blocking_points),
            "nearest_m": (
                round(nearest["distance_mm"] / 1000.0, 3)
                if nearest is not None else None
            ),
            "nearest_angle_deg": (
                round(relative_angle(nearest), 1)
                if nearest is not None else None
            ),
            "track_candidate_m": (
                round(track_candidate["distance_mm"] / 1000.0, 3)
                if track_candidate is not None else None
            ),
            "track_candidate_angle_deg": (
                round(relative_angle(track_candidate), 1)
                if track_candidate is not None else None
            ),
            "turn_direction": turn_direction,
            "left_distance_m": left_track["distance_m"],
            "left_angle_deg": left_track["angle_deg"],
            "left_forward_m": left_track["forward_m"],
            "right_distance_m": right_track["distance_m"],
            "right_angle_deg": right_track["angle_deg"],
            "right_forward_m": right_track["forward_m"],
            "forward_half_angle_deg": self.forward_angle_deg,
            "points": relative_points,
            "error": str(self.error) if self.error is not None else None,
        }


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hexapod Obstacle Tracking</title><style>
body{margin:0;background:#091015;color:#e5edf3;font:14px system-ui,sans-serif}header{padding:12px 16px;border-bottom:1px solid #293640;display:flex;gap:16px;align-items:center}h1{font-size:18px;margin:0}.status{color:#8bd5a1}main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}section{min-width:0}h2{font-size:14px;margin:0 0 8px;color:#a9bac7}img,canvas{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#03070a;border:1px solid #293640}pre{white-space:pre-wrap;color:#b8c7d1;margin:10px 0 0}@media(max-width:800px){main{grid-template-columns:1fr}}
</style></head><body><header><h1>Hexapod Obstacle Tracking</h1><div id="status" class="status">Starting</div></header><main><section><h2>AI Camera</h2><img id="camera" alt="Annotated camera view"></section><section><h2>RPLIDAR C1</h2><canvas id="lidar" width="800" height="600"></canvas><pre id="details"></pre></section></main><script>
const camera=document.querySelector('#camera'),canvas=document.querySelector('#lidar'),ctx=canvas.getContext('2d'),statusEl=document.querySelector('#status'),details=document.querySelector('#details');
function draw(data){const lidar=data.lidar||{},pts=lidar.points||[],bypass=lidar.bypass||{},track=bypass.tracked_obstacle||{},w=canvas.width,h=canvas.height,cx=w/2,cy=h/2,r=Math.min(w,h)*.44,max=2500;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#29404f';ctx.fillStyle='#8aa0ae';ctx.font='13px system-ui';ctx.textAlign='center';for(let i=1;i<=5;i++){ctx.beginPath();ctx.arc(cx,cy,r*i/5,0,Math.PI*2);ctx.stroke();ctx.fillText((max*i/5000).toFixed(1)+'m',cx+4,cy-r*i/5+15)}ctx.beginPath();ctx.moveTo(cx-r,cy);ctx.lineTo(cx+r,cy);ctx.moveTo(cx,cy-r);ctx.lineTo(cx,cy+r);ctx.stroke();ctx.fillStyle='#44c7e8';for(const p of pts){const a=p.relative_angle_deg*Math.PI/180,d=Math.min(p.distance_mm/max,1)*r,x=cx+Math.sin(a)*d,y=cy-Math.cos(a)*d;ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fill()}if(track.locked&&track.angle_deg!=null&&track.distance_m!=null){const a=track.angle_deg*Math.PI/180,d=Math.min(track.distance_m*1000/max,1)*r,x=cx+Math.sin(a)*d,y=cy-Math.cos(a)*d;ctx.strokeStyle='#ff9f43';ctx.fillStyle='#ff9f43';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);ctx.stroke();ctx.beginPath();ctx.arc(x,y,10,0,Math.PI*2);ctx.fill();ctx.lineWidth=1}statusEl.textContent=(data.motion||'unknown')+' | camera '+(data.camera_live?'live':'waiting')+' | lidar '+(lidar.live?'live':'stale')+' | target '+(track.locked?'LOCKED':'none');details.textContent=`motion: ${data.motion||'unknown'}\nsteering: ${Number(data.steering||0).toFixed(2)}\nphase: ${bypass.phase||'cruise'}\ntarget bearing: ${track.angle_deg==null?'--':track.angle_deg+' deg'}\ntarget range: ${track.distance_m==null?'--':track.distance_m+' m'}\nside clearance: ${track.lateral_m==null?'--':track.lateral_m+' m'}\ntarget scan points: ${track.point_count||0}\nmissed scans: ${track.missing_cycles||0}`;}
function drawLockedPoints(data){const track=(((data.lidar||{}).bypass||{}).tracked_obstacle||{}),points=track.points||[],w=canvas.width,h=canvas.height,cx=w/2,cy=h/2,r=Math.min(w,h)*.44,max=2500;ctx.fillStyle='#ff9f43';for(const p of points){const a=p.relative_angle_deg*Math.PI/180,d=Math.min(p.distance_mm/max,1)*r,x=cx+Math.sin(a)*d,y=cy-Math.cos(a)*d;ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill()}}
function drawMpuHeading(data){const bypass=((data.lidar||{}).bypass||{}),value=v=>v==null?'--':Number(v).toFixed(1)+' deg';details.textContent+=`\nMPU start yaw: ${value(bypass.bypass_start_yaw_deg)}\nMPU current yaw: ${value(bypass.current_yaw_deg)}\nMPU yaw error: ${value(bypass.raw_mpu_heading_error_deg)}\norientation settle: ${bypass.heading_settle_cycles||0}/${3}`}
async function update(){try{const response=await fetch('/state',{cache:'no-store'}),data=await response.json();draw(data);drawLockedPoints(data);drawMpuHeading(data);camera.src='/camera.jpg?t='+Date.now()}catch(error){statusEl.textContent='Dashboard disconnected'}}setInterval(update,200);update();
</script></body></html>"""


def local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        try:
            connection.connect(("10.255.255.255", 1))
            return connection.getsockname()[0]
        except OSError:
            return "127.0.0.1"


class AvoidanceDashboard:
    """Serve the annotated camera frame and locked-target lidar view."""

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
                        body = json.dumps(
                            dashboard.state, separators=(",", ":")
                        ).encode()
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


class ObstacleBypassController:
    """Steer around one obstacle and recover the original travel heading."""

    def __init__(
        self,
        target_distance_m,
        max_steering,
        follow_gain,
        heading_tolerance_deg,
        emergency_distance_m,
        fallback_direction,
    ):
        self.target_distance_m = target_distance_m
        self.max_steering = max_steering
        self.follow_gain = follow_gain
        self.heading_tolerance_deg = heading_tolerance_deg
        self.emergency_distance_m = emergency_distance_m
        self.fallback_direction = fallback_direction
        self.tracker = LockedObstacleTracker()
        self.reset()

    def reset(self, camera_rearm_required=False):
        self.phase = "cruise"
        self.bypass_side = None
        self.heading_error_deg = 0.0
        self.commanded_heading_error_deg = 0.0
        self.start_yaw_deg = None
        self.current_yaw_deg = None
        self.mpu_yaw_sign = None
        self.raw_mpu_heading_error_deg = None
        self.mpu_heading_required = False
        self.mpu_heading_live = False
        self.heading_settle_cycles = 0
        self.heading_source = "commanded"
        self.phase_cycles = 0
        self.side_missing_cycles = 0
        self.side_passed_cycles = 0
        self.filtered_side_distance_m = None
        self.last_side_angle_deg = None
        self.last_distance_error_m = None
        self.last_steering = 0.0
        self.wrap_countersteer_cycles = 0
        self.camera_rearm_required = camera_rearm_required
        self.tracker.reset()

    def _start_bypass(
        self,
        lidar_status,
        current_yaw_deg=None,
        preserve_heading=False,
    ):
        self.phase = "veer_out"
        self.bypass_side = (
            lidar_status.get("turn_direction") or self.fallback_direction
        )
        if not preserve_heading:
            self.heading_error_deg = 0.0
            self.commanded_heading_error_deg = 0.0
            self.start_yaw_deg = current_yaw_deg
            self.current_yaw_deg = current_yaw_deg
            self.mpu_yaw_sign = None
            self.raw_mpu_heading_error_deg = None
            self.mpu_heading_required = current_yaw_deg is not None
            self.mpu_heading_live = current_yaw_deg is not None
            self.heading_settle_cycles = 0
            self.heading_source = "commanded"
        self.phase_cycles = 0
        self.side_missing_cycles = 0
        self.side_passed_cycles = 0
        self.filtered_side_distance_m = None
        self.last_side_angle_deg = None
        self.last_distance_error_m = None
        self.wrap_countersteer_cycles = 0
        self.camera_rearm_required = True
        if not preserve_heading or not self.tracker.locked:
            self.tracker.lock(
                lidar_status,
                commanded_heading_deg=self.commanded_heading_error_deg,
            )

    def _update_heading(self, current_yaw_deg):
        if self.phase == "cruise":
            return
        self.mpu_heading_live = (
            self.start_yaw_deg is not None and current_yaw_deg is not None
        )
        if self.start_yaw_deg is not None and current_yaw_deg is not None:
            self.current_yaw_deg = current_yaw_deg
            raw_yaw_error_deg = signed_lidar_angle(
                current_yaw_deg - self.start_yaw_deg
            )
            self.raw_mpu_heading_error_deg = raw_yaw_error_deg
            if (
                self.mpu_yaw_sign is None
                and abs(raw_yaw_error_deg) >= MPU_YAW_SIGN_CALIBRATION_DEG
                and abs(self.commanded_heading_error_deg)
                >= MPU_YAW_SIGN_CALIBRATION_DEG
            ):
                self.mpu_yaw_sign = (
                    1.0
                    if raw_yaw_error_deg * self.commanded_heading_error_deg > 0.0
                    else -1.0
                )
            if self.mpu_yaw_sign is not None:
                self.heading_error_deg = raw_yaw_error_deg * self.mpu_yaw_sign
                self.heading_source = "mpu6050"
            else:
                self.heading_error_deg = self.commanded_heading_error_deg
                self.heading_source = "commanded"
        else:
            self.heading_error_deg = self.commanded_heading_error_deg
            self.heading_source = "commanded"

    def _away_steering(self):
        return -self.max_steering if self.bypass_side == "left" else self.max_steering

    def _obstacle_side_distance(self, lidar_status):
        if self.tracker.locked:
            return self.tracker.side_distance(self.bypass_side)
        # Passing on the left leaves the obstacle on the robot's right, and
        # passing on the right leaves it on the robot's left.
        key = "right_distance_m" if self.bypass_side == "left" else "left_distance_m"
        return lidar_status.get(key)

    def _obstacle_side_angle(self, lidar_status):
        if self.tracker.locked:
            return self.tracker.angle_deg
        key = "right_angle_deg" if self.bypass_side == "left" else "left_angle_deg"
        return lidar_status.get(key)

    def _side_has_passed(self, side_angle_deg):
        if side_angle_deg is None:
            return False
        if self.bypass_side == "left":
            return side_angle_deg >= SIDE_PASSED_ANGLE_DEG
        return side_angle_deg <= -SIDE_PASSED_ANGLE_DEG

    def _heading_steering(self):
        correction = -self.heading_error_deg / FOLLOW_HEADING_DIVISOR_DEG
        limit = self.max_steering * FOLLOW_HEADING_MAX_FRACTION
        return max(-limit, min(limit, correction))

    def _distance_steering(self, side_distance_m):
        if side_distance_m is None:
            self.last_distance_error_m = None
            return 0.0

        if self.filtered_side_distance_m is None:
            self.filtered_side_distance_m = side_distance_m
        else:
            self.filtered_side_distance_m = (
                SIDE_DISTANCE_FILTER_ALPHA * self.filtered_side_distance_m
                + (1.0 - SIDE_DISTANCE_FILTER_ALPHA) * side_distance_m
            )

        distance_error = self.filtered_side_distance_m - self.target_distance_m
        self.last_distance_error_m = distance_error
        if abs(distance_error) <= SIDE_DISTANCE_DEADBAND_M:
            return 0.0

        effective_error = math.copysign(
            abs(distance_error) - SIDE_DISTANCE_DEADBAND_M,
            distance_error,
        )
        toward_obstacle_sign = 1.0 if self.bypass_side == "left" else -1.0
        return (
            toward_obstacle_sign
            * self.follow_gain
            * effective_error
            / self.target_distance_m
        )

    def _follow_steering(self, side_distance_m):
        distance_steering = self._distance_steering(side_distance_m)
        if (
            self.wrap_countersteer_cycles > 0
            and side_distance_m
            >= self.target_distance_m - SIDE_DISTANCE_DEADBAND_M
        ):
            # Reverse the outbound turn once the requested clearance is
            # reached. A right bypass must now steer left around the target;
            # a left bypass mirrors that behavior.
            countersteer = -math.copysign(
                self.max_steering * WRAP_COUNTERSTEER_FRACTION,
                self._away_steering(),
            )
            self.last_steering = countersteer
            return countersteer
        heading_steering = self._heading_steering()
        if distance_steering * heading_steering < 0.0:
            # Clearance wins over heading. In particular, never let the merge
            # correction steer toward an obstacle that is already too close.
            if (
                self.last_distance_error_m is not None
                and self.last_distance_error_m < -SIDE_DISTANCE_DEADBAND_M
            ):
                heading_steering = 0.0
            else:
                heading_steering *= 0.35
        correction = distance_steering + heading_steering
        self.last_steering = max(
            -self.max_steering,
            min(self.max_steering, correction),
        )
        return self.last_steering

    def _return_heading(self):
        self.phase = "return_heading"
        self.phase_cycles = 0
        self.side_missing_cycles = 0
        self.side_passed_cycles = 0
        self.heading_settle_cycles = 0
        self.last_steering = self._heading_steering()
        return {"mode": "return_heading", "steering": self.last_steering}

    def update(self, lidar_status, camera_blocking, current_yaw_deg=None):
        """Return a motion mode and steering command for the next half-cycle."""
        self._update_heading(current_yaw_deg)
        if not lidar_status["live"]:
            self.last_steering = 0.0
            return {"mode": "stopped_lidar_unavailable", "steering": 0.0}

        nearest_m = lidar_status.get("nearest_m")
        if nearest_m is not None and nearest_m <= self.emergency_distance_m:
            self.last_steering = 0.0
            return {"mode": "stopped_emergency", "steering": 0.0}

        # A camera box can remain visible for several frames after the lidar
        # has confirmed that the robot passed the obstacle. Require one clear
        # camera reading before that visual detection may start another bypass.
        if not camera_blocking:
            self.camera_rearm_required = False

        lidar_ahead = lidar_status["blocking"]
        camera_can_trigger = camera_blocking and not self.camera_rearm_required
        if self.phase == "cruise":
            if not (lidar_ahead or camera_can_trigger):
                self.last_steering = 0.0
                return {"mode": "forward", "steering": 0.0}
            self._start_bypass(lidar_status, current_yaw_deg=current_yaw_deg)

        if self.phase != "cruise":
            if not self.tracker.locked and lidar_ahead:
                self.tracker.lock(
                    lidar_status,
                    commanded_heading_deg=self.commanded_heading_error_deg,
                )
            else:
                self.tracker.update(
                    lidar_status,
                    self.commanded_heading_error_deg,
                )

        # The camera starts a bypass, but lidar owns it afterward. Otherwise a
        # large bounding box still visible at the image edge repeatedly forces
        # a full turn-away command and makes the robot orbit the obstacle.
        obstacle_ahead = lidar_ahead

        if self.phase == "veer_out":
            side_distance = self._obstacle_side_distance(lidar_status)
            clearance_reached = (
                side_distance is not None
                and side_distance
                >= self.target_distance_m - SIDE_DISTANCE_DEADBAND_M
            )
            if not obstacle_ahead and clearance_reached and self.phase_cycles >= 2:
                self.phase = "follow_side"
                self.phase_cycles = 0
                self.wrap_countersteer_cycles = WRAP_COUNTERSTEER_CYCLES
                self.filtered_side_distance_m = side_distance
                self.last_side_angle_deg = self._obstacle_side_angle(lidar_status)
                return {
                    "mode": "follow_side",
                    "steering": self._follow_steering(side_distance),
                }
            elif (
                not obstacle_ahead
                and side_distance is None
                and self.phase_cycles >= SIDE_ACQUIRE_MAX_CYCLES
            ):
                return self._return_heading()
            self.last_steering = self._away_steering()
            return {"mode": "veer_out", "steering": self.last_steering}

        if self.phase == "follow_side":
            if obstacle_ahead:
                self.side_missing_cycles = 0
                self.side_passed_cycles = 0
                self.last_steering = self._away_steering()
                return {"mode": "follow_side", "steering": self.last_steering}

            side_distance = self._obstacle_side_distance(lidar_status)
            side_angle = self._obstacle_side_angle(lidar_status)
            if side_distance is None:
                self.side_missing_cycles += 1
                if self.side_missing_cycles >= SIDE_MISSING_CYCLES:
                    return self._return_heading()
                # Preserve the previous distance estimate through short scan
                # gaps while still straightening toward the original heading.
                return {
                    "mode": "follow_side",
                    "steering": self._follow_steering(
                        self.filtered_side_distance_m
                    ),
                }

            self.side_missing_cycles = 0
            self.last_side_angle_deg = side_angle
            track_observed = (
                not self.tracker.locked or self.tracker.missing_cycles == 0
            )
            if track_observed and self._side_has_passed(side_angle):
                self.side_passed_cycles += 1
                if self.side_passed_cycles >= SIDE_PASSED_CYCLES:
                    return self._return_heading()
            else:
                self.side_passed_cycles = 0
            return {
                "mode": "follow_side",
                "steering": self._follow_steering(side_distance),
            }

        if self.phase == "return_heading":
            if obstacle_ahead:
                self.heading_settle_cycles = 0
                self._start_bypass(
                    lidar_status,
                    current_yaw_deg=current_yaw_deg,
                    preserve_heading=True,
                )
                self.last_steering = self._away_steering()
                return {"mode": "veer_out", "steering": self.last_steering}

            if self.mpu_heading_required and not self.mpu_heading_live:
                self.last_steering = 0.0
                return {
                    "mode": "stopped_mpu_heading_unavailable",
                    "steering": 0.0,
                }

            completion_error_deg = (
                self.raw_mpu_heading_error_deg
                if self.mpu_heading_required
                else self.heading_error_deg
            )
            if abs(completion_error_deg) <= self.heading_tolerance_deg:
                self.heading_settle_cycles += 1
                self.last_steering = 0.0
                if self.heading_settle_cycles < MPU_HEADING_SETTLE_CYCLES:
                    return {
                        "mode": "stopped_orientation_settling",
                        "steering": 0.0,
                    }
                self.reset(camera_rearm_required=camera_blocking)
                return {"mode": "forward", "steering": 0.0}
            self.heading_settle_cycles = 0
            self.last_steering = self._heading_steering()
            return {"mode": "return_heading", "steering": self.last_steering}

        self.reset()
        return {"mode": "forward", "steering": 0.0}

    def completed_half_cycle(self, steering, yaw_deg_per_half_cycle):
        """Update commanded-heading estimate after a completed gait half-cycle."""
        if self.phase != "cruise":
            self.commanded_heading_error_deg += steering * yaw_deg_per_half_cycle
            if self.heading_source != "mpu6050":
                self.heading_error_deg = self.commanded_heading_error_deg
            self.phase_cycles += 1
            if self.phase == "follow_side" and self.wrap_countersteer_cycles > 0:
                self.wrap_countersteer_cycles -= 1

    def status(self):
        return {
            "phase": self.phase,
            "bypass_side": self.bypass_side,
            "target_side_distance_m": self.target_distance_m,
            "measured_side_distance_m": (
                round(self.filtered_side_distance_m, 3)
                if self.filtered_side_distance_m is not None else None
            ),
            "side_distance_error_m": (
                round(self.last_distance_error_m, 3)
                if self.last_distance_error_m is not None else None
            ),
            "tracked_side_angle_deg": (
                round(self.last_side_angle_deg, 1)
                if self.last_side_angle_deg is not None else None
            ),
            "estimated_heading_error_deg": round(self.heading_error_deg, 1),
            "raw_mpu_heading_error_deg": (
                round(self.raw_mpu_heading_error_deg, 1)
                if self.raw_mpu_heading_error_deg is not None else None
            ),
            "heading_source": self.heading_source,
            "bypass_start_yaw_deg": (
                round(self.start_yaw_deg, 1)
                if self.start_yaw_deg is not None else None
            ),
            "current_yaw_deg": (
                round(self.current_yaw_deg, 1)
                if self.current_yaw_deg is not None else None
            ),
            "mpu_yaw_sign": self.mpu_yaw_sign,
            "mpu_heading_required": self.mpu_heading_required,
            "mpu_heading_live": self.mpu_heading_live,
            "heading_settle_cycles": self.heading_settle_cycles,
            "steering_command": round(self.last_steering, 3),
            "wrap_countersteer_cycles": self.wrap_countersteer_cycles,
            "tracked_obstacle": self.tracker.status(),
        }


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
                    "assumed_width_m": args.object_width_m,
                }
            items.append(item)
        payload = {
            "sees_anything": bool(items),
            "nearby_obstacle": blocking,
            "motion": motion,
            "detections": items,
            "lidar": lidar_status,
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
    parser.add_argument(
        "--camera-buffers",
        type=int,
        default=DEFAULT_CAMERA_BUFFER_COUNT,
        help="Picamera2 request buffers (default: 6).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help="Camera/lidar dashboard port (default: 8000).",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the live camera/lidar dashboard.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--object-width-m",
        type=float,
        default=DEFAULT_HUMAN_WIDTH_M,
        help="Assumed obstacle width used for distance estimation (default: 0.45m).",
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
        default=0.70,
        help="Begin bypassing a forward lidar return at this range (default: 0.70m).",
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
        "--bypass-distance-m",
        type=float,
        default=0.32,
        help="Target side distance while passing an obstacle (default: 0.32m/about 1ft).",
    )
    parser.add_argument(
        "--bypass-max-steering",
        type=float,
        default=0.75,
        help="Maximum forward-steering command during a bypass (default: 0.75).",
    )
    parser.add_argument(
        "--bypass-follow-gain",
        type=float,
        default=0.80,
        help="Side-distance correction strength (default: 0.80).",
    )
    parser.add_argument(
        "--bypass-heading-tolerance-deg",
        type=float,
        default=1.5,
        help="Heading error allowed before completing a bypass (default: 1.5 degrees).",
    )
    parser.add_argument(
        "--lidar-emergency-distance-m",
        type=float,
        default=0.20,
        help="Hold position if an obstacle is critically close (default: 0.20m).",
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
    parser.add_argument("--focal-length-mm", type=float, default=AI_CAMERA_FOCAL_LENGTH_MM)
    parser.add_argument("--pixel-pitch-um", type=float, default=AI_CAMERA_PIXEL_PITCH_UM)
    parser.add_argument("--sensor-width-px", type=int, default=AI_CAMERA_SENSOR_WIDTH_PX)
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bbox-order", choices=("yx", "xy"))
    parser.add_argument("--postprocess", choices=("", "nanodet"), default=None)
    parser.add_argument("--preserve-aspect-ratio", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if (
        args.object_width_m <= 0.0
        or args.stop_distance_m <= 0.0
        or args.lidar_stop_distance_m <= 0.0
        or args.bypass_distance_m <= 0.0
        or args.lidar_emergency_distance_m <= 0.0
    ):
        parser.error("camera and lidar distance settings must be positive")
    if not 0.0 < args.camera_forward_half_width <= 1.0:
        parser.error("camera-forward-half-width must be within 0..1")
    if not 0.0 < args.lidar_forward_angle_deg < 90.0:
        parser.error("lidar-forward-angle-deg must be between 0 and 90")
    if not 0.0 < args.bypass_max_steering <= 1.0:
        parser.error("bypass-max-steering must be within 0..1")
    if args.bypass_follow_gain <= 0.0:
        parser.error("bypass-follow-gain must be positive")
    if not 0.0 < args.bypass_heading_tolerance_deg < 45.0:
        parser.error("bypass-heading-tolerance-deg must be between 0 and 45")
    if args.lidar_emergency_distance_m >= args.lidar_stop_distance_m:
        parser.error("lidar-emergency-distance-m must be below lidar-stop-distance-m")
    if args.bypass_distance_m <= args.lidar_emergency_distance_m:
        parser.error("bypass-distance-m must exceed lidar-emergency-distance-m")
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
    return args


def obstacle_distance(detection, frame_width, args):
    return estimate_distance_m(
        args.object_width_m,
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


def main():
    args = get_args()
    ensure_lidar_driver()
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
    blocking = False
    motion = "disengaged"
    camera_frame_ready = threading.Event()
    dashboard = None if args.no_web else AvoidanceDashboard(args.web_port)
    last_dashboard_frame = 0.0

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
        nonlocal detections
        outputs = imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            with detections_lock:
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
            return list(detections)

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

    def camera_callback(request):
        camera_frame_ready.set()
        draw_overlay(request)

    imx500.show_network_fw_progress_bar()
    picam2.pre_callback = camera_callback

    walk = None
    home_pose = None
    leveler = None
    walking_enabled = False
    next_swing = None
    status_printer = ObstacleStatusPrinter(args.print_interval)
    lidar_monitor = LidarObstacleMonitor(
        args.lidar_device,
        args.lidar_stop_distance_m,
        args.lidar_forward_angle_deg,
        args.lidar_forward_offset_deg,
    )
    bypass_controller = ObstacleBypassController(
        args.bypass_distance_m,
        args.bypass_max_steering,
        args.bypass_follow_gain,
        args.bypass_heading_tolerance_deg,
        args.lidar_emergency_distance_m,
        args.turn_direction,
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

    def current_heading_yaw():
        if leveler is None or not leveler.enabled:
            return None
        # Navigation decisions need the yaw after the completed gait cycle,
        # not a sample cached before that movement began.
        leveler.attitude(force=True)
        if leveler.read_errors:
            return None
        return leveler.yaw_degrees

    camera_started = False
    try:
        if dashboard is not None:
            try:
                dashboard_url = dashboard.start()
            except OSError as error:
                raise RuntimeError(
                    f"Unable to start camera/lidar dashboard on port "
                    f"{args.web_port}: {error}"
                ) from error
            print(f"Live camera and lidar dashboard: {dashboard_url}")

        print(
            f"Starting RPLIDAR C1 on {args.lidar_device} "
            f"at {LIDAR_BAUDRATE} baud."
        )
        lidar_monitor.start()
        print("Lidar ready; forward path monitoring is active.")

        print(f"Starting AI Camera with {args.camera_buffers} request buffers.")
        picam2.start(config, show_preview=not args.headless)
        camera_started = True
        if not camera_frame_ready.wait(timeout=CAMERA_START_TIMEOUT_S):
            raise RuntimeError(
                "AI Camera started but produced no frames within "
                f"{CAMERA_START_TIMEOUT_S:.0f} seconds. Run "
                "'rpicam-hello -t 5000' outside this program; if it shows the "
                "same V4L2/Unicam errors, reboot and check the camera cable and "
                "Raspberry Pi camera software."
            )
        if intrinsics.preserve_aspect_ratio:
            imx500.set_auto_aspect_ratio()

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
                    bypass_controller.reset()
                    motion = "disengaged"
                    if walk is not None:
                        walk.release_all()
                    home_pose = None
                    leveler = None
                    next_swing = None
                elif b"s" in keys:
                    walking_enabled = False
                    bypass_controller.reset()
                    motion = "stopped"
                    if home_pose is not None:
                        walk.hold_standing_pose(home_pose, neutral_attitude())
                    print("S pressed: walking stopped.")
                elif b"o" in keys:
                    walking_enabled = False
                    bypass_controller.reset()
                    stand_robot()
                    walk.hold_standing_pose(home_pose, neutral_attitude())
                    motion = "standing"
                    print("O pressed: standing still; walking remains disabled.")
                elif b"w" in keys:
                    stand_robot()
                    walking_enabled = True
                    motion = "walking_forward"
                    print("W pressed: forward walking enabled.")

                with detections_lock:
                    current = list(detections)
                frame_size = picam2.camera_configuration()["main"]["size"]
                _, _, camera_blocking = select_obstacle(
                    current, frame_size, args
                )
                lidar_status = lidar_monitor.snapshot()
                lidar_blocking = lidar_status["blocking"]
                lidar_unavailable = not lidar_status["live"]
                current_yaw_deg = current_heading_yaw()
                navigation = (
                    bypass_controller.update(
                        lidar_status,
                        camera_blocking,
                        current_yaw_deg=current_yaw_deg,
                    )
                    if walking_enabled
                    else {"mode": "forward", "steering": 0.0}
                )
                navigation_mode = navigation["mode"]
                steering = navigation["steering"]
                bypass_status = bypass_controller.status()
                lidar_status["bypass"] = bypass_status
                lidar_status["steering"] = round(steering, 3)
                blocking = (
                    camera_blocking
                    or lidar_blocking
                    or lidar_unavailable
                    or bypass_status["phase"] != "cruise"
                )
                if not walking_enabled:
                    motion = "stopped" if home_pose is not None else "disengaged"
                elif navigation_mode.startswith("stopped_"):
                    motion = navigation_mode
                elif navigation_mode == "veer_out":
                    motion = f"veering_{bypass_status['bypass_side']}"
                elif navigation_mode == "follow_side":
                    motion = f"passing_{bypass_status['bypass_side']}"
                elif navigation_mode == "return_heading":
                    motion = "merging_to_original_heading"
                else:
                    motion = "walking_forward"
                if dashboard is not None:
                    dashboard.update_state(
                        {
                            "motion": motion,
                            "steering": round(steering, 3),
                            "camera_live": camera_frame_ready.is_set(),
                            "detections": [
                                {
                                    "label": detection.label,
                                    "confidence": round(detection.confidence, 3),
                                    "box": [int(value) for value in detection.box],
                                }
                                for detection in current
                            ],
                            "lidar": lidar_status,
                        }
                    )
                status_printer.print(
                    current,
                    frame_size,
                    args,
                    blocking,
                    motion,
                    lidar_status,
                )

                if not walking_enabled:
                    continue
                if navigation_mode.startswith("stopped_"):
                    walk.hold_standing_pose(home_pose, neutral_attitude())
                    time.sleep(0.05)
                    continue

                direction = 1 if navigation_mode == "forward" else 4
                stance_tripod = (
                    walk.TRIPOD_B if next_swing == walk.TRIPOD_A else walk.TRIPOD_A
                )
                walk.walk_half_cycle(
                    home_pose,
                    next_swing,
                    stance_tripod,
                    direction=direction,
                    steering=steering,
                    hip_swing_scale=1.0,
                    interpolation_steps=args.walk_steps,
                    frame_delay=args.walk_frame_delay,
                    attitude=autonomous_walk_attitude(direction),
                )
                next_swing = stance_tripod
                bypass_controller.completed_half_cycle(
                    steering,
                    walk.WALK_STEER_YAW_DEG_PER_HALF_CYCLE,
                )
                leveler.clear_correction()
    except KeyboardInterrupt:
        print("\nCtrl+C: quitting and disengaging.")
        return 0
    except RuntimeError as error:
        print(f"Startup/runtime failure: {error}", file=sys.stderr)
        return 2
    finally:
        try:
            lidar_monitor.stop()
        except Exception as error:
            print(f"Lidar shutdown failed: {error}", file=sys.stderr)
        try:
            if camera_started:
                picam2.stop()
        except Exception as error:
            print(f"Camera shutdown failed: {error}", file=sys.stderr)
        finally:
            try:
                picam2.close()
            except Exception as error:
                print(f"Camera close failed: {error}", file=sys.stderr)
            if walk is not None:
                walk.release_all()
            if dashboard is not None:
                dashboard.stop()


if __name__ == "__main__":
    raise SystemExit(main())
