"""Walk forward and turn around obstacles seen by the AI Camera and lidar.

The robot stands up when the program starts, but does not walk until W is
pressed. W starts/resumes forward walking, S stops all walking while holding
the standing pose, P disengages the servos, and O stands up again without
walking. Camera monitoring continues throughout. Ctrl+C exits the whole
program.

The camera identifies objects while the RPLIDAR C1 measures the physical path
in front of the robot. Lidar readings also choose the clearer turn direction.
"""

import argparse
import asyncio
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from functools import lru_cache

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


def signed_lidar_angle(angle):
    """Convert 0..360 degrees to -180..180, with forward at zero."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


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

        turn_direction = None
        if nearest is not None:
            obstacle_angle = relative_angle(nearest)
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
            "turn_direction": turn_direction,
            "forward_half_angle_deg": self.forward_angle_deg,
            "error": str(self.error) if self.error is not None else None,
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
    parser.add_argument("--leveling-scale", type=float, default=0.65)
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
    ):
        parser.error("camera and lidar distance settings must be positive")
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
    if not 0.0 <= args.leveling_scale <= 1.0:
        parser.error("leveling-scale must be within 0..1")
    if args.focal_length_mm <= 0.0 or args.pixel_pitch_um <= 0.0:
        parser.error("camera measurements must be positive")
    if args.sensor_width_px <= 0:
        parser.error("sensor-width-px must be positive")
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
        buffer_count=12,
    )
    detections = []
    blocking = False
    motion = "disengaged"

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
            return detections
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
        detections = parsed[:args.max_detections]
        return detections

    def draw_overlay(request, stream="main"):
        with MappedArray(request, stream) as mapped:
            height, width = mapped.array.shape[:2]
            color = (0, 0, 255) if blocking else (0, 255, 0)
            cv2.putText(mapped.array, motion, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2)
            for detection in detections:
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

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=not args.headless)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    if not args.headless:
        picam2.pre_callback = draw_overlay

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

    def autonomous_attitude(direction=0):
        level_attitude = leveler.attitude()
        roll_scale, pitch_scale = walk.moving_level_scales(direction)
        return {
            "roll": level_attitude["roll"] * roll_scale * args.leveling_scale,
            "pitch": level_attitude["pitch"] * pitch_scale * args.leveling_scale,
        }

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
                        walk.hold_standing_pose(home_pose, autonomous_attitude())
                    print("S pressed: walking stopped.")
                elif b"o" in keys:
                    walking_enabled = False
                    avoiding = False
                    clear_frames = 0
                    stand_robot()
                    walk.hold_standing_pose(home_pose, autonomous_attitude())
                    motion = "standing"
                    print("O pressed: standing still; walking remains disabled.")
                elif b"w" in keys:
                    stand_robot()
                    walking_enabled = True
                    motion = "walking_forward"
                    print("W pressed: forward walking enabled.")

                metadata = picam2.capture_metadata()
                current = parse_detections(metadata)
                frame_size = picam2.camera_configuration()["main"]["size"]
                _, _, camera_blocking = select_obstacle(
                    current, frame_size, args
                )
                lidar_status = lidar_monitor.snapshot()
                lidar_blocking = lidar_status["blocking"]
                lidar_unavailable = not lidar_status["live"]
                blocking_now = camera_blocking or lidar_blocking or lidar_unavailable

                if walking_enabled:
                    if blocking_now:
                        if not avoiding:
                            avoid_turn_direction = (
                                lidar_status["turn_direction"]
                                or args.turn_direction
                            )
                        avoiding = True
                        clear_frames = 0
                    elif avoiding:
                        clear_frames += 1
                        if clear_frames >= args.clear_frames:
                            avoiding = False
                            clear_frames = 0

                blocking = blocking_now
                if not walking_enabled:
                    motion = "stopped" if home_pose is not None else "disengaged"
                elif lidar_unavailable:
                    motion = "stopped_lidar_unavailable"
                elif avoiding:
                    motion = f"turning_{avoid_turn_direction}"
                else:
                    motion = "walking_forward"
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
                if lidar_unavailable:
                    walk.hold_standing_pose(home_pose, autonomous_attitude())
                    time.sleep(0.05)
                    continue

                direction = (
                    (-3 if avoid_turn_direction == "left" else 3)
                    if avoiding else 1
                )
                stance_tripod = (
                    walk.TRIPOD_B if next_swing == walk.TRIPOD_A else walk.TRIPOD_A
                )
                walk.walk_half_cycle(
                    home_pose,
                    next_swing,
                    stance_tripod,
                    direction=direction,
                    hip_swing_scale=args.turn_scale if avoiding else 1.0,
                    interpolation_steps=args.walk_steps,
                    frame_delay=args.walk_frame_delay,
                    attitude_provider=lambda: autonomous_attitude(direction),
                )
                next_swing = stance_tripod
    except KeyboardInterrupt:
        print("\nCtrl+C: quitting and disengaging.")
        return 0
    finally:
        lidar_monitor.stop()
        picam2.stop()
        if walk is not None:
            walk.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
