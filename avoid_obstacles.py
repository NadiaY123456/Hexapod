"""Walk forward and turn around close obstacles seen by the AI Camera.

The robot stands up when the program starts, but does not walk until W is
pressed. W starts/resumes forward walking, S stops all walking while holding
the standing pose, P disengages the servos, and O stands up again without
walking. Camera monitoring continues throughout. Ctrl+C exits the whole
program.

Distance is estimated from bounding-box width, as in human_distance.py. Since
different objects have different real widths, --object-width-m should be tuned
for the obstacles expected in the robot's environment.
"""

import argparse
import json
import os
import select
import sys
import termios
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
    """Print every AI Camera detection and its estimated distance as JSON."""

    def __init__(self, interval):
        self.interval = interval
        self.last_print = 0.0

    def print(self, detections, frame_size, args, blocking, motion):
        now = time.monotonic()
        if now - self.last_print < self.interval:
            return
        self.last_print = now
        frame_w, frame_h = frame_size
        items = []
        for detection in detections:
            distance_m = obstacle_distance(detection, frame_w, args)
            center_x, center_y = detection.center
            item = {
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "box": [int(value) for value in detection.box],
                "center": [int(center_x), int(center_y)],
                "offset": [
                    round((center_x - frame_w / 2) / (frame_w / 2), 3),
                    round((center_y - frame_h / 2) / (frame_h / 2), 3),
                ],
                "nearby": distance_m is not None
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

    if args.object_width_m <= 0.0 or args.stop_distance_m <= 0.0:
        parser.error("object-width-m and stop-distance-m must be positive")
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


def select_obstacle(detections, frame_size, args):
    """Return the nearest detection and whether any detection is nearby."""
    candidates = []
    for detection in detections:
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
                nearby = distance_m is not None and distance_m <= args.stop_distance_m
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
    clear_frames = 0
    next_swing = None
    status_printer = ObstacleStatusPrinter(args.print_interval)

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
                _, _, blocking_now = select_obstacle(
                    current, frame_size, args
                )

                if walking_enabled:
                    if blocking_now:
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
                elif avoiding:
                    motion = f"turning_{args.turn_direction}"
                else:
                    motion = "walking_forward"
                status_printer.print(current, frame_size, args, blocking, motion)

                if not walking_enabled:
                    continue

                direction = (
                    (-3 if args.turn_direction == "left" else 3)
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
        picam2.stop()
        if walk is not None:
            walk.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
