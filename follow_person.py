"""Use the Raspberry Pi AI Camera to make the hexapod follow a person.

Press Space in the terminal to pause/resume following. Press Q or Escape to
quit. The camera preview stays active while following is paused.
"""

import argparse
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from functools import lru_cache

from ai_camera_object_detection import (
    DEFAULT_MODEL,
    Detection,
    DetectionPrinter,
    import_camera_stack,
    load_labels,
    rectangle_to_box,
)
import controller_walk as walk


@dataclass
class FollowCommand:
    direction: int
    steering: float
    description: str


class KeyboardControls:
    """Nonblocking single-key terminal input with automatic terminal restore."""

    def __init__(self):
        self.fd = None
        self.original_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.original_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        else:
            print("Warning: stdin is not a terminal; Space/Q controls are disabled.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_settings)

    def read_keys(self):
        keys = []
        if self.fd is None:
            return keys
        while select.select([sys.stdin], [], [], 0)[0]:
            keys.append(os.read(self.fd, 1))
        return keys


def get_args():
    parser = argparse.ArgumentParser(
        description="Follow a detected person using the hexapod tripod gait."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--labels", help="Optional labels file, one label per line.")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--fps", type=int, help="Override camera inference frame rate.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.25,
        help="Seconds between JSON detection-location updates (default: 0.25).",
    )
    parser.add_argument(
        "--center-deadzone",
        type=float,
        default=0.25,
        help="Horizontal normalized error treated as centered (default: 0.25).",
    )
    parser.add_argument(
        "--turn-in-place-error",
        type=float,
        default=0.68,
        help="Error above which the robot turns in place (default: 0.68).",
    )
    parser.add_argument(
        "--stop-area",
        type=float,
        default=0.30,
        help="Stop approaching when person box fills this frame fraction (default: 0.30).",
    )
    parser.add_argument(
        "--max-steering",
        type=float,
        default=0.25,
        help="Maximum walking steering correction (default: 0.25).",
    )
    parser.add_argument(
        "--turn-scale",
        type=float,
        default=0.30,
        help="Scale applied to autonomous in-place hip swing (default: 0.30).",
    )
    parser.add_argument(
        "--tracking-alpha",
        type=float,
        default=0.20,
        help="Target smoothing factor; lower is calmer (default: 0.20).",
    )
    parser.add_argument(
        "--lost-timeout",
        type=float,
        default=0.6,
        help="Stop after this many seconds without a person (default: 0.6).",
    )
    parser.add_argument(
        "--bbox-normalization", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--bbox-order", choices=("yx", "xy"))
    parser.add_argument("--postprocess", choices=("", "nanodet"), default=None)
    parser.add_argument(
        "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    if not 0.0 < args.center_deadzone < args.turn_in_place_error <= 1.0:
        parser.error("deadzone must be below turn-in-place-error, both within 0..1")
    if not 0.0 < args.stop_area < 1.0:
        parser.error("stop-area must be within 0..1")
    if not 0.0 <= args.max_steering <= 1.0:
        parser.error("max-steering must be within 0..1")
    if not 0.0 < args.turn_scale <= 1.0:
        parser.error("turn-scale must be within 0..1")
    if not 0.0 < args.tracking_alpha <= 1.0:
        parser.error("tracking-alpha must be within 0..1")
    if args.print_interval < 0.0:
        parser.error("print-interval cannot be negative")
    return args


def choose_person(detections):
    people = [item for item in detections if item.label.lower() == "person"]
    if not people:
        return None
    # Prefer the largest visible person, then confidence, to avoid switching
    # between a nearby target and small background detections.
    return max(people, key=lambda item: (item.box[2] * item.box[3], item.confidence))


def smooth_detection(previous, current, alpha):
    """Low-pass filter a target so detection jitter cannot reverse the gait."""
    if previous is None:
        return current

    def blend(old, new):
        return old + alpha * (new - old)

    box = tuple(blend(old, new) for old, new in zip(previous.box, current.box))
    center = tuple(
        blend(old, new) for old, new in zip(previous.center, current.center)
    )
    return Detection(
        label=current.label,
        category=current.category,
        confidence=current.confidence,
        box=box,
        center=center,
    )


def command_for_person(person, frame_size, args):
    frame_w, frame_h = frame_size
    center_error = (person.center[0] - frame_w / 2) / (frame_w / 2)
    area_ratio = (person.box[2] * person.box[3]) / (frame_w * frame_h)

    # Distance safety takes priority over centering. Once close enough, do not
    # rotate back and forth just because the box jitters horizontally.
    if area_ratio >= args.stop_area:
        return FollowCommand(0, 0.0, "close enough; hold still"), center_error, area_ratio

    if abs(center_error) >= args.turn_in_place_error:
        # Camera +x is screen-right. The physical in-place turn direction on
        # this robot is opposite controller_walk's direction-name convention.
        direction = 3 if center_error > 0 else -3
        side = "right" if center_error > 0 else "left"
        return FollowCommand(direction, 0.0, f"turn in place {side}"), center_error, area_ratio

    if abs(center_error) <= args.center_deadzone:
        # Direction 1 is controller_walk's plain D-pad-forward gait.
        return FollowCommand(1, 0.0, "walk straight forward"), center_error, area_ratio

    usable_range = args.turn_in_place_error - args.center_deadzone
    steering = (abs(center_error) - args.center_deadzone) / usable_range
    steering = min(args.max_steering, steering * args.max_steering)
    # The physical steering bias is opposite the camera's horizontal error on
    # this robot. In-place turning has its own separately calibrated sign.
    steering *= -1.0 if center_error > 0 else 1.0
    side = "right" if center_error > 0 else "left"
    return FollowCommand(4, steering, f"walk forward; steer {side}"), center_error, area_ratio


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
    target = None
    following = True
    status_text = "FOLLOWING: looking for person"
    detection_printer = DetectionPrinter(args.print_interval, [])

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
            color = (0, 255, 0) if following else (0, 165, 255)
            cv2.line(mapped.array, (width // 2, 0), (width // 2, height), color, 1)
            cv2.putText(mapped.array, status_text, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            if target is not None:
                x, y, box_w, box_h = target.box
                cv2.rectangle(mapped.array, (x, y), (x + box_w, y + box_h), color, 3)
                cv2.putText(mapped.array, f"person {target.confidence:.2f}",
                            (x + 4, max(50, y + 22)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=not args.headless)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    if not args.headless:
        picam2.pre_callback = draw_overlay

    home_pose = None
    last_seen = 0.0
    next_swing = walk.TRIPOD_A
    last_command = None
    print("Camera ready. Standing up; keep clear of the robot.")
    try:
        walk.validate_ik_constants()
        home_pose = walk.run_stand_up_sequence()
        walk.hold_standing_pose(home_pose)
        print("Following enabled. Space=pause/resume, Q or Escape=quit.")
        with KeyboardControls() as keyboard:
            while True:
                for key in keyboard.read_keys():
                    if key == b" ":
                        following = not following
                        if not following:
                            walk.hold_standing_pose(home_pose)
                        print("Following enabled." if following else "Following paused.")
                    elif key.lower() == b"q" or key == b"\x1b":
                        return 0

                metadata = picam2.capture_metadata()
                current_target = choose_person(parse_detections(metadata))
                frame_size = picam2.camera_configuration()["main"]["size"]
                detection_printer.print(detections, frame_size)
                now = time.monotonic()
                if current_target is not None:
                    target = smooth_detection(
                        target,
                        current_target,
                        args.tracking_alpha,
                    )
                    last_seen = now
                elif now - last_seen > args.lost_timeout:
                    target = None

                if not following:
                    status_text = "PAUSED (Space to follow)"
                    continue
                if target is None:
                    status_text = "FOLLOWING: looking for person"
                    walk.hold_standing_pose(home_pose)
                    last_command = None
                    continue

                command, error, area = command_for_person(target, frame_size, args)
                status_text = f"{command.description}  error={error:+.2f} area={area:.2f}"
                command_key = (command.direction, round(command.steering, 2))
                if command_key != last_command:
                    print(status_text)
                    last_command = command_key
                if command.direction == 0:
                    walk.hold_standing_pose(home_pose)
                    continue

                stance_tripod = walk.TRIPOD_B if next_swing == walk.TRIPOD_A else walk.TRIPOD_A
                walk.walk_half_cycle(home_pose, next_swing, stance_tripod,
                                     direction=command.direction,
                                     steering=command.steering,
                                     hip_swing_scale=(
                                         args.turn_scale
                                         if abs(command.direction) == 3
                                         else 1.0
                                     ))
                next_swing = stance_tripod
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    finally:
        if home_pose is not None:
            walk.hold_standing_pose(home_pose)
        picam2.stop()
        walk.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
