"""Use the Raspberry Pi AI Camera to make the hexapod follow a person.

Press Space in the terminal to pause/resume following. Press Q or Escape to
quit. The camera preview stays active while following is paused. If a moving
person is briefly lost, the robot continues along the last measured image-
motion vector for up to the configured search timeout.
"""

import argparse
import math
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


class TargetMotionTracker:
    """Filter target motion and extrapolate through short detection dropouts."""

    def __init__(self, position_alpha, velocity_alpha, velocity_deadzone, max_speed):
        self.position_alpha = position_alpha
        self.velocity_alpha = velocity_alpha
        self.velocity_deadzone = velocity_deadzone
        self.max_speed = max_speed
        self.detection = None
        self.velocity = (0.0, 0.0)
        self.last_raw_center = None
        self.last_update = None
        self.last_seen = None

    def update(self, current, frame_size, now):
        frame_w, frame_h = frame_size
        if self.last_raw_center is not None and self.last_update is not None:
            dt = max(0.001, now - self.last_update)
            instant_velocity = (
                (current.center[0] - self.last_raw_center[0]) / (frame_w / 2) / dt,
                (current.center[1] - self.last_raw_center[1]) / (frame_h / 2) / dt,
            )
            speed = math.hypot(*instant_velocity)
            if speed > self.max_speed:
                scale = self.max_speed / speed
                instant_velocity = (
                    instant_velocity[0] * scale,
                    instant_velocity[1] * scale,
                )
            if speed < self.velocity_deadzone:
                instant_velocity = (0.0, 0.0)
            self.velocity = tuple(
                old + self.velocity_alpha * (new - old)
                for old, new in zip(self.velocity, instant_velocity)
            )
            if math.hypot(*self.velocity) < self.velocity_deadzone:
                self.velocity = (0.0, 0.0)

        self.detection = smooth_detection(
            self.detection,
            current,
            self.position_alpha,
        )
        self.last_raw_center = current.center
        self.last_update = now
        self.last_seen = now
        return self.detection

    def age(self, now):
        return None if self.last_seen is None else max(0.0, now - self.last_seen)

    def is_moving(self):
        return math.hypot(*self.velocity) >= self.velocity_deadzone

    def reset(self):
        self.detection = None
        self.velocity = (0.0, 0.0)
        self.last_raw_center = None
        self.last_update = None
        self.last_seen = None

    def predict(self, frame_size, now):
        if self.detection is None or self.last_seen is None:
            return None

        frame_w, frame_h = frame_size
        elapsed = max(0.0, now - self.last_seen)
        shift_x = self.velocity[0] * elapsed * (frame_w / 2)
        shift_y = self.velocity[1] * elapsed * (frame_h / 2)
        center_x = max(-frame_w * 0.25, min(frame_w * 1.25,
                                            self.detection.center[0] + shift_x))
        center_y = max(-frame_h * 0.25, min(frame_h * 1.25,
                                            self.detection.center[1] + shift_y))
        box_x, box_y, box_w, box_h = self.detection.box
        return Detection(
            label=self.detection.label,
            category=self.detection.category,
            confidence=self.detection.confidence,
            box=(
                box_x + center_x - self.detection.center[0],
                box_y + center_y - self.detection.center[1],
                box_w,
                box_h,
            ),
            center=(center_x, center_y),
        )


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


def get_args(default_target_label="person"):
    parser = argparse.ArgumentParser(
        description=(
            f"Follow a detected {default_target_label} using the hexapod tripod gait."
        )
    )
    parser.add_argument(
        "--target-label",
        default=default_target_label,
        help=f"Detection label to follow (default: {default_target_label}).",
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
        default=0.20,
        help="Horizontal normalized error treated as centered (default: 0.20).",
    )
    parser.add_argument(
        "--turn-in-place-error",
        type=float,
        default=0.58,
        help="Error above which the robot turns in place (default: 0.58).",
    )
    parser.add_argument(
        "--stop-area",
        type=float,
        default=0.30,
        help="Near edge of the comfortable distance band (default: 0.30).",
    )
    parser.add_argument(
        "--back-away-area",
        type=float,
        default=0.50,
        help=(
            "First close-range requirement: target box must fill at least "
            "this frame fraction before reversing (default: 0.50)."
        ),
    )
    parser.add_argument(
        "--back-away-height",
        type=float,
        default=0.82,
        help=(
            "Second close-range requirement: target box must fill at least "
            "this fraction of frame height before reversing (default: 0.82)."
        ),
    )
    parser.add_argument(
        "--close-turn-error",
        type=float,
        default=0.46,
        help="At close range, turn only beyond this horizontal error (default: 0.46).",
    )
    parser.add_argument(
        "--max-steering",
        type=float,
        default=0.30,
        help="Maximum walking steering correction (default: 0.30).",
    )
    parser.add_argument(
        "--turn-scale",
        type=float,
        default=0.50,
        help="Scale applied to autonomous in-place hip swing (default: 0.50).",
    )
    parser.add_argument(
        "--tracking-alpha",
        type=float,
        default=0.40,
        help="Target smoothing factor; higher reacts faster (default: 0.40).",
    )
    parser.add_argument(
        "--velocity-alpha",
        type=float,
        default=0.45,
        help="Target velocity smoothing factor (default: 0.45).",
    )
    parser.add_argument(
        "--velocity-deadzone",
        type=float,
        default=0.08,
        help="Ignore slower normalized image motion as detector jitter (default: 0.08).",
    )
    parser.add_argument(
        "--max-target-speed",
        type=float,
        default=3.0,
        help="Maximum normalized image velocity used for prediction (default: 3.0).",
    )
    parser.add_argument(
        "--lost-timeout",
        type=float,
        default=0.25,
        help="Time before a predicted dropout is reported as lost (default: 0.25).",
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=3.0,
        help=(
            "Continue following the last-seen motion vector after detection "
            "is lost (default: 3.0 seconds)."
        ),
    )
    parser.add_argument(
        "--walk-steps",
        type=int,
        default=7,
        help="Interpolation steps per autonomous half-cycle (default: 7).",
    )
    parser.add_argument(
        "--walk-frame-delay",
        type=float,
        default=0.022,
        help="Seconds between autonomous gait frames (default: 0.022).",
    )
    parser.add_argument(
        "--leveling-scale",
        type=float,
        default=0.65,
        help="Scale applied to live MPU6050 walking correction (default: 0.65).",
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
    if not 0.0 < args.stop_area < args.back_away_area < 1.0:
        parser.error("stop-area must be below back-away-area, both within 0..1")
    if not 0.0 < args.back_away_height <= 1.0:
        parser.error("back-away-height must be within 0..1")
    if not args.center_deadzone < args.close_turn_error <= 1.0:
        parser.error("close-turn-error must exceed center-deadzone and be at most 1")
    if not 0.0 <= args.max_steering <= 1.0:
        parser.error("max-steering must be within 0..1")
    if not 0.0 < args.turn_scale <= 1.0:
        parser.error("turn-scale must be within 0..1")
    if not 0.0 < args.tracking_alpha <= 1.0:
        parser.error("tracking-alpha must be within 0..1")
    if not 0.0 < args.velocity_alpha <= 1.0:
        parser.error("velocity-alpha must be within 0..1")
    if args.velocity_deadzone < 0.0:
        parser.error("velocity-deadzone cannot be negative")
    if args.max_target_speed <= 0.0:
        parser.error("max-target-speed must be positive")
    if args.lost_timeout < 0.0:
        parser.error("lost-timeout cannot be negative")
    if args.search_timeout < args.lost_timeout:
        parser.error("search-timeout must be at least lost-timeout")
    if args.walk_steps < 1:
        parser.error("walk-steps must be at least 1")
    if args.walk_frame_delay < 0.0:
        parser.error("walk-frame-delay cannot be negative")
    if not 0.0 <= args.leveling_scale <= 1.0:
        parser.error("leveling-scale must be within 0..1")
    if args.print_interval < 0.0:
        parser.error("print-interval cannot be negative")
    return args


def choose_target(detections, target_label):
    matches = [
        item for item in detections
        if item.label.lower() == target_label.lower()
    ]
    if not matches:
        return None
    # Prefer the largest visible target, then confidence, to avoid switching
    # between a nearby target and small background detections.
    return max(matches, key=lambda item: (item.box[2] * item.box[3], item.confidence))


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
    height_ratio = person.box[3] / frame_h

    # Reverse only with strong, orientation-resistant evidence that the person
    # is extremely close. Area alone changes when a person turns or walks
    # sideways, while vertical image height mostly changes with distance.
    # Requiring the person to be centered also prevents backing away in
    # response to somebody simply crossing the camera's field of view.
    too_close = (
        area_ratio >= args.back_away_area
        and height_ratio >= args.back_away_height
    )
    if too_close:
        if abs(center_error) <= args.center_deadzone:
            return (
                FollowCommand(
                    -1,
                    0.0,
                    f"very close (height={height_ratio:.2f}); walk straight backward",
                ),
                center_error,
                area_ratio,
            )
        direction = 3 if center_error > 0 else -3
        side = "right" if center_error > 0 else "left"
        return (
            FollowCommand(
                direction,
                0.0,
                f"very close but off center; turn {side} before backing",
            ),
            center_error,
            area_ratio,
        )

    # Hold within the comfortable distance band. Allow an in-place correction
    # only if the person is clearly far to one side.
    if area_ratio >= args.stop_area:
        if abs(center_error) >= args.close_turn_error:
            direction = 3 if center_error > 0 else -3
            side = "right" if center_error > 0 else "left"
            return (
                FollowCommand(direction, 0.0, f"close; gently turn {side}"),
                center_error,
                area_ratio,
            )
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
    # Walking steering follows controller_walk's left-stick x convention:
    # positive steers physically right and negative steers physically left.
    # In-place turning has its own separately calibrated direction mapping.
    steering *= 1.0 if center_error > 0 else -1.0
    side = "right" if center_error > 0 else "left"
    return FollowCommand(4, steering, f"walk forward; steer {side}"), center_error, area_ratio


def main(default_target_label="person"):
    args = get_args(default_target_label)
    target_name = args.target_label.strip().lower()
    if not target_name:
        print("Target label cannot be empty.", file=sys.stderr)
        return 2
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
    target_is_predicted = False
    following = True
    status_text = f"FOLLOWING: looking for {target_name}"
    detection_printer = DetectionPrinter(args.print_interval, [])
    motion_tracker = TargetMotionTracker(
        args.tracking_alpha,
        args.velocity_alpha,
        args.velocity_deadzone,
        args.max_target_speed,
    )

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
                x, y, box_w, box_h = (
                    int(round(value)) for value in target.box
                )
                cv2.rectangle(mapped.array, (x, y), (x + box_w, y + box_h), color, 3)
                cv2.putText(mapped.array, f"{target_name} {target.confidence:.2f}",
                            (x + 4, max(50, y + 22)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
                velocity_x, velocity_y = motion_tracker.velocity
                vector_end = (
                    int(target.center[0] + velocity_x * width * 0.18),
                    int(target.center[1] + velocity_y * height * 0.18),
                )
                cv2.arrowedLine(
                    mapped.array,
                    (int(target.center[0]), int(target.center[1])),
                    vector_end,
                    (255, 255, 0) if target_is_predicted else color,
                    2,
                    tipLength=0.25,
                )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=not args.headless)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    if not args.headless:
        picam2.pre_callback = draw_overlay

    home_pose = None
    leveler = None
    next_swing = walk.TRIPOD_A
    last_command = None
    print("Camera ready. Standing up; keep clear of the robot.")
    try:
        walk.validate_ik_constants()
        home_pose = walk.run_stand_up_sequence()
        walk.hold_standing_pose(home_pose)
        leveler = walk.LevelingController()

        def autonomous_attitude(direction=0):
            level_attitude = leveler.attitude()
            roll_scale, pitch_scale = walk.moving_level_scales(direction)
            return {
                "roll": level_attitude["roll"] * roll_scale * args.leveling_scale,
                "pitch": level_attitude["pitch"] * pitch_scale * args.leveling_scale,
            }

        print("Following enabled. Space=pause/resume, Q or Escape=quit.")
        with KeyboardControls() as keyboard:
            while True:
                for key in keyboard.read_keys():
                    if key == b" ":
                        following = not following
                        if not following:
                            walk.hold_standing_pose(home_pose, autonomous_attitude())
                        print("Following enabled." if following else "Following paused.")
                    elif key.lower() == b"q" or key == b"\x1b":
                        return 0

                metadata = picam2.capture_metadata()
                current_target = choose_target(
                    parse_detections(metadata),
                    target_name,
                )
                frame_size = picam2.camera_configuration()["main"]["size"]
                detection_printer.print(detections, frame_size)
                now = time.monotonic()
                if current_target is not None:
                    target = motion_tracker.update(
                        current_target,
                        frame_size,
                        now,
                    )
                    target_is_predicted = False
                else:
                    target_age = motion_tracker.age(now)
                    prediction_timeout = (
                        args.search_timeout
                        if motion_tracker.is_moving()
                        else args.lost_timeout
                    )
                    if target_age is not None and target_age <= prediction_timeout:
                        target = motion_tracker.predict(frame_size, now)
                        target_is_predicted = target is not None
                    else:
                        target = None
                        target_is_predicted = False
                        motion_tracker.reset()

                if not following:
                    status_text = "PAUSED (Space to follow)"
                    continue
                if target is None:
                    status_text = f"FOLLOWING: looking for {target_name}"
                    walk.hold_standing_pose(home_pose, autonomous_attitude())
                    last_command = None
                    continue

                command, error, area = command_for_person(target, frame_size, args)
                velocity_x, velocity_y = motion_tracker.velocity
                if target_is_predicted:
                    target_age = motion_tracker.age(now)
                    prediction_state = (
                        "detection gap"
                        if target_age is not None and target_age <= args.lost_timeout
                        else "target lost"
                    )
                    if command.direction in (-1, -4):
                        # Never reverse toward unseen terrain using only a stale
                        # close-range box. Wait for a real detection instead.
                        command = FollowCommand(
                            0,
                            0.0,
                            f"{prediction_state}; too-close detection lost; hold still",
                        )
                    else:
                        command.description = (
                            f"{prediction_state}; predict motion; {command.description}"
                        )

                status_text = (
                    f"{command.description}  error={error:+.2f} area={area:.2f} "
                    f"velocity=({velocity_x:+.2f},{velocity_y:+.2f})"
                )
                command_key = (
                    command.direction,
                    round(command.steering, 2),
                    round(velocity_x, 1),
                    round(velocity_y, 1),
                    target_is_predicted,
                )
                if command_key != last_command:
                    print(status_text)
                    last_command = command_key
                if command.direction == 0:
                    walk.hold_standing_pose(home_pose, autonomous_attitude())
                    continue

                stance_tripod = walk.TRIPOD_B if next_swing == walk.TRIPOD_A else walk.TRIPOD_A
                walk.walk_half_cycle(home_pose, next_swing, stance_tripod,
                                     direction=command.direction,
                                     steering=command.steering,
                                     hip_swing_scale=(
                                         args.turn_scale
                                         if abs(command.direction) == 3
                                         else 1.0
                                     ),
                                     interpolation_steps=args.walk_steps,
                                     frame_delay=args.walk_frame_delay,
                                     attitude_provider=lambda: autonomous_attitude(
                                         command.direction
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
