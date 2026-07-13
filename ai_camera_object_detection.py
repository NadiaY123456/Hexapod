import argparse
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache


DEFAULT_MODEL = (
    "/usr/share/imx500-models/"
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)
AI_CAMERA_FOCAL_LENGTH_MM = 4.74
AI_CAMERA_PIXEL_PITCH_UM = 1.55
AI_CAMERA_SENSOR_WIDTH_PX = 4056
DEFAULT_HUMAN_WIDTH_M = 0.45


@dataclass
class Detection:
    label: str
    category: int
    confidence: float
    box: tuple
    center: tuple


def focal_length_pixels(
    frame_width_px,
    focal_length_mm=AI_CAMERA_FOCAL_LENGTH_MM,
    pixel_pitch_um=AI_CAMERA_PIXEL_PITCH_UM,
    sensor_width_px=AI_CAMERA_SENSOR_WIDTH_PX,
):
    native_focal_px = focal_length_mm / (pixel_pitch_um / 1000.0)
    return native_focal_px * frame_width_px / sensor_width_px


def estimate_distance_m(
    object_width_m,
    box_width_px,
    frame_width_px,
    focal_length_mm=AI_CAMERA_FOCAL_LENGTH_MM,
    pixel_pitch_um=AI_CAMERA_PIXEL_PITCH_UM,
    sensor_width_px=AI_CAMERA_SENSOR_WIDTH_PX,
):
    if object_width_m <= 0.0 or box_width_px <= 0 or frame_width_px <= 0:
        return None
    focal_px = focal_length_pixels(
        frame_width_px,
        focal_length_mm,
        pixel_pitch_um,
        sensor_width_px,
    )
    return object_width_m * focal_px / box_width_px


class DetectionPrinter:
    def __init__(
        self,
        min_interval,
        target_labels,
        distance_target=None,
        object_width_m=None,
        focal_length_mm=AI_CAMERA_FOCAL_LENGTH_MM,
        pixel_pitch_um=AI_CAMERA_PIXEL_PITCH_UM,
        sensor_width_px=AI_CAMERA_SENSOR_WIDTH_PX,
    ):
        self.min_interval = min_interval
        self.target_labels = {label.lower() for label in target_labels}
        self.distance_target = distance_target.lower() if distance_target else None
        self.object_width_m = object_width_m
        self.focal_length_mm = focal_length_mm
        self.pixel_pitch_um = pixel_pitch_um
        self.sensor_width_px = sensor_width_px
        self.last_print_time = 0.0

    def should_print(self):
        now = time.monotonic()
        if now - self.last_print_time < self.min_interval:
            return False
        self.last_print_time = now
        return True

    def print(self, detections, frame_size):
        if not self.should_print():
            return

        selected = [
            detection
            for detection in detections
            if not self.target_labels
            or detection.label.lower() in self.target_labels
        ]
        if not selected:
            print("detections: none")
            return

        frame_w, frame_h = frame_size
        payload = []
        for detection in selected:
            center_x, center_y = detection.center
            box_width_px = int(detection.box[2])
            item = {
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "box": [int(value) for value in detection.box],
                "width_px": box_width_px,
                "center": [int(center_x), int(center_y)],
                "offset": [
                    round((center_x - frame_w / 2) / (frame_w / 2), 3),
                    round((center_y - frame_h / 2) / (frame_h / 2), 3),
                ],
            }
            if (
                self.distance_target == detection.label.lower()
                and self.object_width_m is not None
            ):
                focal_px = focal_length_pixels(
                    frame_w,
                    self.focal_length_mm,
                    self.pixel_pitch_um,
                    self.sensor_width_px,
                )
                distance_m = estimate_distance_m(
                    self.object_width_m,
                    box_width_px,
                    frame_w,
                    self.focal_length_mm,
                    self.pixel_pitch_um,
                    self.sensor_width_px,
                )
                if distance_m is not None:
                    item["distance"] = {
                        "meters": round(distance_m, 2),
                        "feet": round(distance_m * 3.28084, 2),
                        "assumed_width_m": self.object_width_m,
                        "focal_length_px": round(focal_px, 2),
                    }
            payload.append(item)
        print(json.dumps({"detections": payload}, separators=(",", ":")))


def import_camera_stack():
    try:
        import cv2
        from picamera2 import MappedArray, Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import (
            NetworkIntrinsics,
            postprocess_nanodet_detection,
        )
    except ImportError as error:
        print(
            "Missing Raspberry Pi AI Camera dependencies. Install them on the Pi with:\n"
            "  sudo apt update\n"
            "  sudo apt install imx500-all python3-opencv python3-munkres\n\n"
            f"Original import error: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return cv2, MappedArray, Picamera2, IMX500, NetworkIntrinsics, postprocess_nanodet_detection


def get_args(
    default_targets=None,
    default_distance_target=None,
    default_object_width_m=None,
):
    parser = argparse.ArgumentParser(
        description="Run Raspberry Pi AI Camera object detection and print JSON detections."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to an IMX500 .rpk model.")
    parser.add_argument("--labels", help="Optional labels text file, one label per line.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Minimum confidence.")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold.")
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--fps", type=int, help="Override camera frame rate.")
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument("--no-draw", action="store_true", help="Do not draw preview boxes.")
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.25,
        help="Seconds between terminal detection updates.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=list(default_targets or []),
        help="Only print matching labels. Can be repeated, for example --target person.",
    )
    parser.add_argument(
        "--distance-target",
        default=default_distance_target,
        help="Label whose bounding-box width should be used for distance estimation.",
    )
    parser.add_argument(
        "--object-width-m",
        type=float,
        default=default_object_width_m,
        help="Assumed real target width in metres.",
    )
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        default=AI_CAMERA_FOCAL_LENGTH_MM,
        help="Camera focal length in millimetres (default: 4.74).",
    )
    parser.add_argument(
        "--pixel-pitch-um",
        type=float,
        default=AI_CAMERA_PIXEL_PITCH_UM,
        help="Sensor pixel pitch in micrometres (default: 1.55).",
    )
    parser.add_argument(
        "--sensor-width-px",
        type=int,
        default=AI_CAMERA_SENSOR_WIDTH_PX,
        help="Native horizontal sensor resolution (default: 4056).",
    )
    parser.add_argument(
        "--bbox-normalization",
        action=argparse.BooleanOptionalAction,
        help="Override model bounding box normalization.",
    )
    parser.add_argument(
        "--bbox-order",
        choices=("yx", "xy"),
        help="Override bounding box order.",
    )
    parser.add_argument(
        "--postprocess",
        choices=("", "nanodet"),
        default=None,
        help="Override model post-processing type.",
    )
    parser.add_argument(
        "--preserve-aspect-ratio",
        action=argparse.BooleanOptionalAction,
        help="Preserve input tensor aspect ratio.",
    )
    args = parser.parse_args()
    if args.object_width_m is not None and args.object_width_m <= 0.0:
        parser.error("object-width-m must be positive")
    if args.distance_target and args.object_width_m is None:
        parser.error("distance-target requires object-width-m")
    if args.focal_length_mm <= 0.0:
        parser.error("focal-length-mm must be positive")
    if args.pixel_pitch_um <= 0.0:
        parser.error("pixel-pitch-um must be positive")
    if args.sensor_width_px <= 0:
        parser.error("sensor-width-px must be positive")
    return args


def rectangle_to_box(rectangle):
    if hasattr(rectangle, "x"):
        return (
            int(rectangle.x),
            int(rectangle.y),
            int(rectangle.width),
            int(rectangle.height),
        )
    x, y, w, h = rectangle
    return int(x), int(y), int(w), int(h)


def load_labels(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as label_file:
        return [line.strip() for line in label_file if line.strip()]


def main(
    default_targets=None,
    default_distance_target=None,
    default_object_width_m=None,
):
    args = get_args(
        default_targets,
        default_distance_target,
        default_object_width_m,
    )
    (
        cv2,
        MappedArray,
        Picamera2,
        IMX500,
        NetworkIntrinsics,
        postprocess_nanodet_detection,
    ) = import_camera_stack()

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
    frame_rate = args.fps or intrinsics.inference_rate
    config = picam2.create_preview_configuration(
        controls={"FrameRate": frame_rate},
        buffer_count=12,
    )
    printer = DetectionPrinter(
        args.print_interval,
        args.target,
        distance_target=args.distance_target,
        object_width_m=args.object_width_m,
        focal_length_mm=args.focal_length_mm,
        pixel_pitch_um=args.pixel_pitch_um,
        sensor_width_px=args.sensor_width_px,
    )
    detections = []

    @lru_cache
    def get_labels():
        model_labels = intrinsics.labels or []
        if intrinsics.ignore_dash_labels:
            model_labels = [label for label in model_labels if label and label != "-"]
        return model_labels

    def label_for(category):
        labels = get_labels()
        category = int(category)
        if 0 <= category < len(labels):
            return labels[category]
        return f"class_{category}"

    def parse_detections(metadata):
        nonlocal detections

        outputs = imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            return detections

        input_w, input_h = imx500.get_input_size()
        if intrinsics.postprocess == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=outputs[0],
                conf=args.threshold,
                iou_thres=args.iou,
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
            rect = imx500.convert_inference_coords(coords, metadata, picam2)
            x, y, w, h = rectangle_to_box(rect)
            parsed.append(
                Detection(
                    label=label_for(category),
                    category=int(category),
                    confidence=float(score),
                    box=(x, y, w, h),
                    center=(x + w / 2, y + h / 2),
                )
            )

        detections = parsed[: args.max_detections]
        return detections

    def draw_detections(request, stream="main"):
        if args.no_draw:
            return
        with MappedArray(request, stream) as mapped:
            for detection in detections:
                x, y, w, h = detection.box
                label = f"{detection.label} ({detection.confidence:.2f})"
                if (
                    args.distance_target
                    and detection.label.lower() == args.distance_target.lower()
                ):
                    frame_width = mapped.array.shape[1]
                    distance_m = estimate_distance_m(
                        args.object_width_m,
                        w,
                        frame_width,
                        args.focal_length_mm,
                        args.pixel_pitch_um,
                        args.sensor_width_px,
                    )
                    if distance_m is not None:
                        label += f" {w}px {distance_m:.2f}m"
                cv2.rectangle(mapped.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    mapped.array,
                    label,
                    (x + 4, max(18, y + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    1,
                )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=not args.headless)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    if not args.headless and not args.no_draw:
        picam2.pre_callback = draw_detections

    print("AI Camera object detection running. Press Ctrl+C to stop.")
    try:
        while True:
            metadata = picam2.capture_metadata()
            current_detections = parse_detections(metadata)
            size = picam2.camera_configuration()["main"]["size"]
            printer.print(current_detections, size)
    except KeyboardInterrupt:
        print("\nStopping object detection.")
    finally:
        picam2.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
