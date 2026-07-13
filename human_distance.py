"""Detect humans with the AI Camera and estimate distance from box width."""

from ai_camera_object_detection import DEFAULT_HUMAN_WIDTH_M, main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_targets=["person"],
            default_distance_target="person",
            default_object_width_m=DEFAULT_HUMAN_WIDTH_M,
        )
    )
