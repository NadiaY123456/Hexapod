import argparse
import math
import os
import struct
import time


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

AXIS_MIN = -32767
AXIS_MAX = 32767
STICK_DEADZONE = 0.18
DPAD_THRESHOLD = 0.50

# Common Linux joystick axis mappings for Xbox-style controllers.
LEFT_STICK_AXES = (0, 1)
RIGHT_STICK_AXES = (3, 4)
DPAD_AXES = (6, 7)


def normalize_axis(value):
    if value < 0:
        return max(-1.0, value / abs(AXIS_MIN))
    return min(1.0, value / AXIS_MAX)


def stick_angle_degrees(x, y):
    # Linux joystick y is usually negative when pushed up.
    angle = math.degrees(math.atan2(x, -y))
    if angle > 180:
        return angle - 360
    if angle <= -180:
        return angle + 360
    return angle


def stick_state(axis_values, axes):
    x = axis_values.get(axes[0], 0.0)
    y = axis_values.get(axes[1], 0.0)
    magnitude = min(1.0, math.hypot(x, y))

    if magnitude < STICK_DEADZONE:
        return "center"

    angle = stick_angle_degrees(x, y)
    return f"{angle:6.1f} degrees, strength {magnitude:.2f}"


def dpad_state(axis_values):
    x = axis_values.get(DPAD_AXES[0], 0.0)
    y = axis_values.get(DPAD_AXES[1], 0.0)

    if y <= -DPAD_THRESHOLD:
        return "up"
    if y >= DPAD_THRESHOLD:
        return "down"
    if x <= -DPAD_THRESHOLD:
        return "left"
    if x >= DPAD_THRESHOLD:
        return "right"

    return "center"


def print_if_changed(label, value, previous_values):
    if previous_values.get(label) == value:
        return

    previous_values[label] = value
    print(f"{label}: {value}", flush=True)


def open_controller(device_path):
    while True:
        try:
            return open(device_path, "rb")
        except FileNotFoundError:
            print(f"Waiting for controller at {device_path}...")
            time.sleep(1)


def read_controller(device_path):
    axis_values = {}
    previous_values = {}

    print(f"Reading controller events from {device_path}")
    print("Move the D-pad or joysticks. Press Ctrl+C to quit.")
    print(
        "Expected mapping: left stick axes 0/1, right stick axes 3/4, "
        "D-pad axes 6/7."
    )

    with open_controller(device_path) as controller:
        while True:
            event = controller.read(8)
            if len(event) != 8:
                continue

            _, raw_value, event_type, number = struct.unpack("IhBB", event)
            event_type_without_init = event_type & ~JS_EVENT_INIT

            if event_type_without_init == JS_EVENT_AXIS:
                axis_values[number] = normalize_axis(raw_value)

                if number in LEFT_STICK_AXES:
                    print_if_changed(
                        "left joystick",
                        stick_state(axis_values, LEFT_STICK_AXES),
                        previous_values,
                    )
                elif number in RIGHT_STICK_AXES:
                    print_if_changed(
                        "right joystick",
                        stick_state(axis_values, RIGHT_STICK_AXES),
                        previous_values,
                    )
                elif number in DPAD_AXES:
                    print_if_changed("D-pad", dpad_state(axis_values), previous_values)
                else:
                    print_if_changed(
                        f"axis {number}",
                        f"{axis_values[number]:.2f}",
                        previous_values,
                    )

            elif event_type_without_init == JS_EVENT_BUTTON:
                state = "pressed" if raw_value else "released"
                print_if_changed(f"button {number}", state, previous_values)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print MSI GC30 controller D-pad and joystick input."
    )
    parser.add_argument(
        "device",
        nargs="?",
        default=os.environ.get("CONTROLLER_DEVICE", "/dev/input/js0"),
        help="Linux joystick device path. Default: /dev/input/js0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        read_controller(args.device)
    except KeyboardInterrupt:
        print("\nController test stopped.")
