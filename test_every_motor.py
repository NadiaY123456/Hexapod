import argparse
import sys
import termios
import tty
from contextlib import contextmanager

from adafruit_servokit import ServoKit


kits = {
    "0x40": ServoKit(channels=16, address=0x40),
    "0x41": ServoKit(channels=16, address=0x41),
}

# Copied from controller_walk.py because that file has the trusted numbering.
# The controller code calls the ankle-style joint "foot".
LEG_CHANNELS = {
    "leg1": {"driver": "0x40", "foot": 0, "knee": 1, "hip": 2},
    "leg2": {"driver": "0x40", "foot": 4, "knee": 5, "hip": 6},
    "leg3": {"driver": "0x40", "foot": 8, "knee": 9, "hip": 10},
    "leg4": {"driver": "0x41", "foot": 0, "knee": 1, "hip": 2},
    "leg5": {"driver": "0x41", "foot": 4, "knee": 5, "hip": 6},
    "leg6": {"driver": "0x41", "foot": 8, "knee": 9, "hip": 10},
}

JOINT_DISPLAY_NAMES = {
    "hip": "hip",
    "knee": "knee",
    "foot": "ankle",
}

JOINT_ORDER = ("hip", "knee", "foot")
MIN_ANGLE = 0
MAX_ANGLE = 180


def clamp_angle(angle):
    return max(MIN_ANGLE, min(MAX_ANGLE, angle))


def build_motor_map():
    motors = {}
    motor_number = 1

    for leg_name in sorted(LEG_CHANNELS):
        channels = LEG_CHANNELS[leg_name]

        for joint_name in JOINT_ORDER:
            motors[motor_number] = {
                "leg": leg_name,
                "joint": joint_name,
                "display_joint": JOINT_DISPLAY_NAMES[joint_name],
                "driver": channels["driver"],
                "channel": channels[joint_name],
            }
            motor_number += 1

    return motors


def configure_servo(motor):
    driver = kits[motor["driver"]]
    servo = driver.servo[motor["channel"]]
    servo.actuation_range = 180
    servo.set_pulse_width_range(700, 2300)
    return servo


def print_motor_map(motors):
    print("\nMotor numbering:")
    print("  motor | leg  | joint | PCA9685 | port/channel")
    print("  ------+------+-------+---------+-------------")

    for number, motor in motors.items():
        print(
            f"  {number:>5} | {motor['leg']:<4} | "
            f"{motor['display_joint']:<5} | {motor['driver']:<7} | "
            f"{motor['channel']}"
        )

    print()


def prompt_motor_number(motors):
    while True:
        value = input("Enter motor number, m to show map, or q to quit: ").strip().lower()

        if value == "q":
            return None
        if value == "m":
            print_motor_map(motors)
            continue

        try:
            motor_number = int(value)
        except ValueError:
            print("Please enter a motor number, m, or q.")
            continue

        if motor_number in motors:
            return motor_number

        print(f"Motor must be from 1 to {max(motors)}.")


@contextmanager
def raw_terminal():
    settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def read_key():
    return sys.stdin.read(1).lower()


def test_motor(motor_number, motor, start_angle):
    servo = configure_servo(motor)
    angle = start_angle

    print(
        f"\nTesting motor {motor_number}: {motor['leg']} "
        f"{motor['display_joint']} "
        f"(PCA9685 {motor['driver']}, channel {motor['channel']})"
    )
    print(f"Starting tracked angle is {angle} degrees.")
    print("Keys: g = +1 degree, h = -1 degree, n = choose a new motor, q = quit")

    with raw_terminal():
        while True:
            key = read_key()

            if key == "g":
                angle = clamp_angle(angle + 1)
                servo.angle = angle
                print(f"\rMotor {motor_number} -> {angle:3} degrees", end="", flush=True)
            elif key == "h":
                angle = clamp_angle(angle - 1)
                servo.angle = angle
                print(f"\rMotor {motor_number} -> {angle:3} degrees", end="", flush=True)
            elif key == "n":
                print()
                servo.angle = None
                return True
            elif key == "q":
                print()
                servo.angle = None
                return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively identify and nudge every hexapod motor."
    )
    parser.add_argument(
        "--start-angle",
        type=int,
        default=90,
        help="Tracked starting angle used before g/h nudges. Default: 90",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_angle = clamp_angle(args.start_angle)
    motors = build_motor_map()

    print_motor_map(motors)

    while True:
        motor_number = prompt_motor_number(motors)

        if motor_number is None:
            break

        choose_another = test_motor(motor_number, motors[motor_number], start_angle)

        if not choose_another:
            break

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
