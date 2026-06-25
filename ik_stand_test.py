import math
import time
from adafruit_servokit import ServoKit


# Hexapod IK stand test.
#
# Per-leg channel order:
#   foot/ankle = base channel + 0
#   knee       = base channel + 1
#   hip/body   = base channel + 2
#
# This file uses two PCA9685 boards:
#   left_kit  at I2C address 0x40
#   right_kit at I2C address 0x41
#
# Before running with the robot on the floor, test with legs lifted or power
# ready to disconnect. Servo directions almost always need per-leg adjustment.


LEFT_KIT_ADDRESS = 0x40
RIGHT_KIT_ADDRESS = 0x41

left_kit = ServoKit(channels=16, address=LEFT_KIT_ADDRESS)
right_kit = ServoKit(channels=16, address=RIGHT_KIT_ADDRESS)
KITS = [left_kit, right_kit]


# Link lengths in inches.
# UPPER_LEG is the shorter link.
# LOWER_LEG is the longer link.
UPPER_LEG = 3.254817
LOWER_LEG = 5.118565

# If your hip/body servo rotates a short horizontal link before the knee link
# starts, put that link length here. Leave 0.0 if the knee plane starts at the
# hip/body servo axis.
HIP_LINK = 0.0


# Current assumed wiring:
#   board 0: three legs, channels 0-2, 4-6, 8-10
#   board 1: three legs, channels 0-2, 4-6, 8-10
#
# Each leg has foot, knee, hip in that order.
LEG_CHANNELS = {
    "front_left": {"kit": 0, "foot": 0, "knee": 1, "hip": 2},
    "middle_left": {"kit": 0, "foot": 4, "knee": 5, "hip": 6},
    "rear_left": {"kit": 0, "foot": 8, "knee": 9, "hip": 10},
    "front_right": {"kit": 1, "foot": 0, "knee": 1, "hip": 2},
    "middle_right": {"kit": 1, "foot": 4, "knee": 5, "hip": 6},
    "rear_right": {"kit": 1, "foot": 8, "knee": 9, "hip": 10},
}


# Servo center angles. Adjust these after calibration.
NEUTRALS = {
    leg_name: {"foot": 90, "knee": 90, "hip": 90}
    for leg_name in LEG_CHANNELS
}


# Change a direction to -1 if that joint moves backward.
# Left/right sides often need opposite signs.
DIRECTIONS = {
    "front_left": {"foot": 1, "knee": 1, "hip": 1},
    "middle_left": {"foot": 1, "knee": 1, "hip": 1},
    "rear_left": {"foot": 1, "knee": 1, "hip": 1},
    "front_right": {"foot": 1, "knee": 1, "hip": 1},
    "middle_right": {"foot": 1, "knee": 1, "hip": 1},
    "rear_right": {"foot": 1, "knee": 1, "hip": 1},
}


# Per-joint trims in degrees. Use these to fix small mechanical offsets after
# the directions and neutrals are correct.
TRIMS = {
    leg_name: {"foot": 0, "knee": 0, "hip": 0}
    for leg_name in LEG_CHANNELS
}


# Conservative limits. Narrow these if a joint binds.
MIN_ANGLE = 0
MAX_ANGLE = 180

STEP_DELAY = 0.04


# Foot targets are in inches, relative to each leg's hip/body joint.
# x = forward/back, y = away from body centerline, z = down/up
#
# More negative z means a lower foot. If the robot tries to crouch instead of
# stand, reverse the sign convention in leg_ik or change the target z values.
CROUCH_FEET = {
    "front_left": (3.5, 3.0, -2.5),
    "middle_left": (0.0, 3.5, -2.5),
    "rear_left": (-3.5, 3.0, -2.5),
    "front_right": (3.5, -3.0, -2.5),
    "middle_right": (0.0, -3.5, -2.5),
    "rear_right": (-3.5, -3.0, -2.5),
}

STAND_FEET = {
    "front_left": (4.0, 3.5, -5.0),
    "middle_left": (0.0, 4.0, -5.0),
    "rear_left": (-4.0, 3.5, -5.0),
    "front_right": (4.0, -3.5, -5.0),
    "middle_right": (0.0, -4.0, -5.0),
    "rear_right": (-4.0, -3.5, -5.0),
}


legs = {}


def clamp(value, low, high):
    return max(low, min(high, value))


def clamp_angle(angle):
    return clamp(angle, MIN_ANGLE, MAX_ANGLE)


def setup_servos():
    for leg_name, channels in LEG_CHANNELS.items():
        kit = KITS[channels["kit"]]
        legs[leg_name] = {}

        for joint_name in ("foot", "knee", "hip"):
            servo = kit.servo[channels[joint_name]]
            servo.actuation_range = 180
            servo.set_pulse_width_range(700, 2300)
            legs[leg_name][joint_name] = servo


def servo_angle(leg_name, joint_name, ik_angle):
    neutral = NEUTRALS[leg_name][joint_name]
    direction = DIRECTIONS[leg_name][joint_name]
    trim = TRIMS[leg_name][joint_name]
    return clamp_angle(neutral + direction * ik_angle + trim)


def leg_ik(x, y, z):
    """Return hip, knee, foot IK offsets in degrees for one leg."""
    hip_angle = math.degrees(math.atan2(y, x))

    horizontal = math.sqrt(x * x + y * y) - HIP_LINK
    distance = math.sqrt(horizontal * horizontal + z * z)

    min_reach = abs(UPPER_LEG - LOWER_LEG) + 0.001
    max_reach = UPPER_LEG + LOWER_LEG - 0.001
    distance = clamp(distance, min_reach, max_reach)

    knee_cos = (
        (UPPER_LEG * UPPER_LEG + LOWER_LEG * LOWER_LEG - distance * distance)
        / (2 * UPPER_LEG * LOWER_LEG)
    )
    knee_inner = math.acos(clamp(knee_cos, -1, 1))

    foot_to_hip_angle = math.atan2(z, horizontal)
    upper_cos = (
        (UPPER_LEG * UPPER_LEG + distance * distance - LOWER_LEG * LOWER_LEG)
        / (2 * UPPER_LEG * distance)
    )
    upper_offset = math.acos(clamp(upper_cos, -1, 1))

    knee_angle = math.degrees(foot_to_hip_angle + upper_offset)
    foot_angle = math.degrees(math.pi - knee_inner)

    return hip_angle, knee_angle, foot_angle


def set_leg_target(leg_name, target):
    x, y, z = target
    hip_angle, knee_angle, foot_angle = leg_ik(x, y, z)

    legs[leg_name]["hip"].angle = servo_angle(leg_name, "hip", hip_angle)
    legs[leg_name]["knee"].angle = servo_angle(leg_name, "knee", knee_angle)
    legs[leg_name]["foot"].angle = servo_angle(leg_name, "foot", foot_angle)


def set_all_targets(targets):
    for leg_name, target in targets.items():
        set_leg_target(leg_name, target)


def interpolate_targets(start_targets, end_targets, steps=80):
    for step in range(steps + 1):
        t = step / steps
        current_targets = {}

        for leg_name in start_targets:
            sx, sy, sz = start_targets[leg_name]
            ex, ey, ez = end_targets[leg_name]
            current_targets[leg_name] = (
                sx + (ex - sx) * t,
                sy + (ey - sy) * t,
                sz + (ez - sz) * t,
            )

        set_all_targets(current_targets)
        time.sleep(STEP_DELAY)


def release_all():
    print("Releasing all servos.")
    for leg in legs.values():
        for servo in leg.values():
            servo.angle = None


def print_targets(name, targets):
    print(name)
    for leg_name, target in targets.items():
        hip_angle, knee_angle, foot_angle = leg_ik(*target)
        print(
            f"  {leg_name}: target={target}, "
            f"hip={hip_angle:.1f}, knee={knee_angle:.1f}, foot={foot_angle:.1f}"
        )


try:
    setup_servos()

    print("Starting IK stand test.")
    print("Using two PCA9685 boards and all 18 servos.")
    print("Channel order per leg: foot=0, knee=1, hip=2.")
    print("Be ready to unplug power if anything binds or tips.")

    print_targets("Crouch IK offsets:", CROUCH_FEET)
    print_targets("Stand IK offsets:", STAND_FEET)

    print("Step 1: Move to crouch pose.")
    set_all_targets(CROUCH_FEET)
    time.sleep(2)

    print("Step 2: Raise body into standing pose.")
    interpolate_targets(CROUCH_FEET, STAND_FEET, steps=120)

    print("Standing pose reached. Holding position.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    release_all()
