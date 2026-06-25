import time
from adafruit_servokit import ServoKit

kits = {
    "0x40": ServoKit(channels=16, address=0x40),
    "0x41": ServoKit(channels=16, address=0x41),
}

# Channel layout per leg:
# channel 0 = foot
# channel 1 = knee
# channel 2 = hip/body joint
#
# IMPORTANT:
# This program NEVER commands the hip/body joint.
# It only creates servo objects for foot and knee.

LEG_CHANNELS = {
    "leg1": {"driver": "0x40", "foot": 0, "knee": 1},
    "leg2": {"driver": "0x40", "foot": 4, "knee": 5},
    "leg3": {"driver": "0x40", "foot": 8, "knee": 9},
    "leg4": {"driver": "0x41", "foot": 0, "knee": 1},
    "leg5": {"driver": "0x41", "foot": 4, "knee": 5},
    "leg6": {"driver": "0x41", "foot": 8, "knee": 9},
}

NEUTRALS = {
    "leg1": {"foot": 90, "knee": 90},
    "leg2": {"foot": 90, "knee": 90},
    "leg3": {"foot": 90, "knee": 90},
    "leg4": {"foot": 90, "knee": 90},
    "leg5": {"foot": 90, "knee": 90},
    "leg6": {"foot": 90, "knee": 90},
}

# Change to -1 if a joint moves the wrong direction.
DIRECTIONS = {
    "leg1": {"foot": 1, "knee": 1},
    "leg2": {"foot": 1, "knee": 1},
    "leg3": {"foot": 1, "knee": 1},
    "leg4": {"foot": 1, "knee": 1},
    "leg5": {"foot": 1, "knee": 1},
    "leg6": {"foot": 1, "knee": 1},
}

# Motion tuning values.
# These are offsets from neutral.
KNEE_UP = -25
FOOT_IN = -60
KNEE_DOWN_PUSH = 25

STEP_DELAY = 0.04

legs = {}

for leg_name, channels in LEG_CHANNELS.items():
    legs[leg_name] = {}
    driver = kits[channels["driver"]]

    for joint_name in ("foot", "knee"):
        servo = driver.servo[channels[joint_name]]
        servo.actuation_range = 180
        servo.set_pulse_width_range(700, 2300)
        legs[leg_name][joint_name] = servo


def clamp_angle(angle):
    return max(0, min(180, angle))


def apply_offset(leg_name, joint_name, offset):
    neutral = NEUTRALS[leg_name][joint_name]
    direction = DIRECTIONS[leg_name][joint_name]
    return clamp_angle(neutral + direction * offset)


def set_leg_offsets(leg_name, foot_offset, knee_offset):
    foot_angle = apply_offset(leg_name, "foot", foot_offset)
    knee_angle = apply_offset(leg_name, "knee", knee_offset)

    legs[leg_name]["foot"].angle = foot_angle
    legs[leg_name]["knee"].angle = knee_angle


def set_all_legs_offsets(foot_offset, knee_offset, delay=0.0):
    print(f"Offsets → foot={foot_offset:.1f}, knee={knee_offset:.1f}")

    for leg_name in legs:
        set_leg_offsets(leg_name, foot_offset, knee_offset)

    if delay > 0:
        time.sleep(delay)


def interpolate_pose(start_pose, end_pose, steps=60):
    # Pose format: [foot_offset, knee_offset]
    for step in range(steps + 1):
        t = step / steps

        foot = start_pose[0] + (end_pose[0] - start_pose[0]) * t
        knee = start_pose[1] + (end_pose[1] - start_pose[1]) * t

        set_all_legs_offsets(foot, knee)
        time.sleep(STEP_DELAY)


def release_all():
    print("Releasing foot and knee servos only.")
    for leg_name in legs:
        legs[leg_name]["foot"].angle = None
        legs[leg_name]["knee"].angle = None


try:
    print("Starting stand-up sequence.")
    print("Only foot and knee joints will move.")
    print("Hip/body joints are never commanded by this program.")
    print("Be ready to unplug power if anything binds or tips.")

    neutral_pose = [0, 0]

    print("Step 1: Set foot and knee to neutral")
    set_all_legs_offsets(0, 0, delay=2)

    print("Step 2: Bend knees up")
    knee_up_pose = [0, KNEE_UP]
    interpolate_pose(neutral_pose, knee_up_pose, steps=80)
    time.sleep(1)

    print("Step 3: Bend feet in")
    foot_in_pose = [FOOT_IN, KNEE_UP]
    interpolate_pose(knee_up_pose, foot_in_pose, steps=80)
    time.sleep(1)

    print("Step 4: Bend knees down to push up")
    standing_pose = [FOOT_IN, KNEE_DOWN_PUSH]
    interpolate_pose(foot_in_pose, standing_pose, steps=120)

    print("Standing pose reached. Holding foot and knee positions.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    release_all()
