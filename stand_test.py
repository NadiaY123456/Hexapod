import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

# Channel layout:
# Leg 1: channels 0, 1, 2
# Leg 2: channels 4, 5, 6
# Leg 3: channels 8, 9, 10
LEG_CHANNELS = {
    "leg1": [0, 1, 2],
    "leg2": [4, 5, 6],
    "leg3": [8, 9, 10],
}

# Joint order: [hip, knee, ankle]
JOINT_NAMES = ["ankle", "knee", "hip"]

# Adjust these if your mechanical neutral is not exactly 90.
NEUTRALS = {
    "leg1": [90, 90, 90],
    "leg2": [90, 90, 90],
    "leg3": [90, 90, 90],
}

# Change a value to -1 if that joint moves opposite of expected.
DIRECTIONS = {
    "leg1": [1, 1, 1],
    "leg2": [1, 1, 1],
    "leg3": [1, 1, 1],
}

# Tune these carefully.
# These are offsets from neutral, not absolute angles.
LEGS_OUT_POSE = {
    "ankle": 0,
    "knee": -25,
    "hip": -35,
}

OUTER_BENT_POSE = {
    "ankle": 0,
    "knee": -25,
    "hip": 10,
}

STANDING_POSE = {
    "ankle": 0,
    "knee": 20,
    "hip": 15,
}

STEP_DELAY = 0.04

legs = {}

for leg_name, channels in LEG_CHANNELS.items():
    legs[leg_name] = []

    for channel in channels:
        servo = kit.servo[channel]
        servo.actuation_range = 180
        servo.set_pulse_width_range(700, 2300)
        legs[leg_name].append(servo)


def clamp_angle(angle):
    return max(0, min(180, angle))


def apply_offset(leg_name, joint_index, offset):
    neutral = NEUTRALS[leg_name][joint_index]
    direction = DIRECTIONS[leg_name][joint_index]
    return clamp_angle(neutral + direction * offset)


def set_leg_offsets(leg_name, hip_offset, knee_offset, ankle_offset):
    offsets = [hip_offset, knee_offset, ankle_offset]

    for i, offset in enumerate(offsets):
        angle = apply_offset(leg_name, i, offset)
        legs[leg_name][i].angle = angle


def set_all_legs_offsets(hip_offset, knee_offset, ankle_offset, delay=0.0):
    print(
        f"Offsets → hip={hip_offset:.1f}, "
        f"knee={knee_offset:.1f}, ankle={ankle_offset:.1f}"
    )

    for leg_name in legs:
        set_leg_offsets(leg_name, hip_offset, knee_offset, ankle_offset)

    if delay > 0:
        time.sleep(delay)


def interpolate_pose(start_pose, end_pose, steps=60):
    for step in range(steps + 1):
        t = step / steps

        hip = start_pose[0] + (end_pose[0] - start_pose[0]) * t
        knee = start_pose[1] + (end_pose[1] - start_pose[1]) * t
        ankle = start_pose[2] + (end_pose[2] - start_pose[2]) * t

        set_all_legs_offsets(hip, knee, ankle)
        time.sleep(STEP_DELAY)


def release_all():
    print("Releasing all servos...")

    for leg_name in legs:
        for servo in legs[leg_name]:
            servo.angle = None


try:
    print("Starting standing sequence.")
    print("Be ready to unplug power if anything binds or tips.")

    neutral_pose = [0, 0, 0]

    print("Step 1: Neutral")
    set_all_legs_offsets(0, 0, 0, delay=2)

    print("Step 2: Bend knees up")
    knee_up_pose = [0, KNEE_UP, 0]
    interpolate_pose(neutral_pose, knee_up_pose, steps=80)
    time.sleep(1)

    print("Step 3: Bend ankles in")
    ankle_in_pose = [0, KNEE_UP, ANKLE_IN]
    interpolate_pose(knee_up_pose, ankle_in_pose, steps=80)
    time.sleep(1)

    print("Step 4: Push up by bending knees down")
    standing_pose = [0, KNEE_DOWN_PUSH, ANKLE_IN]
    interpolate_pose(ankle_in_pose, standing_pose, steps=120)

    print("Standing pose reached. Holding.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    release_all()