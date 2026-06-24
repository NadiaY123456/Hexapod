import time

from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

LEG_CHANNELS = {
    "leg1": [0, 1, 2],
    "leg2": [4, 5, 6],
    "leg3": [8, 9, 10],
}

JOINT_NAMES = ["ankle", "knee", "hip"]

NEUTRALS = {
    "leg1": [90, 90, 90],
    "leg2": [90, 90, 90],
    "leg3": [90, 90, 90],
}

# Change signs if a joint moves the wrong way.
DIRECTIONS = {
    "leg1": [1, 1, 1],
    "leg2": [1, 1, 1],
    "leg3": [1, 1, 1],
}

# Tune these carefully.
# These are offsets from neutral, not absolute angles.
LEGS_OUT_POSE = {
    "hip": 0,
    "knee": -25,
    "ankle": -35,
}

OUTER_BENT_POSE = {
    "hip": 0,
    "knee": -25,
    "ankle": 10,
}

STANDING_POSE = {
    "hip": 0,
    "knee": 20,
    "ankle": 15,
}

STEP_DELAY = 0.04
HOLD_DELAY = 2.0

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


def set_all_legs_pose(pose):
    for leg_name in legs:
        set_leg_offsets(
            leg_name,
            pose["hip"],
            pose["knee"],
            pose["ankle"],
        )


def interpolate_pose(start_pose, end_pose, steps=50):
    for step in range(steps + 1):
        t = step / steps

        pose = {
            "hip": start_pose["hip"] + (end_pose["hip"] - start_pose["hip"]) * t,
            "knee": start_pose["knee"] + (end_pose["knee"] - start_pose["knee"]) * t,
            "ankle": start_pose["ankle"] + (end_pose["ankle"] - start_pose["ankle"]) * t,
        }

        print(
            f"hip={pose['hip']:.1f}, "
            f"knee={pose['knee']:.1f}, "
            f"ankle={pose['ankle']:.1f}"
        )

        set_all_legs_pose(pose)
        time.sleep(STEP_DELAY)


def release_all():
    print("Releasing all servos...")
    for leg_name in legs:
        for servo in legs[leg_name]:
            servo.angle = None


try:
    print("Starting stand-up sequence.")
    print("Be ready to unplug power if anything binds or the robot tips.")

    print("Step 1: Move to legs-out starting pose.")
    set_all_legs_pose(LEGS_OUT_POSE)
    time.sleep(HOLD_DELAY)

    print("Step 2: Bend outer joints first.")
    interpolate_pose(
        LEGS_OUT_POSE,
        OUTER_BENT_POSE,
        steps=60,
    )
    time.sleep(HOLD_DELAY)

    print("Step 3: Bend inner joints to push up.")
    interpolate_pose(
        OUTER_BENT_POSE,
        STANDING_POSE,
        steps=80,
    )
    time.sleep(HOLD_DELAY)

    print("Standing pose reached. Holding position.")

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    # Comment this out if you want the robot to keep holding itself up
    # after Ctrl+C.
    release_all()