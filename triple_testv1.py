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

# Joint order inside each leg:
# [hip, knee, ankle]
JOINT_NAMES = ["hip", "knee", "ankle"]

# Conservative center positions.
# You can adjust these if your servo horns are mounted at offsets.
NEUTRALS = {
    "leg1": [90, 90, 90],
    "leg2": [90, 90, 90],
    "leg3": [90, 90, 90],
}

# Direction multipliers.
# Use -1 if a servo moves opposite of what you want.
DIRECTIONS = {
    "leg1": [1, 1, 1],
    "leg2": [1, 1, 1],
    "leg3": [1, 1, 1],
}

# Create servo objects
legs = {}

for leg_name, channels in LEG_CHANNELS.items():
    legs[leg_name] = []

    for channel in channels:
        servo = kit.servo[channel]
        servo.actuation_range = 180

        # Conservative pulse range for MG996R-style servos.
        # If you need more range later, try 600–2400.
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


def set_all_legs_offsets(hip_offset, knee_offset, ankle_offset, delay=1.0):
    print(
        f"All legs → hip offset {hip_offset}, "
        f"knee offset {knee_offset}, ankle offset {ankle_offset}"
    )

    for leg_name in legs:
        set_leg_offsets(leg_name, hip_offset, knee_offset, ankle_offset)

    time.sleep(delay)


def set_all_legs_absolute(hip_angle, knee_angle, ankle_angle, delay=1.0):
    print(f"All legs absolute → hip {hip_angle}, knee {knee_angle}, ankle {ankle_angle}")

    for leg_name in legs:
        legs[leg_name][0].angle = clamp_angle(hip_angle)
        legs[leg_name][1].angle = clamp_angle(knee_angle)
        legs[leg_name][2].angle = clamp_angle(ankle_angle)

    time.sleep(delay)


def center_all(delay=2.0):
    print("Centering all legs...")

    for leg_name in legs:
        for i in range(3):
            legs[leg_name][i].angle = NEUTRALS[leg_name][i]

    time.sleep(delay)


def release_all():
    print("Releasing all servos...")

    for leg_name in legs:
        for servo in legs[leg_name]:
            servo.angle = None


try:
    center_all(3)

    print("Small synchronized movement test...")

    # Very small test first
    set_all_legs_offsets(0, 0, 0, 1)
    set_all_legs_offsets(5, 0, 0, 1)
    set_all_legs_offsets(-5, 0, 0, 1)
    set_all_legs_offsets(0, 5, 0, 1)
    set_all_legs_offsets(0, -5, 0, 1)
    set_all_legs_offsets(0, 0, 5, 1)
    set_all_legs_offsets(0, 0, -5, 1)
    set_all_legs_offsets(0, 0, 0, 1)

    print("Larger synchronized movement test...")

    # Slightly larger movement
    set_all_legs_offsets(10, 0, 0, 1)
    set_all_legs_offsets(-10, 0, 0, 1)
    set_all_legs_offsets(0, 10, 0, 1)
    set_all_legs_offsets(0, -10, 0, 1)
    set_all_legs_offsets(0, 0, 10, 1)
    set_all_legs_offsets(0, 0, -10, 1)
    set_all_legs_offsets(0, 0, 0, 1)

    print("Simple pose test...")

    # Basic coordinated leg poses
    set_all_legs_offsets(10, -10, 10, 1)
    set_all_legs_offsets(-10, 10, -10, 1)
    set_all_legs_offsets(0, 0, 0, 1)

    print("Test complete.")

finally:
    release_all()

