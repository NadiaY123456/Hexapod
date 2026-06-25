import math
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
# This program sets each hip/body joint to 90 once at startup.
# After that, the IK loop only commands foot and knee.

LEG_CHANNELS = {
    "leg1": {"driver": "0x40", "foot": 0, "knee": 1, "hip": 2},
    "leg2": {"driver": "0x40", "foot": 4, "knee": 5, "hip": 6},
    "leg3": {"driver": "0x40", "foot": 8, "knee": 9, "hip": 10},
    "leg4": {"driver": "0x41", "foot": 0, "knee": 1, "hip": 2},
    "leg5": {"driver": "0x41", "foot": 4, "knee": 5, "hip": 6},
    "leg6": {"driver": "0x41", "foot": 8, "knee": 9, "hip": 10},
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

STEP_DELAY = 0.04
CONTACT_SETTLE_DELAY = 1.0
STAND_STEPS = 120
HIP_START_ANGLE = 90

# IK constants measured from leg assembly.step in millimeters.
# The 90-degree servo pose is treated as the starting foot-contact pose.
UPPER_LEG_LENGTH = 65.58  # knee joint to foot joint, projected in the side plane
LOWER_LEG_LENGTH = 78.42  # foot joint to the bottom contact point
BODY_LIFT_AFTER_CONTACT = 15.0
MEASURED_FLAT_FOOT_ANGLE = 21.5

# Start slightly tucked so the feet are not as far out before the lift begins.
# More negative foot offset pulls the feet inward. More negative knee offset
# starts with the knees more bent.
CONTACT_KNEE_OFFSET = -30.0
CONTACT_FOOT_OFFSET = MEASURED_FLAT_FOOT_ANGLE - NEUTRALS["leg1"]["foot"]

# Real-world correction: move foot and knee together during the lift. The knee
# correction is positive because the physical robot needs the opposite knee
# direction from the earlier inward-bend tests in order to stand.
FOOT_DOWN_EXTRA_DURING_LIFT = -12.0
KNEE_STAND_EXTRA_DURING_LIFT = 20.0

# Model angles for the 90-degree starting pose.
# 0 degrees means the segment points straight down in the side-plane model.
KNEE_MODEL_ZERO_DEG = -49.6
FOOT_MODEL_ZERO_DEG = 79.2
KNEE_MODEL_DIRECTION = 1
FOOT_MODEL_DIRECTION = 1

legs = {}
hips = {}

for leg_name, channels in LEG_CHANNELS.items():
    legs[leg_name] = {}
    driver = kits[channels["driver"]]

    for joint_name in ("foot", "knee"):
        servo = driver.servo[channels[joint_name]]
        servo.actuation_range = 180
        servo.set_pulse_width_range(700, 2300)
        legs[leg_name][joint_name] = servo

    hip_servo = driver.servo[channels["hip"]]
    hip_servo.actuation_range = 180
    hip_servo.set_pulse_width_range(700, 2300)
    hips[leg_name] = hip_servo


def validate_ik_constants():
    missing = []

    constants = {
        "UPPER_LEG_LENGTH": UPPER_LEG_LENGTH,
        "LOWER_LEG_LENGTH": LOWER_LEG_LENGTH,
        "BODY_LIFT_AFTER_CONTACT": BODY_LIFT_AFTER_CONTACT,
        "KNEE_MODEL_ZERO_DEG": KNEE_MODEL_ZERO_DEG,
        "FOOT_MODEL_ZERO_DEG": FOOT_MODEL_ZERO_DEG,
    }

    for name, value in constants.items():
        if value is None:
            missing.append(name)

    if missing:
        raise ValueError(
            "Fill in these IK constants before running stand_test_ik.py: "
            + ", ".join(missing)
        )


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
    print(f"Offsets -> foot={foot_offset:.1f}, knee={knee_offset:.1f}")

    for leg_name in legs:
        set_leg_offsets(leg_name, foot_offset, knee_offset)

    if delay > 0:
        time.sleep(delay)


def set_all_hips_to_start_angle(delay=1.0):
    print(f"Setting all hip/body joints to {HIP_START_ANGLE} degrees once.")

    for hip_servo in hips.values():
        hip_servo.angle = HIP_START_ANGLE

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


def offsets_to_model_angles(foot_offset, knee_offset):
    knee_angle = KNEE_MODEL_ZERO_DEG + KNEE_MODEL_DIRECTION * knee_offset
    foot_angle = FOOT_MODEL_ZERO_DEG + FOOT_MODEL_DIRECTION * foot_offset
    return math.radians(knee_angle), math.radians(foot_angle)


def model_angles_to_offsets(knee_angle, foot_angle):
    knee_offset = (math.degrees(knee_angle) - KNEE_MODEL_ZERO_DEG) / KNEE_MODEL_DIRECTION
    foot_offset = (
        math.degrees(foot_angle) - FOOT_MODEL_ZERO_DEG
    ) / FOOT_MODEL_DIRECTION
    return [foot_offset, knee_offset]


def foot_position_from_offsets(foot_offset, knee_offset):
    knee_angle, foot_angle = offsets_to_model_angles(foot_offset, knee_offset)

    x = (
        UPPER_LEG_LENGTH * math.sin(knee_angle)
        + LOWER_LEG_LENGTH * math.sin(knee_angle + foot_angle)
    )
    z = (
        UPPER_LEG_LENGTH * math.cos(knee_angle)
        + LOWER_LEG_LENGTH * math.cos(knee_angle + foot_angle)
    )

    return x, z


def clamp_reachable_target(x, z):
    max_reach = UPPER_LEG_LENGTH + LOWER_LEG_LENGTH - 0.001
    min_reach = abs(UPPER_LEG_LENGTH - LOWER_LEG_LENGTH) + 0.001
    distance = math.hypot(x, z)

    if distance == 0:
        return 0.0, min_reach

    if distance > max_reach:
        scale = max_reach / distance
        return x * scale, z * scale

    if distance < min_reach:
        scale = min_reach / distance
        return x * scale, z * scale

    return x, z


def solve_leg_ik(x, z, previous_pose):
    x, z = clamp_reachable_target(x, z)

    reach_squared = x * x + z * z
    cos_foot = (
        reach_squared - UPPER_LEG_LENGTH**2 - LOWER_LEG_LENGTH**2
    ) / (2 * UPPER_LEG_LENGTH * LOWER_LEG_LENGTH)
    cos_foot = max(-1.0, min(1.0, cos_foot))

    base_angle = math.atan2(x, z)
    foot_magnitude = math.acos(cos_foot)

    candidates = []
    for foot_angle in (foot_magnitude, -foot_magnitude):
        knee_angle = base_angle - math.atan2(
            LOWER_LEG_LENGTH * math.sin(foot_angle),
            UPPER_LEG_LENGTH + LOWER_LEG_LENGTH * math.cos(foot_angle),
        )
        pose = model_angles_to_offsets(knee_angle, foot_angle)
        pose_error = abs(pose[0] - previous_pose[0]) + abs(pose[1] - previous_pose[1])
        candidates.append((pose_error, pose))

    return min(candidates, key=lambda item: item[0])[1]


def ik_lift_from_contact(contact_pose, body_lift, steps=STAND_STEPS):
    contact_x, contact_z = foot_position_from_offsets(contact_pose[0], contact_pose[1])
    previous_pose = contact_pose

    print(
        "IK contact foot position -> "
        f"x={contact_x:.1f}, z={contact_z:.1f}; lifting body {body_lift:.1f}"
    )

    for step in range(steps + 1):
        t = step / steps
        target_x = contact_x
        target_z = contact_z + body_lift * t
        pose = solve_leg_ik(target_x, target_z, previous_pose)
        pose[0] = CONTACT_FOOT_OFFSET + FOOT_DOWN_EXTRA_DURING_LIFT * t
        pose[1] += KNEE_STAND_EXTRA_DURING_LIFT * t

        set_all_legs_offsets(pose[0], pose[1])
        previous_pose = pose
        time.sleep(STEP_DELAY)

    return previous_pose


def release_all():
    print("Releasing foot and knee servos only.")
    for leg_name in legs:
        legs[leg_name]["foot"].angle = None
        legs[leg_name]["knee"].angle = None


try:
    validate_ik_constants()

    print("Starting IK stand-up sequence.")
    print("Only foot and knee joints will move.")
    print("Hip/body joints are set to 90 once, then left alone.")
    print("Be ready to unplug power if anything binds or tips.")

    contact_pose = [CONTACT_FOOT_OFFSET, CONTACT_KNEE_OFFSET]

    print("Step 1: Set hip/body joints to start angle")
    set_all_hips_to_start_angle(delay=1)

    print("Step 2: Set foot and knee to tucked contact pose")
    set_all_legs_offsets(contact_pose[0], contact_pose[1], delay=2)
    time.sleep(CONTACT_SETTLE_DELAY)

    print("Step 3: IK lift with feet held in place")
    standing_pose = ik_lift_from_contact(
        contact_pose,
        BODY_LIFT_AFTER_CONTACT,
        steps=STAND_STEPS,
    )

    print("Standing pose reached. Holding foot and knee positions.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    release_all()
