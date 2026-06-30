import math
import sys
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
# This program centers the hip/body joints during stand-up, then uses them for
# the tripod walking gait after the robot is standing.

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

# Per-leg calibration trims. The foot values come from the screenshot and are
# relative to leg1's 21.5 degree reference:
#   leg1=21.5, leg2=24.5, leg3=19.5, leg4=21.5, leg5=30.0, leg6=23.5
TRIMS = {
    "leg1": {"foot": 8.0, "knee": 0.0},
    "leg2": {"foot": 3.0, "knee": 0.0},
    "leg3": {"foot": 8.0, "knee": 0.0},
    "leg4": {"foot": 8.0, "knee": 0.0},
    "leg5": {"foot": 8.5, "knee": 0.0},
    "leg6": {"foot": 8.0, "knee": 0.0},
}

# Hip/body yaw calibration for walking. Legs 1-3 and 4-6 are assumed to be on
# opposite sides, so their hip servos need opposite signs to move the feet in
# the same world direction. If the robot turns in place instead of walking
# forward, flip the signs for legs 4-6.
HIP_NEUTRALS = {
    "leg1": 90,
    "leg2": 90,
    "leg3": 90,
    "leg4": 90,
    "leg5": 90,
    "leg6": 90,
}

HIP_DIRECTIONS = {
    "leg1": 1,
    "leg2": 1,
    "leg3": 1,
    "leg4": -1,
    "leg5": -1,
    "leg6": -1,
}

HIP_TRIMS = {
    "leg1": 5.0,
    "leg2": 0.0,
    "leg3": 5.0,
    "leg4": 0.0,
    "leg5": 2.0,
    "leg6": 15.0,
}

STEP_DELAY = 0.025
CONTACT_SETTLE_DELAY = 1.0
STAND_STEPS = 50
HIP_START_ANGLE = 90

# IK constants measured from leg assembly.step in millimeters.
# The 90-degree servo pose is treated as the starting foot-contact pose.
UPPER_LEG_LENGTH = 65.58  # knee joint to foot joint, projected in the side plane
LOWER_LEG_LENGTH = 78.42  # foot joint to the bottom contact point
BODY_LIFT_AFTER_CONTACT = 15.0
MEASURED_FLAT_FOOT_ANGLE = 30.0

# Start closer to the body so the feet are not so far out from center.
CONTACT_KNEE_OFFSET = -12.0
CONTACT_FOOT_OFFSET = MEASURED_FLAT_FOOT_ANGLE - NEUTRALS["leg1"]["foot"]

# Real-world correction: couple foot motion directly to knee motion. When the
# knees move in the stand direction, the foot joints move out with them so the
# ground contact point stays planted instead of sliding inward.
FOOT_OUT_PER_KNEE_DEGREE = 0.34
KNEE_STAND_EXTRA_DURING_LIFT = 20.0
FOOT_COMMAND_SHIFT = 90.0

# Final pose from commit 70a7ad9, used as a second-phase drop after the
# restored 229222f stand-up pose.
DROP_TO_FOOT_ANGLE = 4.0
DROP_TO_KNEE_ANGLE = 88.0
DROP_TO_POSE = [
    DROP_TO_FOOT_ANGLE - NEUTRALS["leg1"]["foot"],
    DROP_TO_KNEE_ANGLE - NEUTRALS["leg1"]["knee"],
]
DROP_STEPS = 30

# Fast tripod gait groups:
#   tripod A = legs 1, 3, 5
#   tripod B = legs 2, 4, 6
WALK_AFTER_STAND = False
KEYBOARD_WALK_AFTER_STAND = True
WALK_CYCLES = 8
TRIPOD_A = ("leg1", "leg3", "leg5")
TRIPOD_B = ("leg2", "leg4", "leg6")

WALK_LIFT_FOOT_DELTA = 28.0
WALK_LIFT_KNEE_DELTA = -19.0
WALK_HIP_SWING_DEG = 15.0
# Left-veer correction: give the 0x40 side a slightly longer hip stride and
# the 0x41 side a slightly shorter stride. If the veer gets worse, swap these
# scale values between the two sides.
HIP_SWING_SCALE = {
    "leg1": 1.06,
    "leg2": 1.06,
    "leg3": 1.06,
    "leg4": 0.94,
    "leg5": 0.94,
    "leg6": 0.94,
}
WALK_HALF_CYCLE_STEPS = 7
WALK_FRAME_DELAY = 0.018
WALK_SETTLE_DELAY = 0.03
KEY_RELEASE_TIMEOUT = 0.25

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
    trim = TRIMS[leg_name][joint_name]
    command_shift = FOOT_COMMAND_SHIFT if joint_name == "foot" else 0.0
    return clamp_angle(neutral + direction * offset + trim + command_shift)


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


def set_selected_legs_offsets(leg_names, foot_offset, knee_offset):
    for leg_name in leg_names:
        set_leg_offsets(leg_name, foot_offset, knee_offset)


def set_all_hips_to_start_angle(delay=1.0):
    print("Setting all hip/body joints to calibrated center.")

    for leg_name in hips:
        set_leg_hip_offset(leg_name, 0.0)

    if delay > 0:
        time.sleep(delay)


def hip_angle_from_offset(leg_name, hip_offset):
    neutral = HIP_NEUTRALS[leg_name]
    direction = HIP_DIRECTIONS[leg_name]
    trim = HIP_TRIMS[leg_name]
    return clamp_angle(neutral + direction * hip_offset + trim)


def set_leg_hip_offset(leg_name, hip_offset):
    hips[leg_name].angle = hip_angle_from_offset(leg_name, hip_offset)


def set_all_hip_offsets(hip_offset, delay=0.0):
    for leg_name in hips:
        set_leg_hip_offset(leg_name, hip_offset)

    if delay > 0:
        time.sleep(delay)


def interpolate_hip_offsets(group_offsets_start, group_offsets_end, steps=60, delay=STEP_DELAY):
    for step in range(steps + 1):
        t = step / steps

        for leg_name, start_offset in group_offsets_start.items():
            end_offset = group_offsets_end[leg_name]
            hip_offset = start_offset + (end_offset - start_offset) * t
            set_leg_hip_offset(leg_name, hip_offset)

        time.sleep(delay)


def interpolate_pose(start_pose, end_pose, steps=60):
    # Pose format: [foot_offset, knee_offset]
    for step in range(steps + 1):
        t = step / steps

        foot = start_pose[0] + (end_pose[0] - start_pose[0]) * t
        knee = start_pose[1] + (end_pose[1] - start_pose[1]) * t

        set_all_legs_offsets(foot, knee)
        time.sleep(STEP_DELAY)


def interpolate_selected_pose(leg_names, start_pose, end_pose, steps=60):
    # Pose format: [foot_offset, knee_offset]
    for step in range(steps + 1):
        t = step / steps

        foot = start_pose[0] + (end_pose[0] - start_pose[0]) * t
        knee = start_pose[1] + (end_pose[1] - start_pose[1]) * t

        set_selected_legs_offsets(leg_names, foot, knee)
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
        pose[1] += KNEE_STAND_EXTRA_DURING_LIFT * t
        knee_delta = pose[1] - CONTACT_KNEE_OFFSET
        pose[0] = CONTACT_FOOT_OFFSET + FOOT_OUT_PER_KNEE_DEGREE * knee_delta

        set_all_legs_offsets(pose[0], pose[1])
        previous_pose = pose
        time.sleep(STEP_DELAY)

    return previous_pose


class KeyboardInput:
    def __enter__(self):
        self.is_windows = sys.platform.startswith("win")
        self.old_settings = None

        if self.is_windows:
            import msvcrt

            self.msvcrt = msvcrt
        else:
            if not sys.stdin.isatty():
                raise RuntimeError("Keyboard walk needs an interactive terminal.")

            import select
            import termios
            import tty

            self.select = select
            self.termios = termios
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_settings is not None:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_settings)

    def read_key(self):
        if self.is_windows:
            if self.msvcrt.kbhit():
                return self.msvcrt.getwch().lower()
            return None

        readable, _, _ = self.select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read(1).lower()
        return None


def read_latest_walk_key(keyboard):
    latest_key = None
    while True:
        key = keyboard.read_key()
        if key is None:
            return latest_key
        if key in ("w", "s", "q"):
            latest_key = key


def set_walk_frame(home_pose, swing_tripod, stance_tripod, t, direction=1):
    eased_t = t * t * (3 - 2 * t)
    lift = math.sin(math.pi * t)
    swing_hip = direction * (
        -WALK_HIP_SWING_DEG + (2 * WALK_HIP_SWING_DEG * eased_t)
    )
    stance_hip = direction * (
        WALK_HIP_SWING_DEG - (2 * WALK_HIP_SWING_DEG * eased_t)
    )
    swing_foot = home_pose[0] + WALK_LIFT_FOOT_DELTA * lift
    swing_knee = home_pose[1] + WALK_LIFT_KNEE_DELTA * lift

    for leg_name in swing_tripod:
        set_leg_offsets(leg_name, swing_foot, swing_knee)
        set_leg_hip_offset(leg_name, swing_hip * HIP_SWING_SCALE[leg_name])

    for leg_name in stance_tripod:
        set_leg_offsets(leg_name, home_pose[0], home_pose[1])
        set_leg_hip_offset(leg_name, stance_hip * HIP_SWING_SCALE[leg_name])


def tripod_start_offsets(direction=1):
    all_leg_names = tuple(legs.keys())
    return {
        leg_name: (
            direction
            * (
                -WALK_HIP_SWING_DEG
                if leg_name in TRIPOD_A
                else WALK_HIP_SWING_DEG
            )
            * HIP_SWING_SCALE[leg_name]
        )
        for leg_name in all_leg_names
    }


def walk_half_cycle(home_pose, swing_tripod, stance_tripod, direction=1):
    for step in range(WALK_HALF_CYCLE_STEPS + 1):
        set_walk_frame(
            home_pose,
            swing_tripod,
            stance_tripod,
            step / WALK_HALF_CYCLE_STEPS,
            direction=direction,
        )
        time.sleep(WALK_FRAME_DELAY)


def walk_tripod_cycles(home_pose, cycles=WALK_CYCLES):
    all_leg_names = tuple(legs.keys())
    hip_center = {leg_name: 0.0 for leg_name in all_leg_names}
    hip_start = tripod_start_offsets(direction=1)

    print(
        "Starting fast small-step tripod walk: "
        f"A={TRIPOD_A}, B={TRIPOD_B}, cycles={cycles}"
    )
    print(
        "Walk tuning -> "
        f"hip_swing={WALK_HIP_SWING_DEG}, "
        f"hip_scale={HIP_SWING_SCALE}, "
        f"lift_foot={WALK_LIFT_FOOT_DELTA}, "
        f"lift_knee={WALK_LIFT_KNEE_DELTA}, "
        f"steps={WALK_HALF_CYCLE_STEPS}, delay={WALK_FRAME_DELAY}"
    )

    set_all_legs_offsets(home_pose[0], home_pose[1], delay=WALK_SETTLE_DELAY)
    interpolate_hip_offsets(
        hip_center,
        hip_start,
        steps=WALK_HALF_CYCLE_STEPS,
        delay=WALK_FRAME_DELAY,
    )

    for cycle in range(1, cycles + 1):
        print(f"Walk cycle {cycle}: tripod A swing, tripod B stance")
        walk_half_cycle(home_pose, TRIPOD_A, TRIPOD_B, direction=1)

        print(f"Walk cycle {cycle}: tripod B swing, tripod A stance")
        walk_half_cycle(home_pose, TRIPOD_B, TRIPOD_A, direction=1)

    print("Tripod walk complete.")
    set_all_legs_offsets(home_pose[0], home_pose[1])
    interpolate_hip_offsets(
        hip_start,
        hip_center,
        steps=WALK_HALF_CYCLE_STEPS,
        delay=WALK_FRAME_DELAY,
    )


def keyboard_walk_control(home_pose):
    next_swing = TRIPOD_A
    direction = 0
    last_key_time = 0.0

    print("Keyboard walk ready: hold W to walk forward, hold S to walk backward.")
    print("Release the key to pause. Press Q to stop and hold standing pose.")

    set_all_legs_offsets(home_pose[0], home_pose[1], delay=WALK_SETTLE_DELAY)
    set_all_hip_offsets(0.0, delay=0.0)

    with KeyboardInput() as keyboard:
        while True:
            key = read_latest_walk_key(keyboard)
            now = time.monotonic()

            if key == "q":
                print("Keyboard walk stopped.")
                break
            if key == "w":
                direction = 1
                last_key_time = now
            elif key == "s":
                direction = -1
                last_key_time = now

            if direction and now - last_key_time > KEY_RELEASE_TIMEOUT:
                direction = 0
                set_all_legs_offsets(home_pose[0], home_pose[1])
                set_all_hip_offsets(0.0, delay=0.0)

            if not direction:
                time.sleep(0.02)
                continue

            stance_tripod = TRIPOD_B if next_swing == TRIPOD_A else TRIPOD_A
            walk_half_cycle(home_pose, next_swing, stance_tripod, direction=direction)
            next_swing = stance_tripod

    set_all_legs_offsets(home_pose[0], home_pose[1])
    set_all_hip_offsets(0.0, delay=0.0)


def release_all():
    print("Releasing foot, knee, and hip servos.")
    for leg_name in legs:
        legs[leg_name]["foot"].angle = None
        legs[leg_name]["knee"].angle = None
    for hip_servo in hips.values():
        hip_servo.angle = None


try:
    validate_ik_constants()

    print("Starting IK stand-up sequence.")
    print("Foot and knee joints move during stand-up.")
    print("Hip/body joints are centered during stand-up and sweep during walking.")
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

    print("Step 4: Drop into the 70a7ad9 final pose")
    interpolate_pose(standing_pose, DROP_TO_POSE, steps=DROP_STEPS)

    if KEYBOARD_WALK_AFTER_STAND:
        print("Step 5: Keyboard walking control")
        try:
            keyboard_walk_control(DROP_TO_POSE)
        except RuntimeError as error:
            print(f"{error} Holding standing pose instead.")
    elif WALK_AFTER_STAND:
        print("Step 5: Run tripod walking cycles")
        walk_tripod_cycles(DROP_TO_POSE, cycles=WALK_CYCLES)

    print("Standing pose reached. Holding foot and knee positions.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    release_all()
