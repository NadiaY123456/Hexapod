import math
import argparse
import os
import select
import struct
import time
from adafruit_servokit import ServoKit

try:
    import board
    import adafruit_mpu6050
except ImportError:
    board = None
    adafruit_mpu6050 = None


def create_mpu_i2c_bus():
    if board is None:
        return None
    try:
        return board.I2C()
    except Exception as error:
        print(f"MPU6050 I2C bus initialization failed: {error}")
        return None


MPU_I2C_BUS = create_mpu_i2c_bus()

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
    "leg1": {"foot": 8.0, "knee": -4.0},
    "leg2": {"foot": 11.0, "knee": -4.0},
    "leg3": {"foot": 8.0, "knee": 1.0},
    "leg4": {"foot": 8.0, "knee": -7.0},
    "leg5": {"foot": 8.5, "knee": -7.0},
    "leg6": {"foot": 8.0, "knee": -8.0},
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

STEP_DELAY = 0.015
CONTACT_SETTLE_DELAY = 0.0
STAND_STEPS = 35
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
DROP_TO_FOOT_ANGLE = 10.0
DROP_TO_KNEE_ANGLE = 88.0
DROP_TO_POSE = [
    DROP_TO_FOOT_ANGLE - NEUTRALS["leg1"]["foot"],
    DROP_TO_KNEE_ANGLE - NEUTRALS["leg1"]["knee"],
]
SIT_FOOT_OFFSET = CONTACT_FOOT_OFFSET + 6.0
SIT_KNEE_OFFSET = CONTACT_KNEE_OFFSET - 18.0
SIT_POSE = [SIT_FOOT_OFFSET, SIT_KNEE_OFFSET]
DROP_STEPS = 20

# Fast tripod gait groups:
#   tripod A = legs 1, 3, 5
#   tripod B = legs 2, 4, 6
WALK_AFTER_STAND = False
CONTROLLER_WALK_AFTER_STAND = True
WALK_CYCLES = 8
TRIPOD_A = ("leg1", "leg3", "leg5")
TRIPOD_B = ("leg2", "leg4", "leg6")

WALK_LIFT_FOOT_DELTA = 34.0
WALK_LIFT_KNEE_DELTA = -23.0
WALK_LIFT_SCALE = {
    "leg1": 1.0,
    "leg2": 1.0,
    "leg3": 1.18,
    "leg4": 1.0,
    "leg5": 1.0,
    "leg6": 1.18,
}
WALK_HIP_SWING_DEG = 15.0
LATERAL_HIP_SWING_DEG = 24.0
RIGHT_CRAB_HIP_SWING_SCALE = 1.15
BODY_YAW_HIP_SWING_DEG = 36.0
TURN_IN_PLACE_HIP_SWING_DEG = 24.0
STEER_WHILE_WALKING_AMOUNT = 0.85
BACKWARD_STEERING_TRIM = 0.12
LATERAL_FORE_AFT_TRIM = 0.18
# Right-veer correction by tripod group. Tripod A is legs 1, 3, 5; tripod B is
# legs 2, 4, 6. If the veer gets worse, swap the A/B scale values.
HIP_SWING_SCALE = {
    "leg1": 0.98,
    "leg2": 1.02,
    "leg3": 0.98,
    "leg4": 1.02,
    "leg5": 0.98,
    "leg6": 1.02,
}
BACKWARD_HIP_SWING_SCALE = {
    "leg1": 1.0,
    "leg2": 1.0,
    "leg3": 1.0,
    "leg4": 1.0,
    "leg5": 1.0,
    "leg6": 1.0,
}
LATERAL_HIP_SWING_SCALE = {
    "leg1": 1.0,
    "leg2": 0.0,
    "leg3": -1.0,
    "leg4": 1.0,
    "leg5": 0.0,
    "leg6": -1.0,
}
ROTATE_HIP_SWING_SCALE = {
    "leg1": 1.0,
    "leg2": 1.0,
    "leg3": 1.0,
    "leg4": -1.0,
    "leg5": -1.0,
    "leg6": -1.0,
}
STANCE_HIP_SCALE = {
    "leg1": 1.0,
    "leg2": 0.60,
    "leg3": 1.0,
    "leg4": 1.0,
    "leg5": 1.0,
    "leg6": 1.0,
}
WALK_HALF_CYCLE_STEPS = 8
WALK_FRAME_DELAY = 0.025
WALK_SETTLE_DELAY = 0.04
ANALOG_WALK_MIN_SPEED_SCALE = 0.35
ANALOG_WALK_MAX_SPEED_SCALE = 1.35

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
AXIS_MIN = -32767
AXIS_MAX = 32767
DPAD_THRESHOLD = 0.50
DPAD_AXES = (6, 7)
LEFT_STICK_AXES = (0, 1)
LEFT_STICK_DEADZONE = 0.25
RIGHT_STICK_AXES = (2, 3)
RIGHT_STICK_DEADZONE = 0.35
RIGHT_STICK_ATTITUDE_DEADZONE = 0.18
BODY_ROLL_FOOT_DEG = 16.0
BODY_ROLL_KNEE_DEG = -12.0
BODY_PITCH_FOOT_DEG = 22.0
BODY_PITCH_KNEE_DEG = -16.0
MPU6050_ADDRESS = 0x68
LEVELING_ENABLED = True
LEVEL_ROLL_GAIN = 0.04
LEVEL_PITCH_GAIN = 0.04
LEVEL_ROLL_SIGN = -1.0
LEVEL_PITCH_SIGN = -1.0
LEVEL_MAX_ATTITUDE = 0.70
LEVEL_FILTER_ALPHA = 0.75
LEVEL_SAMPLE_INTERVAL = 0.05
LEVEL_MAX_READ_ERRORS = 8
# MSI GC30 Linux joystick button numbers. Physical X reports as 3 and physical
# Y reports as 4 on this controller.
A_BUTTON_NUMBERS = (0,)
B_BUTTON_NUMBERS = (1,)
X_BUTTON_NUMBERS = (3,)
Y_BUTTON_NUMBERS = (4,)
DEFAULT_CONTROLLER_DEVICE = "/dev/input/js0"

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


def normalize_axis(value):
    if value < 0:
        return max(-1.0, value / abs(AXIS_MIN))
    return min(1.0, value / AXIS_MAX)


class ControllerInput:
    def __init__(self, device_path):
        self.device_path = device_path
        self.controller = None
        self.axis_values = {
            DPAD_AXES[0]: 0.0,
            DPAD_AXES[1]: 0.0,
            LEFT_STICK_AXES[0]: 0.0,
            LEFT_STICK_AXES[1]: 0.0,
            RIGHT_STICK_AXES[0]: 0.0,
            RIGHT_STICK_AXES[1]: 0.0,
        }
        self.button_values = {
            A_BUTTON_NUMBERS[0]: False,
            B_BUTTON_NUMBERS[0]: False,
            X_BUTTON_NUMBERS[0]: False,
            Y_BUTTON_NUMBERS[0]: False,
        }

    def __enter__(self):
        while True:
            try:
                self.controller = open(self.device_path, "rb")
                os.set_blocking(self.controller.fileno(), False)
                print(f"Controller connected at {self.device_path}.")
                break
            except FileNotFoundError:
                print(f"Waiting for controller at {self.device_path}...")
                time.sleep(1)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.controller is not None:
            self.controller.close()

    def read_event(self):
        readable, _, _ = select.select([self.controller], [], [], 0)
        if readable:
            event = self.controller.read(8)
            if len(event) == 8:
                return struct.unpack("IhBB", event)
        return None


def dpad_direction(axis_values):
    x = axis_values.get(DPAD_AXES[0], 0.0)
    y = axis_values.get(DPAD_AXES[1], 0.0)

    if y <= -DPAD_THRESHOLD:
        return 1
    if y >= DPAD_THRESHOLD:
        return -1
    if x <= -DPAD_THRESHOLD:
        return 2
    if x >= DPAD_THRESHOLD:
        return -2

    return 0


def stick_angle_degrees(x, y):
    # Linux joystick y is usually negative when pushed up.
    angle = math.degrees(math.atan2(x, -y))
    if angle > 180:
        return angle - 360
    if angle <= -180:
        return angle + 360
    return angle


def apply_deadzone(value, deadzone):
    magnitude = abs(value)
    if magnitude < deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(max(0.0, min(1.0, scaled)), value)


def left_stick_steered_walk(axis_values):
    x = axis_values.get(LEFT_STICK_AXES[0], 0.0)
    y = axis_values.get(LEFT_STICK_AXES[1], 0.0)
    travel = max(-1.0, min(1.0, -y))

    if abs(travel) < LEFT_STICK_DEADZONE:
        return 0, None, 0.0

    angle = stick_angle_degrees(x, y)
    direction = 4 if travel > 0.0 else -4
    steering = max(-1.0, min(1.0, x))
    if direction < 0:
        steering = max(-1.0, min(1.0, steering + BACKWARD_STEERING_TRIM))
    return direction, angle, steering


def right_stick_attitude(axis_values):
    x = axis_values.get(RIGHT_STICK_AXES[0], 0.0)
    y = axis_values.get(RIGHT_STICK_AXES[1], 0.0)
    roll = -apply_deadzone(x, RIGHT_STICK_ATTITUDE_DEADZONE)
    pitch = apply_deadzone(y, RIGHT_STICK_ATTITUDE_DEADZONE)
    return roll, pitch


def left_stick_speed_scale(axis_values):
    y = axis_values.get(LEFT_STICK_AXES[1], 0.0)
    travel = apply_deadzone(-y, LEFT_STICK_DEADZONE)
    if travel == 0.0:
        return 1.0
    amount = abs(travel)
    speed_range = ANALOG_WALK_MAX_SPEED_SCALE - ANALOG_WALK_MIN_SPEED_SCALE
    return ANALOG_WALK_MIN_SPEED_SCALE + speed_range * amount


def movement_speed_scale(controller, direction):
    if abs(direction) == 4:
        return left_stick_speed_scale(controller.axis_values)
    return 1.0


def clamp_unit(value):
    return max(-1.0, min(1.0, value))


class LevelingController:
    def __init__(self):
        self.enabled = False
        self.mpu = None
        self.roll = 0.0
        self.pitch = 0.0
        self.read_errors = 0
        self.last_sample_time = 0.0

        if not LEVELING_ENABLED:
            return

        if adafruit_mpu6050 is None:
            print("MPU6050 leveling disabled: adafruit_mpu6050 is not installed.")
            return
        if MPU_I2C_BUS is None:
            print("MPU6050 leveling disabled: I2C bus is unavailable.")
            return

        try:
            self.mpu = adafruit_mpu6050.MPU6050(
                MPU_I2C_BUS,
                address=MPU6050_ADDRESS,
            )
            self.enabled = True
            print(f"MPU6050 leveling enabled at address 0x{MPU6050_ADDRESS:02x}.")
        except Exception as error:
            print(f"MPU6050 leveling disabled: {error}")

    def attitude(self):
        if not self.enabled:
            return {"roll": 0.0, "pitch": 0.0}

        now = time.monotonic()
        if now - self.last_sample_time < LEVEL_SAMPLE_INTERVAL:
            return {"roll": self.roll, "pitch": self.pitch}
        self.last_sample_time = now

        try:
            accel_x, accel_y, accel_z = self.mpu.acceleration
        except Exception as error:
            self.read_errors += 1
            if self.read_errors >= LEVEL_MAX_READ_ERRORS:
                print(
                    "MPU6050 read failed repeatedly; disabling leveling: "
                    f"{error}"
                )
                self.enabled = False
            elif self.read_errors == 1:
                print(f"MPU6050 read failed; retrying leveling reads: {error}")
            return {"roll": 0.0, "pitch": 0.0}

        self.read_errors = 0

        roll_degrees = math.degrees(
            math.atan2(accel_y, math.sqrt(accel_x * accel_x + accel_z * accel_z))
        )
        pitch_degrees = math.degrees(
            math.atan2(-accel_x, math.sqrt(accel_y * accel_y + accel_z * accel_z))
        )

        target_roll = max(
            -LEVEL_MAX_ATTITUDE,
            min(
                LEVEL_MAX_ATTITUDE,
                LEVEL_ROLL_SIGN * roll_degrees * LEVEL_ROLL_GAIN,
            ),
        )
        target_pitch = max(
            -LEVEL_MAX_ATTITUDE,
            min(
                LEVEL_MAX_ATTITUDE,
                LEVEL_PITCH_SIGN * pitch_degrees * LEVEL_PITCH_GAIN,
            ),
        )

        self.roll = LEVEL_FILTER_ALPHA * self.roll + (
            1.0 - LEVEL_FILTER_ALPHA
        ) * target_roll
        self.pitch = LEVEL_FILTER_ALPHA * self.pitch + (
            1.0 - LEVEL_FILTER_ALPHA
        ) * target_pitch

        return {"roll": self.roll, "pitch": self.pitch}


def combined_attitude(manual_attitude, leveler=None):
    level_attitude = (
        leveler.attitude()
        if leveler is not None
        else {"roll": 0.0, "pitch": 0.0}
    )
    return {
        "roll": clamp_unit(manual_attitude.get("roll", 0.0) + level_attitude["roll"]),
        "pitch": clamp_unit(
            manual_attitude.get("pitch", 0.0) + level_attitude["pitch"]
        ),
    }


def button_turn_direction(button_values):
    if any(button_values.get(number, False) for number in B_BUTTON_NUMBERS):
        return 3
    if any(button_values.get(number, False) for number in X_BUTTON_NUMBERS):
        return -3
    return 0


def direction_name(direction, angle=None):
    names = {
        1: "forward",
        -1: "backward",
        2: "left",
        -2: "right",
        3: "rotate left",
        -3: "rotate right",
        4: "steered forward",
        -4: "steered backward",
        0: "paused",
    }
    if angle is None or direction not in (-4, -3, 3, 4):
        return names[direction]
    stick_name = "left stick" if abs(direction) == 4 else "button"
    return f"{names[direction]} ({stick_name} angle {angle:.1f})"


def controller_direction(controller):
    turn_direction = button_turn_direction(controller.button_values)
    if turn_direction:
        return turn_direction, None, 0.0

    walk_direction, walk_angle, steering = left_stick_steered_walk(
        controller.axis_values
    )
    if walk_direction:
        return walk_direction, walk_angle, steering

    return dpad_direction(controller.axis_values), None, 0.0


def movement_controls_centered(axis_values, button_values=None):
    if button_values is None:
        button_values = {}

    return (
        dpad_direction(axis_values) == 0
        and left_stick_steered_walk(axis_values)[0] == 0
        and button_turn_direction(button_values) == 0
    )


def read_latest_controller_direction(controller, attitude):
    latest_command = None
    stop_requested = False
    posture_action = None
    attitude_changed = False

    while True:
        event = controller.read_event()
        if event is None:
            return latest_command, stop_requested, posture_action, attitude_changed

        _, raw_value, event_type, number = event
        event_type_without_init = event_type & ~JS_EVENT_INIT

        if event_type_without_init == JS_EVENT_AXIS and number in (
            DPAD_AXES + LEFT_STICK_AXES
        ):
            controller.axis_values[number] = normalize_axis(raw_value)
            latest_command = controller_direction(controller)
        elif (
            event_type_without_init == JS_EVENT_AXIS
            and number in RIGHT_STICK_AXES
        ):
            controller.axis_values[number] = normalize_axis(raw_value)
            attitude["roll"], attitude["pitch"] = right_stick_attitude(
                controller.axis_values
            )
            attitude_changed = True
        elif event_type_without_init == JS_EVENT_BUTTON and number in (
            A_BUTTON_NUMBERS
            + B_BUTTON_NUMBERS
            + X_BUTTON_NUMBERS
            + Y_BUTTON_NUMBERS
        ):
            controller.button_values[number] = bool(raw_value)
            latest_command = controller_direction(controller)
            if not raw_value:
                continue
            if number in X_BUTTON_NUMBERS or number in B_BUTTON_NUMBERS:
                continue
            if number in A_BUTTON_NUMBERS:
                posture_action = "sit"
            elif number in Y_BUTTON_NUMBERS:
                posture_action = "stand"


def hip_motion_scale(leg_name, direction, steering=0.0):
    if direction in (-4, 4):
        walk_direction = 1 if direction > 0 else -1
        turn_bias = steering * STEER_WHILE_WALKING_AMOUNT
        return walk_direction * (
            1.0 + turn_bias * ROTATE_HIP_SWING_SCALE[leg_name]
        )

    if direction in (-1, 1):
        direction_scale = BACKWARD_HIP_SWING_SCALE[leg_name] if direction < 0 else 1.0
        return direction * direction_scale

    if direction in (-2, 2):
        lateral_direction = 1 if direction > 0 else -1
        return lateral_direction * (
            LATERAL_HIP_SWING_SCALE[leg_name] + LATERAL_FORE_AFT_TRIM
        )

    if direction in (-3, 3):
        rotate_direction = 1 if direction > 0 else -1
        return rotate_direction * ROTATE_HIP_SWING_SCALE[leg_name]

    return 0.0


def body_attitude_offsets(leg_name, attitude):
    roll = attitude.get("roll", 0.0)
    pitch = attitude.get("pitch", 0.0)

    side_sign = 1.0 if leg_name in ("leg1", "leg2", "leg3") else -1.0
    pitch_signs = {
        "leg1": 1.0,
        "leg2": 0.0,
        "leg3": -1.0,
        "leg4": -1.0,
        "leg5": 0.0,
        "leg6": 1.0,
    }
    pitch_sign = pitch_signs[leg_name]

    foot = (
        roll * side_sign * BODY_ROLL_FOOT_DEG
        + pitch * pitch_sign * BODY_PITCH_FOOT_DEG
    )
    knee = (
        roll * side_sign * BODY_ROLL_KNEE_DEG
        + pitch * pitch_sign * BODY_PITCH_KNEE_DEG
    )
    return foot, knee


def set_walk_frame(
    home_pose,
    swing_tripod,
    stance_tripod,
    t,
    direction=1,
    steering=0.0,
    attitude=None,
):
    if attitude is None:
        attitude = {"roll": 0.0, "pitch": 0.0}

    eased_t = t * t * (3 - 2 * t)
    lift = math.sin(math.pi * t)
    if direction in (-2, 2):
        hip_swing = LATERAL_HIP_SWING_DEG
        if direction < 0:
            hip_swing *= RIGHT_CRAB_HIP_SWING_SCALE
    elif direction in (-3, 3):
        hip_swing = TURN_IN_PLACE_HIP_SWING_DEG
    else:
        hip_swing = WALK_HIP_SWING_DEG
    swing_hip = -hip_swing + (2 * hip_swing * eased_t)
    stance_hip = hip_swing - (2 * hip_swing * eased_t)

    for leg_name in swing_tripod:
        hip_scale = hip_motion_scale(leg_name, direction, steering)
        leg_lift = lift * WALK_LIFT_SCALE[leg_name]
        attitude_foot, attitude_knee = body_attitude_offsets(leg_name, attitude)
        swing_foot = home_pose[0] + attitude_foot + WALK_LIFT_FOOT_DELTA * leg_lift
        swing_knee = home_pose[1] + attitude_knee + WALK_LIFT_KNEE_DELTA * leg_lift
        set_leg_offsets(leg_name, swing_foot, swing_knee)
        set_leg_hip_offset(
            leg_name,
            swing_hip * HIP_SWING_SCALE[leg_name] * hip_scale,
        )

    for leg_name in stance_tripod:
        hip_scale = hip_motion_scale(leg_name, direction, steering)
        attitude_foot, attitude_knee = body_attitude_offsets(leg_name, attitude)
        set_leg_offsets(
            leg_name,
            home_pose[0] + attitude_foot,
            home_pose[1] + attitude_knee,
        )
        set_leg_hip_offset(
            leg_name,
            stance_hip
            * HIP_SWING_SCALE[leg_name]
            * hip_scale
            * STANCE_HIP_SCALE[leg_name],
        )


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


def walk_half_cycle(
    home_pose,
    swing_tripod,
    stance_tripod,
    direction=1,
    steering=0.0,
    attitude=None,
):
    for step in range(WALK_HALF_CYCLE_STEPS + 1):
        set_walk_frame(
            home_pose,
            swing_tripod,
            stance_tripod,
            step / WALK_HALF_CYCLE_STEPS,
            direction=direction,
            steering=steering,
            attitude=attitude,
        )
        time.sleep(WALK_FRAME_DELAY)


def hold_standing_pose(home_pose, attitude=None):
    if attitude is None:
        attitude = {"roll": 0.0, "pitch": 0.0}

    for leg_name in legs:
        foot_offset, knee_offset = body_attitude_offsets(leg_name, attitude)
        set_leg_offsets(
            leg_name,
            home_pose[0] + foot_offset,
            home_pose[1] + knee_offset,
        )
    set_all_hip_offsets(0.0, delay=0.0)


def command_sit_pose(home_pose):
    print("A button: sitting.")
    set_all_hip_offsets(0.0, delay=0.0)
    interpolate_pose(home_pose, SIT_POSE, steps=DROP_STEPS)
    print("Rest pose reached. Releasing servos.")
    release_all()


def command_stand_pose(home_pose):
    print("Y button: standing.")
    set_all_hip_offsets(0.0, delay=0.0)
    interpolate_pose(SIT_POSE, home_pose, steps=DROP_STEPS)
    hold_standing_pose(home_pose)


def poll_controller_motion(controller, direction, steering, attitude):
    latest_command, stop_requested, posture_action, _ = (
        read_latest_controller_direction(controller, attitude)
    )

    if stop_requested or posture_action is not None:
        return 0, 0.0, stop_requested, posture_action

    if latest_command is not None:
        direction, _, steering = latest_command

    if movement_controls_centered(controller.axis_values, controller.button_values):
        return 0, 0.0, False, None

    return direction, steering, False, None


def controller_walk_half_cycle(
    home_pose,
    swing_tripod,
    stance_tripod,
    controller,
    direction=1,
    steering=0.0,
    attitude=None,
    leveler=None,
):
    if attitude is None:
        attitude = {"roll": 0.0, "pitch": 0.0}

    for step in range(WALK_HALF_CYCLE_STEPS + 1):
        direction, steering, stop_requested, posture_action = poll_controller_motion(
            controller,
            direction,
            steering,
            attitude,
        )

        if stop_requested or posture_action is not None or not direction:
            hold_standing_pose(home_pose, combined_attitude(attitude, leveler))
            return direction, steering, stop_requested, posture_action, False

        active_attitude = combined_attitude(attitude, leveler)
        set_walk_frame(
            home_pose,
            swing_tripod,
            stance_tripod,
            step / WALK_HALF_CYCLE_STEPS,
            direction=direction,
            steering=steering,
            attitude=active_attitude,
        )
        speed_scale = movement_speed_scale(controller, direction)
        time.sleep(WALK_FRAME_DELAY / speed_scale)

    return direction, steering, False, None, True


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
        f"lateral_hip_swing={LATERAL_HIP_SWING_DEG}, "
        f"right_crab_hip_swing_scale={RIGHT_CRAB_HIP_SWING_SCALE}, "
        f"body_yaw_hip_swing={BODY_YAW_HIP_SWING_DEG}, "
        f"turn_in_place_hip_swing={TURN_IN_PLACE_HIP_SWING_DEG}, "
        f"steer_while_walking={STEER_WHILE_WALKING_AMOUNT}, "
        f"backward_steering_trim={BACKWARD_STEERING_TRIM}, "
        f"lateral_fore_aft_trim={LATERAL_FORE_AFT_TRIM}, "
        f"analog_walk_speed={ANALOG_WALK_MIN_SPEED_SCALE}"
        f"-{ANALOG_WALK_MAX_SPEED_SCALE}, "
        f"hip_scale={HIP_SWING_SCALE}, "
        f"backward_scale={BACKWARD_HIP_SWING_SCALE}, "
        f"lateral_scale={LATERAL_HIP_SWING_SCALE}, "
        f"rotate_scale={ROTATE_HIP_SWING_SCALE}, "
        f"stance_scale={STANCE_HIP_SCALE}, "
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


def controller_walk_control(home_pose, device_path):
    next_swing = TRIPOD_A
    direction = 0
    steering = 0.0
    posture = "stand"
    last_reported_direction = None
    last_reported_steering = None
    last_reported_attitude = None
    attitude = {"roll": 0.0, "pitch": 0.0}
    is_centered = True
    movement_locked = False

    print(
        "Controller walk ready: hold D-pad up/down to walk forward/backward, "
        "left/right to strafe."
    )
    print("Push the left stick up/down to walk forward/backward while steering.")
    print("Push the right stick for small roll/pitch body attitude trims.")
    print("Press A to sit/down. Press Y to stand/up.")
    print("Press B for CCW turn in place. Press X for CW turn in place.")
    print("Release movement controls to pause. Press Ctrl+C to stop.")

    set_all_legs_offsets(home_pose[0], home_pose[1], delay=WALK_SETTLE_DELAY)
    set_all_hip_offsets(0.0, delay=0.0)
    leveler = LevelingController()

    with ControllerInput(device_path) as controller:
        while True:
            latest_command, stop_requested, posture_action, attitude_changed = (
                read_latest_controller_direction(controller, attitude)
            )

            if attitude_changed:
                if posture == "sit":
                    attitude["roll"] = 0.0
                    attitude["pitch"] = 0.0
                else:
                    attitude_snapshot = (
                        round(attitude["roll"], 2),
                        round(attitude["pitch"], 2),
                    )
                    if attitude_snapshot != last_reported_attitude:
                        print(
                            "Right stick attitude: "
                            f"roll {attitude['roll']:.2f}, pitch {attitude['pitch']:.2f}"
                        )
                        last_reported_attitude = attitude_snapshot
                    if not direction:
                        hold_standing_pose(
                            home_pose,
                            combined_attitude(attitude, leveler),
                        )

            if stop_requested:
                direction = 0
                steering = 0.0
                movement_locked = True
                print("Safety stop: holding standing pose.")
                last_reported_direction = direction
                last_reported_steering = steering

            if posture_action == "sit":
                direction = 0
                steering = 0.0
                attitude["roll"] = 0.0
                attitude["pitch"] = 0.0
                command_sit_pose(home_pose)
                posture = "sit"
                is_centered = True
                last_reported_direction = 0
                last_reported_steering = 0.0
                continue

            if posture_action == "stand":
                direction = 0
                steering = 0.0
                attitude["roll"] = 0.0
                attitude["pitch"] = 0.0
                command_stand_pose(home_pose)
                posture = "stand"
                is_centered = True
                last_reported_direction = 0
                last_reported_steering = 0.0
                continue

            if movement_locked:
                if not is_centered:
                    hold_standing_pose(
                        home_pose,
                        combined_attitude(attitude, leveler),
                    )
                    is_centered = True

                if movement_controls_centered(
                    controller.axis_values,
                    controller.button_values,
                ):
                    movement_locked = False
                    print("Movement controls centered. Controller walk ready.")

                time.sleep(0.02)
                continue

            if posture == "sit":
                time.sleep(0.02)
                continue

            if latest_command is not None:
                direction, angle, steering = latest_command

                steering_changed = (
                    last_reported_steering is None
                    or abs(steering - last_reported_steering) >= 0.20
                )
                if direction != last_reported_direction or (
                    abs(direction) == 4 and steering_changed
                ):
                    if abs(direction) == 4:
                        print(
                            "Controller direction: "
                            f"{direction_name(direction, angle)}, steering {steering:.2f}"
                        )
                    else:
                        print(
                            f"Controller direction: {direction_name(direction, angle)}"
                        )
                    last_reported_direction = direction
                    last_reported_steering = steering

            if movement_controls_centered(
                controller.axis_values,
                controller.button_values,
            ):
                direction = 0
                steering = 0.0
                if last_reported_direction not in (None, 0):
                    print("Controller direction: paused")
                    last_reported_direction = 0
                    last_reported_steering = 0.0

            if not direction:
                hold_standing_pose(home_pose, combined_attitude(attitude, leveler))
                is_centered = True
                time.sleep(0.02)
                continue

            is_centered = False
            stance_tripod = TRIPOD_B if next_swing == TRIPOD_A else TRIPOD_A
            (
                direction,
                steering,
                stop_requested,
                posture_action,
                cycle_complete,
            ) = controller_walk_half_cycle(
                home_pose,
                next_swing,
                stance_tripod,
                controller,
                direction=direction,
                steering=steering,
                attitude=attitude,
                leveler=leveler,
            )

            if stop_requested:
                direction = 0
                steering = 0.0
                movement_locked = True
                print("Safety stop: holding standing pose.")
                last_reported_direction = direction
                last_reported_steering = steering

            if posture_action == "sit":
                direction = 0
                steering = 0.0
                attitude["roll"] = 0.0
                attitude["pitch"] = 0.0
                command_sit_pose(home_pose)
                posture = "sit"
                is_centered = True
                last_reported_direction = 0
                last_reported_steering = 0.0
                continue

            if posture_action == "stand":
                direction = 0
                steering = 0.0
                attitude["roll"] = 0.0
                attitude["pitch"] = 0.0
                command_stand_pose(home_pose)
                posture = "stand"
                is_centered = True
                last_reported_direction = 0
                last_reported_steering = 0.0
                continue

            if not direction:
                hold_standing_pose(home_pose, combined_attitude(attitude, leveler))
                is_centered = True
                if last_reported_direction not in (None, 0):
                    print("Controller direction: paused")
                    last_reported_direction = 0
                    last_reported_steering = 0.0
                continue

            if cycle_complete:
                next_swing = stance_tripod

    hold_standing_pose(home_pose, combined_attitude(attitude, leveler))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stand up the hexapod, then walk using an MSI GC30 D-pad."
    )
    parser.add_argument(
        "device",
        nargs="?",
        default=os.environ.get("CONTROLLER_DEVICE", DEFAULT_CONTROLLER_DEVICE),
        help=f"Linux joystick device path. Default: {DEFAULT_CONTROLLER_DEVICE}",
    )
    return parser.parse_args()


def release_all():
    print("Releasing foot, knee, and hip servos.")
    for leg_name in legs:
        legs[leg_name]["foot"].angle = None
        legs[leg_name]["knee"].angle = None
    for hip_servo in hips.values():
        hip_servo.angle = None


def run_stand_up_sequence():
    print("Starting IK stand-up sequence from stand_test_ik.")
    print("Foot and knee joints move during stand-up.")
    print("Hip/body joints are centered during stand-up and sweep during walking.")
    print("Be ready to unplug power if anything binds or tips.")

    contact_pose = [CONTACT_FOOT_OFFSET, CONTACT_KNEE_OFFSET]

    print("Step 1: Set hip/body joints to start angle")
    set_all_hips_to_start_angle(delay=0)

    print("Step 2: Set foot and knee to tucked contact pose")
    set_all_legs_offsets(contact_pose[0], contact_pose[1], delay=0)
    time.sleep(CONTACT_SETTLE_DELAY)

    print("Step 3: IK lift with feet held in place")
    standing_pose = ik_lift_from_contact(
        contact_pose,
        BODY_LIFT_AFTER_CONTACT,
        steps=STAND_STEPS,
    )

    print("Step 4: Drop into the 70a7ad9 final pose")
    interpolate_pose(standing_pose, DROP_TO_POSE, steps=DROP_STEPS)

    return DROP_TO_POSE


try:
    args = parse_args()
    validate_ik_constants()

    walk_home_pose = run_stand_up_sequence()

    if CONTROLLER_WALK_AFTER_STAND:
        print("Step 5: Controller walking control")
        try:
            controller_walk_control(walk_home_pose, args.device)
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
