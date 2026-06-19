import time
from adafruit_servokit import ServoKit

# PCA9685 16-channel servo driver
kit = ServoKit(channels=16)

# First 3 channels
hip = kit.servo[0]
knee = kit.servo[1]
ankle = kit.servo[2]



servos = [hip, knee, ankle]

# MG996R is usually treated as a 180-degree servo.
# Keep pulse range conservative at first.
for servo in servos:
    servo.actuation_range = 180
    servo.set_pulse_width_range(600, 2400)


def set_all(hip_angle, knee_angle, ankle_angle, delay=1.0):
    print(f"Hip: {hip_angle}, Knee: {knee_angle}, Ankle: {ankle_angle}")

    hip.angle = hip_angle
    knee.angle = knee_angle
    ankle.angle = ankle_angle

    time.sleep(delay)


def release_all():
    for servo in servos:
        servo.angle = None


try:
    print("Centering all servos...")
    set_all(90, 90, 90, 2)

    print("Small synchronized test...")
    set_all(80, 80, 80, 1)
    set_all(100, 100, 100, 1)
    set_all(90, 90, 90, 1)

    print("Alternating leg pose test...")
    set_all(80, 100, 80, 1)
    set_all(100, 80, 100, 1)
    set_all(90, 90, 90, 1)

    print("Individual joint test...")

    # Test hip only
    set_all(70, 90, 90, 1)
    set_all(110, 90, 90, 1)
    set_all(90, 90, 90, 1)

    # Test knee only
    set_all(90, 70, 90, 1)
    set_all(90, 110, 90, 1)
    set_all(90, 90, 90, 1)

    # Test ankle only
    set_all(90, 90, 70, 1)
    set_all(90, 90, 110, 1)
    set_all(90, 90, 90, 1)

    print("Test complete.")

finally:
    print("Releasing servos.")
    release_all()