import time
from adafruit_servokit import ServoKit

# 16 for the common PCA9685 16-channel servo driver
kit = ServoKit(channels=16)

# Servo plugged into channel 0
servo = kit.servo[0]

# MG996R is usually a 180-degree positional servo.
# Start conservative to avoid forcing the mechanical endpoints.
servo.actuation_range = 180

# Optional: conservative pulse range.
# If the servo buzzes or strains near endpoints, narrow this range.
servo.set_pulse_width_range(600, 2400)

def move(angle, delay=1.0):
    print(f"Moving to {angle} degrees")
    servo.angle = angle
    time.sleep(delay)

try:
    # Center first
    move(90, 2)

    # Small movements first
    move(70, 1)
    move(110, 1)
    move(90, 1)

    # Wider sweep
    move(45, 1)
    move(135, 1)
    move(90, 1)

    # Full-ish sweep, only if everything sounds normal
    move(10, 1)
    move(170, 1)
    move(90, 1)

finally:
    # Release signal. Servo will stop actively holding position.
    servo.angle = None
    print("Done")