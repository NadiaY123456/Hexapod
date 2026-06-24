import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16, address=0x40)

CHANNEL = 1       # change this for each servo
angle = 90        # starting guess

servo = kit.servo[CHANNEL]
servo.actuation_range = 180
servo.set_pulse_width_range(700, 2300)

servo.angle = angle
time.sleep(1)

print("Controls: a = -1 degree, d = +1 degree, q = quit")

while True:
    print(f"Current commanded angle: {angle}")
    key = input("> ").strip().lower()

    if key == "a":
        angle -= 1
    elif key == "d":
        angle += 1
    elif key == "q":
        break
    else:
        continue

    angle = max(0, min(180, angle))
    servo.angle = angle
    time.sleep(0.2)

servo.angle = None
print(f"Final angle was: {angle}")