import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

servos = [kit.servo[0], kit.servo[1], kit.servo[2]]

for servo in servos:
    servo.actuation_range = 180
    servo.set_pulse_width_range(700, 2300)

try:
    print("Moving all servos to center.")
    for servo in servos:
        servo.angle = 90

    time.sleep(3)

    print("Small movement test.")
    positions = [
        (90, 90, 90),
        (85, 90, 90),
        (95, 90, 90),
        (90, 85, 90),
        (90, 95, 90),
        (90, 90, 85),
        (90, 90, 95),
        (90, 90, 90),
    ]

    for hip, knee, ankle in positions:
        print(f"Hip={hip}, Knee={knee}, Ankle={ankle}")
        servos[0].angle = hip
        servos[1].angle = knee
        servos[2].angle = ankle
        time.sleep(1)

finally:
    for servo in servos:
        servo.angle = None