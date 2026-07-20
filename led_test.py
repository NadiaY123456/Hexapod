from gpiozero import LED
from time import sleep

led = LED(27)  # GPIO27, physical pin 13

try:
    while True:
        led.on()
        sleep(0.5)
        led.off()
        sleep(0.5)

except KeyboardInterrupt:
    led.off()