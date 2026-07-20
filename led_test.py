from gpiozero import LED
from time import sleep

led = LED(27)  # GPIO27, physical pin 13

led.on()
sleep(10)
led.off()