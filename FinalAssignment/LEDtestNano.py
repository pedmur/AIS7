# This is a test code for the LED

import Jetson.GPIO as GPIO
import time

LED_PIN = 18  # GPIO pin number where the LED is connected

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    print("LED ON")
    GPIO.output(LED_PIN, GPIO.HIGH)  # Turn LED on
    time.sleep(2)                     # Keep it on for 2 seconds
    print("LED OFF")
    GPIO.output(LED_PIN, GPIO.LOW)   # Turn LED off
finally:

    GPIO.cleanup()  # Clean up GPIO settings