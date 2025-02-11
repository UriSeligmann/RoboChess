import RPi.GPIO as GPIO
import time

# GPIO pin definitions
DIR = 23  # Direction pin
PUL = 24  # Pulse pin
ENA = 25  # Enable pin

# Set up GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(PUL, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

def stepper_step(delay, steps):
    for _ in range(steps):
        GPIO.output(PUL, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(PUL, GPIO.LOW)
        time.sleep(delay)

try:
    print("running")
    while True:
        GPIO.output(ENA, GPIO.LOW)  # Enable the driver
        GPIO.output(DIR, GPIO.HIGH)
        stepper_step(0.001, 200)  # Rotate forward
except KeyboardInterrupt:
    GPIO.cleanup()