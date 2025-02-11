import RPi.GPIO as GPIO
import time

class Stepper:
    def __init__(self, dir_pin, pul_pin, en_pin):
        # Pin parameters are in GPIO formation
        self.dir_pin = dir_pin
        self.pul_pin = pul_pin
        self.en_pin = en_pin
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.dir_pin, GPIO.OUT)
        GPIO.setup(self.pul_pin, GPIO.OUT)
        GPIO.setup(self.en_pin, GPIO.OUT)
    
    def __del__(self):
        GPIO.cleanup()
    
    def step(self, delay, steps, direction):
        # 1 rev = 200 steps
        GPIO.output(self.en_pin, GPIO.LOW)
        GPIO.output(self.dir_pin, direction)
        
        for _ in range(steps):
            GPIO.output(PUL, GPIO.HIGH)
            time.sleep(delay)
            GPIO.output(PUL, GPIO.LOW)
            time.sleep(delay)
            
        GPIO.output(self.en_pin, GPIO.HIGH)
