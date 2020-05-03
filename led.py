import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import time # Import the sleep function from the time module

blue = 5
yellow = 6 
GPIO.setmode(GPIO.BCM) # Use physical pin numbering
GPIO.setup(blue, GPIO.OUT)
GPIO.setup(yellow, GPIO.OUT)
GPIO.setwarnings(False)

while True:
    GPIO.output(blue, True)
    GPIO.output(yellow, False)
    time.sleep(1)
    GPIO.output(blue, False)
    GPIO.output(yellow, True)
    time.sleep(1)
