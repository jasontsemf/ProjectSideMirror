import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import time # Import the sleep function from the time module

GPIO.setmode(GPIO.BCM) # Use physical pin numbering
GPIO.setup(4, GPIO.OUT)
GPIO.setwarnings(False)

while True:
    GPIO.output(4, True)
    time.sleep(1)
    GPIO.output(4, False)
    time.sleep(1)
