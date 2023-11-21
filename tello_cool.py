from djitellopy import Tello
import time

try:    
    drone = Tello()
    drone.connect()
    # CHECK SENSOR READINGS------------------------------------------------------------------
    print('Battery, ', drone.get_battery())
    
    drone.turn_motor_on()
    time.sleep(10)
    drone.turn_motor_off()

except KeyboardInterrupt:
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('keyboard interrupt')
    drone.emergency()
    drone.emergency()
    drone.end()
