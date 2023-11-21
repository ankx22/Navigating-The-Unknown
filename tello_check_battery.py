from djitellopy import Tello

try:    
    drone = Tello()
    drone.connect()
    # CHECK SENSOR READINGS------------------------------------------------------------------
    print('Drone battery is: ', drone.get_battery())
    print(f"Drone Temperature is: {drone.get_temperature()}")

except KeyboardInterrupt:
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('keyboard interrupt')
    drone.emergency()
    drone.emergency()
    drone.end()
