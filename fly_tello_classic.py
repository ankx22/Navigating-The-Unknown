import os
import time
import sys
import csv
import cv2
import numpy as np

from djitellopy import Tello
from threading import Thread






def recordWorker():
    j = 0
    while True:
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = current_path + str("/frames_thread/") + f"frame{j}.png"
        cv2.imwrite(filename, frame)
        j = j+1
def printodo():# Save the odometry of the drone
  odometry = []
  while True:
    with open('odometry.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, filednames = [ 'mpry', 'baro', 'bat', 'mid', 'h', 'agz', 'temph', 'vgz', 'roll', 'agy', 'yaw', 'pitch', 'vgy', 'time', 'vgx', 'templ', 'agx', 'tof'])
        writer.writeheader()
        writer.writerow(drone.get_current_state())
    # odometry.append(tello.get_current_state())
    print(drone.get_current_state())
#---------------------------------------------------------------------------
#----------Execute visual servoing------------------------------
def visual_servo(tello):
    # tello.streamon()
    time.sleep(2)
    # create and start the movement thread
    Thread(target=recordWorker).start()
    Thread(target=printodo).start()
    speed = 45
    tello.go_xyz_speed(0,0,30,speed)
    time.sleep(2)
    tello.go_xyz_speed(0,0,-30,speed)





try:
    current_path = os.path.abspath(__file__)
    current_path = current_path.replace("fly_tello_classic.py","")
    
    drone = Tello()
    drone.connect()

    # CHECK SENSOR READINGS------------------------------------------------------------------
    print('Altitude ', drone.get_distance_tof())
    print('Battery, ', drone.get_battery())

    drone.streamon()
    drone.takeoff()
    time.sleep(2)
    drone.go_xyz_speed(0,0,50,45)
    time.sleep(2)
    visual_servo(drone)
    time.sleep(2)
    drone.streamoff()
    drone.land()

except KeyboardInterrupt:
    # drone.land()
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('keyboard interrupt')
    drone.emergency()
    drone.emergency()
    drone.end()

# Get the waypoints sent to the drone properly
# Z axis is not reliable
# Check all axis once (drone, camera, world)
# Use map instead of window lenght
#--------------------------------------
# Automate the map file reading
# Tuning parameters
# Need continuous frames for good video
# Change window config to check robustness
# Good visualizations (best)
