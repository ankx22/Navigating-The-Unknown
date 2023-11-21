import os
import time
import sys
import csv
import cv2
import numpy as np

sys.path.append('core')
import argparse

import glob
import torch
from PIL import Image
import matplotlib.pyplot as plt

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from djitellopy import Tello
from threading import Thread






DEVICE = 'cuda'#'cpu'#

def load_image(np_image):
    img = torch.from_numpy(np_image).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img,flo,i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    image_path = current_path + str("/flow_outputs/") + f"flow{i:03d}.png"
    # image_path = current_path + "/flow_outputs/flow" + str(i) + ".png"
    # cv2.imwrite(image_path,img_flo[:, :, [2,1,0]]/255.0)
    # final_flo = flo / 255.0
    cv2.imwrite(image_path,flo)
    return flo 


def demo(args,frame_prev,frame_curr,i):
    model = torch.nn.DataParallel(RAFT(args))
    # print("print here",args.model)
    # model.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        np_image1 = frame_prev
        np_image2 = frame_curr

        image1 = load_image(np_image1)
        image2 = load_image(np_image2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        flow_np = viz(image1,flow_up,i)
        return flow_np


def recordWorker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    frame_prev = None
    frame_curr = None
    j = 0
    while True:
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #this is a numpy array
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12)


        print("The shape of the current frame is:",frame.shape)
        # filename = current_path + str("/frames_thread/") + f"frame{j}.png"
        # filename = f'{current_path}/frames_thread/frame{j}.png'
        # filename = os.path.join(current_path, "frames_thread", f"frame{j}.png")
        filename = current_path + "/frames_thread/frame" + str(j) + ".png"

        cv2.imwrite(filename, frame)
        frame_curr = frame.copy()
        if(j > 0):
            current_flow_array = demo(args,frame_prev,frame_curr,j)
            # print("The shape of the current flow array is:",current_flow_array.shape)
        frame_prev = frame_curr.copy()
        j = j+1
        print("value of j is:",j)

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
    current_path = current_path.replace("fly_tello.py","")
    
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
