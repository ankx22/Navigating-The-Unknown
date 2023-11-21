import os
import time
import sys
import csv
import cv2
import numpy as np
import argparse
import glob
import torch
from PIL import Image
import matplotlib.pyplot as plt
from threading import Thread
from djitellopy import Tello

sys.path.append('RAFT/core') # This is in raft folder

from raft import RAFT #in RAFT folder
from utils import flow_viz #in RAFT folder
from utils.utils import InputPadder #in RAFT folder

DEVICE = 'cuda'
######-NN Related functions#######################################
def load_image(imfile):
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(imfile).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def recordWorker():
    j = 0
    while True:
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = current_path + str("/frames_thread/") + f"frame{j}.png"
        cv2.imwrite(filename, frame)
        j = j+1
# def printodo():# Save the odometry of the drone
#   odometry = []
#   while True:
#     with open('odometry.csv', mode='w', newline='') as file:
#         writer = csv.DictWriter(file, filednames = [ 'mpry', 'baro', 'bat', 'mid', 'h', 'agz', 'temph', 'vgz', 'roll', 'agy', 'yaw', 'pitch', 'vgy', 'time', 'vgx', 'templ', 'agx', 'tof'])
#         writer.writeheader()
#         writer.writerow(drone.get_current_state())
#     # odometry.append(tello.get_current_state())
#     print(drone.get_current_state())
#---------------------------------------------------------------------------
#----------Execute visual servoing------------------------------
# def visual_servo(tello):
#     # tello.streamon()
#     time.sleep(2)
#     # create and start the movement thread
#     # Thread(target=recordWorker).start()
#     # Thread(target=printodo).start()
#     speed = 45
#     tello.go_xyz_speed(0,0,30,speed)
#     time.sleep(2)
#     tello.go_xyz_speed(0,0,-30,speed)

#-################--Post processing-#######################
# Helper function to calculate the area of a contour
def contour_area(contour):
    return cv2.contourArea(contour)

def postprocess(i,current_path,image_path):
    # Load the image
    image = cv2.imread(image_path)

    # If the image has an alpha channel, remove it
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 1: Noise reduction with Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    # print("statement 1")
    # Use adaptive thresholding to get a binary image
    adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 2)
    # print("statement 2")
    # Use Canny edge detection
    edges = cv2.Canny(adaptive_thresh, 50, 150)
    # print("statement 3")
    # Dilate the edges to close the gaps
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    print("statement 4")
    # Apply closing to fill in gaps
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # # Step 3: Morphological operations to remove small objects (minor gaps)

    # cv2.imshow('binary_image',closed_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 4: Find contours of the remaining objects (gaps)
    contours, hierarchy = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Sort contours by area and get the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Draw the largest contour and centroid if it exists
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Threshold to filter small contours
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                print("The centroid of the largest contour detected is:", cX, ",", cY)
#---------------------------------------------------------------------------------
    # # Step 2: Apply Otsu's method to perform thresholding
    # ret, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Step 3: Morphological operations to remove small objects (minor gaps)
    # # Define the structuring element
    # kernel = np.ones((3, 3), np.uint8)
    # cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # # Step 4: Find contours of the remaining objects (gaps)
    # contours, hierarchy = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Step 5: Calculate the area of each contour to find the largest one
    # largest_area = 0
    # largest_contour = None
    # for contour in contours:
    #     area = contour_area(contour)
    #     if area > largest_area:
    #         largest_area = area
    #         largest_contour = contour
    # # cleaned_image = (cleaned_image * 255).astype(np.uint8)
    # # cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2RGB)
    # # Draw the largest contour and centroid if it exists
    # if largest_contour is not None:
    #     # Draw the largest contour
    #     cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)

    #     # Calculate the centroid of the contour
    #     M = cv2.moments(largest_contour)
    #     if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #         cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
    #         print("The centroid of the gap detected is:", cX, ",", cY)
    #     else:
    #         print("Can't compute centroid as contour area is zero.")
#------------------------------------------------------------------------
    filepath = current_path + f"/center/frame{i:03d}.png"
    cv2.imwrite(filepath,image)

    return cX, cY
##########################################################################################

try:
    current_path = os.path.abspath(__file__)
    current_path = current_path.replace("fly_tello_mine.py","")

    

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    drone = Tello()
    drone.connect()
    # CHECK SENSOR READINGS------------------------------------------------------------------
    print('Altitude ', drone.get_distance_tof())
    print('Battery, ', drone.get_battery())
    print('Temperature, ', drone.get_temperature())

    drone.streamon()
    drone.takeoff()
    time.sleep(2)
    drone.go_xyz_speed(0,0,30,45)
    time.sleep(2)
    for j in range(0, 7): # Ignores black frames
        frame1 = drone.get_frame_read().frame


    # visual_servo(drone)
    # time.sleep(2)
    

    # Go to a initial location
    # speed = 45
    image_center = np.array([480,360])
    tol = 30
    runs = 0
    framei = 0
    flowi = 0
    
    centers_dict ={}

    while True:
        frame_no = 0
        center_list = []
        while frame_no < 1:
            
            
            try:
                #------Servoing------------------
                drone.move_up(20)
                time.sleep(1)
                drone.move_down(20)
                time.sleep(1)
                #-------------------------------
                #----Reading two frames---------
                print("Reading frame 1")
                frame1 = drone.get_frame_read().frame
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
                H, W, _ = frame1.shape
                print("shape", (H, W))
                filename = current_path + str("/frames/") + f"frame{framei:03d}.png"
                cv2.imwrite(filename, frame1)
                frame1 = cv2.imread(filename)
                framei +=1
                time.sleep(0.2) # keep it low

                frame2 = drone.get_frame_read().frame
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
                H, W, _ = frame2.shape
                print("shape", (H, W))
                filename = current_path + str("/frames/") + f"frame{framei:03d}.png"
                cv2.imwrite(filename, frame2)
                frame2 = cv2.imread(filename)
                # print("shape of frame1",frame1.shape)
                # print("shape of frame2",frame2.shape)
                #-----------------------------------------
                #-------Get optical flow and do post-processing------------------
                with torch.no_grad():
                    print("starting nn")
                    image1 = load_image(frame1)
                    image2 = load_image(frame2)
                    print("loaded images")
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)
                    print("padding done")
                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                    print("model ran")
                    
                    img = image1[0].permute(1,2,0).cpu().numpy()
                    flo = flow_up[0].permute(1,2,0).cpu().numpy()
                    
                    # map flow to rgb image
                    flo = flow_viz.flow_to_image(flo)
                    img_flo = np.concatenate([img, flo], axis=0)
                    image_path = current_path+str("/flow/")+f"frame{flowi:03d}.png"
                    cv2.imwrite(image_path,flo)
                    print("saved the flow",flowi)
                    # drone.send_keepalive() 
                    
                    cX, cY = postprocess(flowi,current_path,image_path)
                    center_list.append([cX,cY])
                    frame_no+=1
                    flowi+=1
                    framei+=1
                    # time.sleep(0.5)
                    # drone.go_xyz_speed(0,0,-12,speed)
                    
                    # import matplotlib.pyplot as plt
                    # plt.imshow(img_flo / 255.0)
                    # plt.show()
                    # speed = 45
                    # drone.go_xyz_speed(0,0,30,speed)
            except Exception as error:
                print(f"An error occurred:{type(error).__name__} - {error}")
                continue
        # drone.send_keepalive()
        runs += 1
        centers_dict[f"run{runs}"] = center_list

        # Find average center
        center_list = np.array(center_list)
        average_center = np.mean(center_list, axis=0)

        #-----Visual servoing algo------------------
        if np.linalg.norm(average_center-image_center)<=tol:
            drone.go_xyz_speed(200,0,0,90)
            time.sleep(3)
            drone.land()
            print(centers_dict)
            break
        # if image_center[0] - average_center[0]>0:
        #     y_command = 20
        # else:
        #     y_command = -20
        # if image_center[1] - average_center[1]>0:
        #     z_command = 20
        # else:
        #     z_command = -20
        conversion_factor = 0.5
        if image_center[0] - average_center[0]>0:
            y_command = int(conversion_factor*(abs(image_center[0] - average_center[0])))
        else:
            y_command = -int(conversion_factor*(abs(image_center[0] - average_center[0])))
        if image_center[1] - average_center[1]>0:
            z_command = int(conversion_factor*(abs(image_center[1] - average_center[1])))
        else:
            z_command = -int(conversion_factor*(abs(image_center[1] - average_center[1])))
        drone.go_xyz_speed(0,y_command,z_command,45)
        time.sleep(3)
except KeyboardInterrupt:
    # drone.land()
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('keyboard interrupt')
    drone.emergency()
    drone.land()
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
