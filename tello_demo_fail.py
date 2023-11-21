import os
import time
import cv2
import numpy as np
from threading import Thread
from djitellopy import Tello

def recordWorker():
    j = 0
    frames_folder = os.path.join(current_path, "frames_thread")

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    while True:
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = os.path.join(frames_folder, f"frame{j}.png")
        cv2.imwrite(filename, frame)
        j += 1
try:
    current_path = os.path.abspath(__file__)
    current_path = current_path.replace("tello_demo_fail.py","")

    drone = Tello()
    drone.connect()
    
    # CHECK SENSOR READINGS------------------------------------------------------------------
    print('Altitude ', drone.get_distance_tof())
    print('Battery, ', drone.get_battery())
    print('Temperature, ', drone.get_temperature())

    drone.streamon()

    Thread(target=recordWorker).start()
    drone.takeoff()
    drone.go_xyz_speed(0,0,30,45)
    time.sleep(4)
    drone.move_up(20)
    drone.move_down(20)
    time.sleep(4)
    drone.move_up(20)
    drone.move_down(20)
    time.sleep(4)
    drone.move_up(20)
    drone.move_down(20)
    time.sleep(5)
    drone.go_xyz_speed(20,10,0,95)
    drone.go_xyz_speed(100,0,0,95)
    drone.land()
    # for j in range(0, 7): # Ignores black frames
    #     frame1 = drone.get_frame_read().frame


    # # visual_servo(drone)
    # # time.sleep(2)
    

    # # Go to a initial location
    # # speed = 45
    # image_center = np.array([480,360])
    # tol = 30
    # runs = 0
    # framei = 0
    # flowi = 0
    
    # centers_dict ={}

    # while True:
    #     frame_no = 0
    #     center_list = []
    #     while frame_no < 1:
            
            
    #         try:
    #             #------Servoing------------------
    #             drone.move_up(20)
    #             time.sleep(1)
    #             drone.move_down(20)
    #             time.sleep(1)
    #             #-------------------------------
    #             #----Reading two frames---------
    #             print("Reading frame 1")
    #             frame1 = drone.get_frame_read().frame
    #             frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    #             H, W, _ = frame1.shape
    #             print("shape", (H, W))
    #             filename = current_path + str("/frames/") + f"frame{framei:03d}.png"
    #             cv2.imwrite(filename, frame1)
    #             frame1 = cv2.imread(filename)
    #             framei +=1
    #             time.sleep(0.2) # keep it low

    #             frame2 = drone.get_frame_read().frame
    #             frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
    #             H, W, _ = frame2.shape
    #             print("shape", (H, W))
    #             filename = current_path + str("/frames/") + f"frame{framei:03d}.png"
    #             cv2.imwrite(filename, frame2)
    #             frame2 = cv2.imread(filename)
    #             # print("shape of frame1",frame1.shape)
    #             # print("shape of frame2",frame2.shape)
    #             #-----------------------------------------
    #             #-------Get optical flow and do post-processing------------------
    #             with torch.no_grad():
    #                 print("starting nn")
    #                 image1 = load_image(frame1)
    #                 image2 = load_image(frame2)
    #                 print("loaded images")
    #                 padder = InputPadder(image1.shape)
    #                 image1, image2 = padder.pad(image1, image2)
    #                 print("padding done")
    #                 flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    #                 print("model ran")
                    
    #                 img = image1[0].permute(1,2,0).cpu().numpy()
    #                 flo = flow_up[0].permute(1,2,0).cpu().numpy()
                    
    #                 # map flow to rgb image
    #                 flo = flow_viz.flow_to_image(flo)
    #                 img_flo = np.concatenate([img, flo], axis=0)
    #                 image_path = current_path+str("/flow/")+f"frame{flowi:03d}.png"
    #                 cv2.imwrite(image_path,flo)
    #                 print("saved the flow",flowi)
    #                 # drone.send_keepalive() 
                    
    #                 cX, cY = postprocess(flowi,current_path,image_path)
    #                 center_list.append([cX,cY])
    #                 frame_no+=1
    #                 flowi+=1
    #                 framei+=1
    #                 # time.sleep(0.5)
    #                 # drone.go_xyz_speed(0,0,-12,speed)
                    
    #                 # import matplotlib.pyplot as plt
    #                 # plt.imshow(img_flo / 255.0)
    #                 # plt.show()
    #                 # speed = 45
    #                 # drone.go_xyz_speed(0,0,30,speed)
    #         except Exception as error:
    #             print(f"An error occurred:{type(error).__name__} - {error}")
    #             continue
    #     # drone.send_keepalive()
    #     runs += 1
    #     centers_dict[f"run{runs}"] = center_list

    #     # Find average center
    #     center_list = np.array(center_list)
    #     average_center = np.mean(center_list, axis=0)

    #     #-----Visual servoing algo------------------
    #     if np.linalg.norm(average_center-image_center)<=tol:
    #         drone.go_xyz_speed(200,0,0,90)
    #         time.sleep(3)
    #         drone.land()
    #         print(centers_dict)
    #         break
    #     # if image_center[0] - average_center[0]>0:
    #     #     y_command = 20
    #     # else:
    #     #     y_command = -20
    #     # if image_center[1] - average_center[1]>0:
    #     #     z_command = 20
    #     # else:
    #     #     z_command = -20
    #     conversion_factor = 0.25
    #     if image_center[0] - average_center[0]>0:
    #         y_command = int(conversion_factor*(abs(image_center[0] - average_center[0])))
    #     else:
    #         y_command = -int(conversion_factor*(abs(image_center[0] - average_center[0])))
    #     if image_center[1] - average_center[1]>0:
    #         z_command = int(conversion_factor*(abs(image_center[1] - average_center[1])))
    #     else:
    #         z_command = -int(conversion_factor*(abs(image_center[1] - average_center[1])))
    #     drone.go_xyz_speed(0,y_command,z_command,45)
    #     time.sleep(3)
except KeyboardInterrupt:
    # drone.land()
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('keyboard interrupt')
    drone.emergency()
    drone.land()
    drone.emergency()
    drone.end()