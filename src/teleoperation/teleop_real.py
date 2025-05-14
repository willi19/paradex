import numpy as np
from scipy.spatial.transform import Rotation

from paradex.io.robot_controller import XArmController, AllegroController, InspireController
from paradex.io.xsens.receiver import XSensReceiver
from paradex.io.contact.receiver import SerialReader
from paradex.io.camera.camera_loader import CameraManager
from paradex.io.robot_controller import retarget

from paradex.utils.file_io import rsc_path

from paradex.utils.file_io import find_latest_index, find_latest_directory

import time
import threading
import os
import chime
import argparse
import json
import shutil
import transforms3d as t3d

hand_name = "inspire"
arm_name = "xarm"

home_wrist_pose = np.array([[0, 1 ,0, 0.5],[0, 0, 1, -0.3],[1, 0, 0, 0.1],[0, 0, 0, 1]])

def load_homepose(hand_name):
    if hand_name == "allegro":
        return  np.load("data/home_pose/allegro_hand_joint_angle.npy")
    elif hand_name == "inspire":
        return np.zeros(6)+500
    

def homo2cart(h):
    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]

        axis, angle = t3d.axangles.mat2axangle(R)
        axis_angle = axis * angle
    else:
        raise ValueError("Invalid input shape.")
    return np.concatenate([t, axis_angle])

def listen_for_exit(stop_event):
    """Listens for 'q' key input to safely exit all processes."""
    while not stop_event.is_set():
        user_input = input()
        if user_input.lower() == "q":
            print("\n[INFO] Exiting program...")
            stop_event.set()  # Set the exit flag
            break




def initialize_teleoperation(save_path):
    controller = {}
    if save_path != None:
        os.makedirs(save_path, exist_ok=True)
        controller["camera"] = CameraManager(save_path, num_cameras=1, is_streaming=False, syncMode=True)
        

    if arm_name == "xarm":
        controller["arm"] = XArmController(save_path)

    if hand_name == "allegro":
        controller["hand"] = AllegroController(save_path)
        if save_path != None:
            controller["contact"] = SerialReader(save_path)
    elif hand_name == "inspire":
        controller["hand"] = InspireController(save_path)
    
    
    controller["xsens"] = XSensReceiver()

    return controller


def main():    
    save_path = None
    print (f"save_path: {save_path}")
    sensors = initialize_teleoperation(save_path)
    
    traj_cnt = 5
    stop_event = threading.Event()

    input_thread = threading.Thread(target=listen_for_exit, args=(stop_event,))
    input_thread.daemon = True  # Ensures the thread exits when the main program exits
    input_thread.start()

    retargetor = retarget.retargetor(arm_name=arm_name, hand_name=hand_name, home_arm_pose=home_wrist_pose)

    homepose_cnt = 0
    grasp_range = {}

    home_hand_pose = load_homepose(hand_name)
    activate_range = {}
    
    for count in range(traj_cnt):
        activate_range[count] = []
        activate_start_time = -1
        activate_end_time = -1
    
        if stop_event.is_set():
            break

        if hand_name is not None:
            sensors["hand"].set_homepose(home_hand_pose)
            sensors["hand"].home_robot()
            
        if arm_name is not None:
            sensors["arm"].set_homepose(homo2cart(home_wrist_pose))
            sensors["arm"].home_robot()

            # alarm during homing
            home_start_time = time.time()
            while sensors["arm"].ready_array[0] != 1:
                if time.time() - home_start_time > 0.3:
                    chime.warning()
                    home_start_time = time.time()
                time.sleep(0.0008)
            chime.success()
        retargetor.reset()
        print("count: =========", count)
        print("Robot homed.")

        grasp_range[count] = {"grasp_start":-1, "grasp_end":-1}

        while not stop_event.is_set():
            # try:
            data = sensors["xsens"].get_data()
            state = data["state"]
            if state == -1: # Xsens not ready
                continue

            state = 0
            if state == 0 or state == 3:
                if activate_start_time == -1:
                    activate_start_time = time.time()

                arm_action, hand_action = retargetor.get_action(data)

            if state == 3:
                if grasp_range[count]["grasp_start"] == -1:
                    grasp_range[count]["grasp_start"] = time.time()
            
            if state != 3 and grasp_range[count]["grasp_start"] != -1:
                grasp_range[count]["grasp_end"] = time.time()    
            
            if state == 1 or state == 2:
                if activate_start_time != -1 and activate_end_time == -1:
                    activate_end_time = time.time()
                    activate_range[count].append((activate_start_time, activate_end_time))
                    activate_start_time = -1
                    activate_end_time = -1

            if state == 1:
                retargetor.pause()
                continue

            if state == 2:
                homepose_cnt += 1
                if homepose_cnt > 30:
                    homepose_cnt = 0
                    arm_action, hand_action = homo2cart(home_wrist_pose), home_hand_pose
                    break
                
            else:
                homepose_cnt = 0

            if arm_name is not None:                
                sensors["arm"].set_target_action(
                                arm_action
                        )
            if hand_name is not None and state != 3:
                sensors["hand"].set_target_action(
                                hand_action
                            )
            # except Exception as e:
            #     print(f"Error: {e}")
            #     break
        

    for key in sensors.keys():
        sensors[key].quit()

    print("Program terminated.")
    exit(0)

if __name__ == "__main__":
    main()