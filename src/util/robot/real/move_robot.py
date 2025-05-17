import numpy as np
from scipy.spatial.transform import Rotation

from paradex.io.robot_controller import XArmController, AllegroController, InspireController
from paradex.io.camera.camera_loader import CameraManager

import time
import os
import chime
import transforms3d as t3d

hand_name = None # "allegro"
arm_name = "xarm"

# home_wrist_pose = np.load("data/home_pose/allegro_eef_frame.npy")
# home_qpos = np.load("data/home_pose/allegro_robot_action.npy")

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

    elif hand_name == "inspire":
        controller["hand"] = InspireController(save_path)
    
    
    return controller

def homo2cart(h):
    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]

        axis, angle = t3d.axangles.mat2axangle(R)
        axis_angle = axis * angle
    else:
        raise ValueError("Invalid input shape.")
    return np.concatenate([t, axis_angle])


# home_pose = np.array([ 0.5,   0.15 , 0.08 ,-1.2 ,-1.6 ,-1.5])
home_pose = np.array([ 0.5,   -0.3 , 0.2 ,-1.2 ,-1.6 , -1.57])
home_hand_pose = np.zeros(16)

def main():
    sensors = initialize_teleoperation(None)
    
    if hand_name is not None:
        sensors["hand"].set_homepose(home_hand_pose)
        sensors["hand"].home_robot()
            
    if arm_name is not None:
        sensors["arm"].set_homepose(home_pose)
        sensors["arm"].home_robot()

        # alarm during homing
        home_start_time = time.time()
        while sensors["arm"].ready_array[0] != 1:
            if time.time() - home_start_time > 0.3:
                chime.warning()
                home_start_time = time.time()
            time.sleep(0.0008)
        chime.success()
    
    for key in sensors.keys():
        sensors[key].quit()
    print("Robot homed.")

if __name__ == "__main__":
    main()