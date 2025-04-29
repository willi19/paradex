import numpy as np
from scipy.spatial.transform import Rotation

from dex_robot.io.robot_controller import XArmController, AllegroController, InspireController
from dex_robot.io.xsens.receiver import XSensReceiver
from dex_robot.io.contact.receiver import SerialReader
from dex_robot.io.camera.camera_loader import CameraManager
from dex_robot.io.robot_controller import retarget

from dex_robot.utils.file_io import rsc_path, capture_path, shared_path

from paradex.utils.io import find_latest_index, find_latest_directory

import time
import threading
import os
import chime
import argparse
import json
import shutil
import transforms3d as t3d

hand_name = "allegro"
arm_name = "xarm"

home_wrist_pose = np.load("data/home_pose/allegro_eef_frame.npy")
home_qpos = np.load("data/home_pose/allegro_robot_action.npy")

def listen_for_exit(stop_event):
    """Listens for 'q' key input to safely exit all processes."""
    while not stop_event.is_set():
        user_input = input()
        if user_input.lower() == "q":
            print("\n[INFO] Exiting program...")
            stop_event.set()  # Set the exit flag
            break

def copy_calib_files(save_path):

    handeye_calib_dir = os.path.join(shared_path, "handeye_calibration")
    handeye_calib_name = find_latest_directory(handeye_calib_dir)
    handeye_calib_path = os.path.join(shared_path, "handeye_calibration", handeye_calib_name, "0", "C2R.npy")

    camparam_dir = os.path.join(shared_path, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_path, "cam_param", camparam_name)

    shutil.copyfile(handeye_calib_path, os.path.join(save_path, "C2R.npy"))
    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))

def load_savepath(name):
    if name == None:
        return None
    index = int(find_latest_index(os.path.join(capture_path, name)))+1
    return os.path.join(capture_path, name, str(index))

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


traj_start = []
traj_end = []

traj_start.append(homo2cart(home_wrist_pose))
traj_end.append(np.array([ 0.5,   0.15 , 0.08 ,-1.2 ,-1.6 ,-1.5]))

traj_start.append(np.array([ 0.5,   -0.1 , 0.12568464 ,-1.00128184 ,-1.49969864 ,-1.37346572]))
traj_end.append(np.array([ 0.5,   0.1 , 0.12568464 ,-1.00128184 ,-1.49969864 ,-1.37346572]))

traj_start.append(np.array([ 0.4,   0.0 , 0.12 ,-1.00128184 ,-1.49969864 ,-1.37346572]))
traj_end.append(np.array([ 0.6,   0.0 , 0.12 ,-1.00128184 ,-1.49969864 ,-1.37346572]))

traj_start.append(np.array([ 0.5,   0.0 , 0.05 ,-1.00128184 ,-1.49969864 ,-1.37346572]))
traj_end.append(np.array([ 0.5,   0.0 , 0.15 ,-1.00128184 ,-1.49969864 ,-1.37346572]))

def get_test_traj(index, t):
    # trajectory have length 30
    ts = traj_start[index]
    te = traj_end[index]

    if index == 0:
        pos = ts + min(1,(t / 10)) * (te - ts)
    
    elif index in [1, 2, 3]:
        # 주기 10, 진폭 (te - ts), t는 실수 가능
        A = (te - ts) / 2
        mid = (ts + te) / 2
        pos = mid + A * np.sin(2 * np.pi * t / 6)

    return pos

joint_limit = []

joint_limit.append([-0.2, 0.2])
joint_limit.append([-0.296, 1.71])

def get_hand_traj(t):
    hand_traj = np.zeros(16)

    for fi in range(3):
        for fj in range(2):
            hand_traj[4*fi+fj] = (joint_limit[fj][0] + joint_limit[fj][1]) / 2 + \
                np.sin(2 * np.pi * t / 6 + fi * np.pi / 4) * (joint_limit[fj][1] - joint_limit[fj][0]) / 2
    return hand_traj

def get_start_pose(index):
    if index == 0:
        return traj_start[index]
    elif index in [1, 2, 3]:
        return traj_start[index] + (traj_end[index] - traj_start[index]) / 2


def main():
    parser = argparse.ArgumentParser(description="Teleoperation for real robot")
    parser.add_argument("--name", type=str, help="Control mode for robot", default="TimeSync")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(capture_path, args.name), exist_ok=True)
    save_path = load_savepath(args.name)

    print (f"save_path: {save_path}")
    sensors = initialize_teleoperation(save_path)
    
    traj_cnt = 4
    stop_event = threading.Event()

    activate_range = {}

    home_hand_pose = get_hand_traj(0)

    for count in range(traj_cnt):
        activate_range[count] = []
        activate_start_time = -1
        
        start_pose = get_start_pose(count)
        if hand_name is not None:
            sensors["hand"].set_homepose(home_hand_pose)
            sensors["hand"].home_robot()
            
        if arm_name is not None:
            sensors["arm"].set_homepose(start_pose)
            sensors["arm"].home_robot()

            # alarm during homing
            home_start_time = time.time()
            while sensors["arm"].ready_array[0] != 1:
                if time.time() - home_start_time > 0.3:
                    chime.warning()
                    home_start_time = time.time()
                time.sleep(0.0008)
            chime.success()

        print("count: =========", count)
        print("Robot homed.")

        activate_start_time = time.time()
        while not stop_event.is_set():
            t = time.time() - activate_start_time
            arm_action, hand_action = get_test_traj(count, t), get_hand_traj(t)
            
            if t > 20:
                break
                
            if arm_name is not None:                
                sensors["arm"].set_target_action(
                                arm_action
                        )
            if hand_name is not None:
                sensors["hand"].set_target_action(
                                hand_action
                            )
        activate_range[count].append([activate_start_time, time.time()])
            
    if save_path != None:
        json.dump(activate_range, open(os.path.join(save_path, "activate_range.json"), 'w'))
        copy_calib_files(save_path)

    for key in sensors.keys():
        sensors[key].quit()

    print("Program terminated.")
    exit(0)

if __name__ == "__main__":
    main()