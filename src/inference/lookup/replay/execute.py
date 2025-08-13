import time
import numpy as np
import chime
import argparse
import json
import os

from paradex.inference.get_lookup_traj import get_traj
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.inference.object_6d import get_current_object_6d
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--inf_index", default=0, type=int)
    parser.add_argument("--obj_name", required=True)
    parser.add_argument("--grasp_type", required=True)

    args = parser.parse_args()
    
    arm_name = "xarm"
    hand_name = "allegro"
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    sensors["hand"] = get_hand(hand_name)
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    
    inf_path = os.path.join(shared_dir, "inference_", "lookup", args.obj_name, args.grasp_type, str(args.inf_index))
    traj = np.load(os.path.join(inf_path, arm_name, "action.npy"))
    hand_traj = np.load(os.path.join(inf_path, hand_name, "action.npy"))

    sensors["arm"].home_robot(traj[0])
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        if time.time() - home_start_time > 0.5:
            chime.warning()
            home_start_time = time.time()
        time.sleep(0.01)
    
    chime.success()
    
    for i in range(len(traj)):
        sensors["arm"].set_action(traj[i])
        sensors["hand"].set_target_action(hand_traj[i])
        time.sleep(0.03)  # Simulate time taken for each action
    
    end_pos= np.array([[0, 0, 1, 0.25],
                    [1, 0, 0, 0.0],
                    [0, 1, 0, 0.10], 
                    [0, 0, 0, 1]])
    sensors["arm"].home_robot(end_pos)
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        time.sleep(0.01)

    chime.info()
    
    for sensor_name, sensor in sensors.items():
        sensor.quit()