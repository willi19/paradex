import time
import numpy as np
import chime
import argparse
import json
import os

from paradex.inference.lookup_table import get_traj
from paradex.inference.util import home_robot

from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.inference.object_6d import get_current_object_6d, normalize_cylinder
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--obj_name", required=True)
    parser.add_argument("--grasp_type", required=True)

    args = parser.parse_args()
    
    arm_name = "xarm"
    hand_name = "allegro"
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    sensors["hand"] = get_hand(hand_name)
    
    demo_idx = args.index
    demo_path = os.path.join("data", "lookup", args.obj_name, args.grasp_type, str(demo_idx))

    pick_traj = np.load(f"{demo_path}/pick.npy")
    place_traj = np.load(f"{demo_path}/place.npy")
    pick_hand_traj = np.load(f"{demo_path}/pick_hand.npy")
    place_hand_traj = np.load(f"{demo_path}/place_hand.npy")
    
    place_position_list = json.load(open(f"data/lookup/{args.obj_name}/obj_pose.json"))
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.35],
                        [0, 1, 0, 0.15], 
                        [0, 0, 0, 1]])
    
    end_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.15], 
                        [0, 0, 0, 1]])
    
    place_id = 3
    place_6D = place_position_list[place_id]
    
    while True:
        home_robot(start_pos.copy())        
        
        # retister object
        pick_6D = get_current_object_6d(args.obj_name)
        pick_6d = normalize_cylinder(pick_6D)
            
        place_6D = np.array(place_position_list[place_id])
        
        traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)
        
        # Prepare execution
        sensors["arm"].home_robot(traj[0])
        
        for i in range(len(traj)):
            sensors["arm"].set_action(traj[i])
            sensors["hand"].set_target_action(hand_traj[i])
            time.sleep(0.03)  # Simulate time taken for each action
        
    sensors["arm"].home_robot(end_pos)
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        time.sleep(0.01)

    chime.info()
    
    for sensor_name, sensor in sensors.items():
        sensor.quit()