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

def get_pose(position):
    pose = np.eye(4)
    pose[:3,:3] = np.array([[0,0,1],[1,0,0],[0,1,0]])
    
    pose[:3,3] = position
    return pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--obj_name", required=True)
    parser.add_argument("--grasp_type", required=True)

    args = parser.parse_args()
    
    path_planning = [(1, 2), (2, 4), (4, 2), (2, 3), (3, 1), (1, 3), (3, 4), (4, 1), (1, 4), (4, 3), (3, 2), (2, 1)]
    
    arm_name = "xarm"
    hand_name = "allegro"
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    # sensors["hand"] = get_hand(hand_name)
    
    demo_idx = args.index
    demo_path = os.path.join("data", "lookup", args.obj_name, args.grasp_type, str(demo_idx))

    pick_traj = np.load(f"{demo_path}/pick.npy")
    place_traj = np.load(f"{demo_path}/place.npy")
    pick_hand_traj = np.load(f"{demo_path}/pick_hand.npy")
    place_hand_traj = np.load(f"{demo_path}/place_hand.npy")
    
    place_position_list = json.load(open(f"data/lookup/{args.obj_name}/obj_pose.json"))
    
    # way_point_list = [np.array([0.3, -0.35, 0.10]), np.array([0.5, -0.25, 0.10]), np.array([0.5, 0.25, 0.10]), np.array([0.3, 0.35, 0.10])]
    way_point_list = [np.array([0.3, -0.35, 0.10]), np.array([0.5, -0.25, 0.10]), np.array([0.5, 0.25, 0.10]), np.array([0.3, 0.35, 0.10])]
    # start_pos= np.array([[0, 0, 1, 0.4],
    #                     [1, 0, 0, -0.35],
    #                     [0, 1, 0, 0.10], 
    #                     [0, 0, 0, 1]])
    
    # end_pos= np.array([[0, 0, 1, 0.4],
    #                     [1, 0, 0, 0.35],
    #                     [0, 1, 0, 0.10], 
    #                     [0, 0, 0, 1]])
    while True:
        pose = get_pose(way_point_list[0])
        sensors["arm"].home_robot(pose)  
        home_start_time = time.time()
        while not sensors["arm"].is_ready():
            time.sleep(0.01)

        chime.info()
        
        # place_id = "1" # input(f"place the object to")
        # if place_id == "-1":
        #     break
        
        # pick_6D = np.eye(4)
        # pick_6D[0, 3] = 0.5
        # pick_6D[1, 3] = 0.3
        # pick_6D[2, 3] = 0.096 - 0.0525
        
        # place_6D = np.array(place_position_list[place_id])
        
        # traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)

        # Prepare execution
        
        length = 100
        # sensors["arm"].home_robot(traj[0])
        # home_start_time = time.time()
        # while not sensors["arm"].is_ready():
        #     if time.time() - home_start_time > 0.5:
        #         chime.warning()
        #         home_start_time = time.time()
        #     time.sleep(0.01)
        
        # chime.success()
        
        for pi in range(len(way_point_list)-1):
            start_pos = get_pose(way_point_list[pi])
            end_pos = get_pose(way_point_list[pi+1])
            
            for i in range(length+1):
                print("start",  pi, i)
                pose = start_pos * (length - i) / length + end_pos * i / length
                pose[:3,:3] = start_pos[:3,:3]
                sensors["arm"].set_action(pose)
                # sensors["hand"].set_target_action(hand_traj[i])
                time.sleep(0.03)  # Simulate time taken for each action
        
        
        sensors["arm"].end()
        # sensors["hand"].end()
        time.sleep(3) # Need distributor to stop
        break
    
    pose = get_pose(way_point_list[-1])
    sensors["arm"].home_robot(pose)
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        time.sleep(0.01)

    chime.info()
    
    for sensor_name, sensor in sensors.items():
        sensor.quit()