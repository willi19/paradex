import time
import numpy as np
import chime
import argparse
import json
import os, sys
from threading import Event

from paradex.inference.simulate import simulate_temp
# from paradex.inference.lookup_table import get_traj
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.inference.object_6d import get_current_object_6d, get_image
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_latest_eef
from paradex.io.capture_pc.ssh import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.pose_utils.optimize_initial_frame import object6d_silhouette
from paradex.pose_utils.retarget_utils import get_keypoint_trajectory, visualize_new_trajectory
from paradex.pose_utils.retarget import position_retarget, qpose_dict_to_traj, wrist6d_traj_to_SE3, play_local_frames
from paradex.utils.keyboard_listener import listen_keyboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--obj_name", required=True)
    parser.add_argument("--vis", action ='store_true')
    parser.add_argument("--no_rot", action="store_true")
    parser.add_argument("--wrist_up", action="store_true")

    # parser.add_argument("--grasp_type", required=True)

    args = parser.parse_args()
    
    path_planning = [(1, 2), (2, 4), (4, 2), (2, 3), (3, 1), (1, 3), (3, 4), (4, 1), (1, 4), (4, 3), (3, 2), (2, 1)]
    
    arm_name = "xarm"
    hand_name = "allegro"
    
    sensors = {}
    # sensors["arm"] = get_arm(arm_name)
    # sensors["hand"] = get_hand(hand_name)
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    git_pull("merging", pc_list)
    
    scene_idx = args.index
    scene_path = os.path.join(shared_dir, "capture_", "hri", args.obj_name, str(scene_idx))

    # get_image()
    start_6d = get_current_object_6d(obj_name=args.obj_name, marker=True)
    
    print(start_6d) # camera space

    # robot space
    hand_trajectory_dict, obj_trajectory_dict = get_keypoint_trajectory(scene_path, start_6d, args.obj_name, no_rot=args.no_rot, wrist_up=args.wrist_up)
    with open("/home/temp_id/paradex/qpose.json", "r") as f:
        q_pose_dict = json.load(f)#position_retarget(hand_trajectory_dict)
    
    q_pose_dict = {int(k): v for k, v in q_pose_dict.items()}
    print(q_pose_dict)
    # q_pose_dict = position_retarget(hand_trajectory_dict)

    # if args.vis:
    #     visualize_new_trajectory(args.obj_name, hand_trajectory_dict, obj_trajectory_dict, q_pose_dict)
    
    
    
    # pick_traj = np.load(f"{demo_path}/pick.npy")
    # place_traj = np.load(f"{demo_path}/place.npy")
    # pick_hand_traj = np.load(f"{demo_path}/pick_hand.npy")
    # place_hand_traj = np.load(f"{demo_path}/place_hand.npy")
    
    # place_position_list = json.load(open(f"data/lookup/{args.obj_name}/obj_pose.json"))
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.35],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    end_pos= np.array([[0, 0, 1, 0.25],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    stop_event = Event()
    start_event = Event()
 
    event_dict = {"q":stop_event, "y":start_event}
    listen_keyboard(event_dict)
    
    while True:
        # sensors["arm"].home_robot(start_pos.copy())  
        # home_start_time = time.time()
        # while not sensors["arm"].is_ready():
        #     time.sleep(0.01)

        # chime.info()
        
        # traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)
        
        eef = load_latest_eef()

        traj, hand_traj = qpose_dict_to_traj(q_pose_dict)
        traj = wrist6d_traj_to_SE3(traj)
        
        for i in range(len(traj)):
            traj[i] = traj[i] @ np.linalg.inv(eef)
        play_local_frames(traj)
        # simulate_temp(traj, hand_traj, start_6d, start_6d, hand_name, args.obj_name, start_event, stop_event)
        
    
    sensors["arm"].home_robot(end_pos)
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        time.sleep(0.01)

    chime.info()
    
    for sensor_name, sensor in sensors.items():
        if sensor_name == "camera":
            continue
        sensor.quit()
        
        
