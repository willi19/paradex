import time
import numpy as np
from threading import Event
import argparse
import json
import os
from datetime import datetime
import cv2
import trimesh
import copy

from paradex.inference.simulate import simulate
from paradex.inference.lookup_table import get_traj
from paradex.inference.util import home_robot
from paradex.inference.object_6d import get_current_object_6d, get_goal_position

from paradex.io.camera.util import get_image
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import run_script

from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path
from paradex.utils.env import get_pcinfo
from paradex.utils.keyboard_listener import listen_keyboard

from paradex.visualization_.renderer import BatchRenderer
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.image.overlay import overlay_mask
from paradex.image.merge import merge_image

from paradex.process.lookup import normalize_cylinder

use_sim = True
pose_index_dict = {
   ("up"): [59, 60],
   ("tip"): [2, 35, 36, 42],
   ("palm"): [0, 1, 27, 30, 38, 41],
   ("tripod"): [9, 13, 34, 44, 45, 54],
   ("lay"): [7, 10, 17, 24, 29, 32, 47, 48]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--marker", default=False, action="store_true")

    args = parser.parse_args()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())

    arm_name = "xarm"
    hand_name = args.hand
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    sensors["hand"] = get_hand(hand_name)
    
    run_script(f"python src/capture/camera/image_client.py", pc_list)
    sensors["camera"] = RemoteCameraController("image", None)
    
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.4],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]]) 
    
    end_pos= np.array([[0, 0, 1, 0.25],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    c2r = load_latest_C2R()
    intrinsic, extrinsic = load_current_camparam()
    
    serial_list = list(extrinsic.keys())
    serial_list.sort()
    
    save_path = os.path.join("inference_rlworld", "lookup", args.object)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    place_id_list = ["1"]
    
    # Detect the placing position
    sensors["camera"].start(f"shared_data/inference/register_{timestamp}")
    sensors["camera"].end()
    
    register_dict = {}
    for img_name in os.listdir(f"{shared_dir}/inference/register_{timestamp}"):
        serial_num = img_name.split(".")[0]
        register_dict[serial_num] = cv2.imread(f"{shared_dir}/inference/register_{timestamp}/{img_name}")
    
    place_position_dict = get_goal_position(register_dict, place_id_list)
    ################################################
    
    shared_path = os.path.join(shared_dir, save_path)
    os.makedirs(shared_path, exist_ok=True)
    if len(os.listdir(shared_path)) == 0:
        capture_idx = 0
    else:
        capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x))) + 1
    
    stop_event = Event()
    start_event = Event()
 
    event_dict = {"q":stop_event, "y":start_event}
    listen_keyboard(event_dict)
    
    while not stop_event.is_set():
        place_id = place_id_list[int(capture_idx) % len(place_id_list)]
        place_6D = place_position_dict[place_id]
        
        home_robot(sensors["arm"], start_pos)

        print("press y after fixing object position")        
        while not start_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        
        if stop_event.is_set():
            break
        
        start_event.clear()
        
        # Capture the Object and get the 6D pose
        sensors["camera"].start(os.path.join("shared_data", save_path, str(capture_idx), "pick"))
        sensors["camera"].end()
        
        pick_img_dir = os.path.join(shared_path, str(capture_idx), "pick")       
        os.makedirs(pick_img_dir, exist_ok=True) 
        img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(pick_img_dir, img_name)) for img_name in os.listdir(pick_img_dir)}
        
        pick_6D = get_current_object_6d(args.object, args.marker, img_dict)
        pick_6D = normalize_cylinder(pick_6D)
        ################################################
        
        while True:
            grasp_type = input("input the grasp type (up, tip, palm, tripod, lay): ")
            if grasp_type in ["up", "tip", "palm", "tripod", "lay"]:
                index = str(np.random.choice(pose_index_dict[(grasp_type)]))
                break
            else:
                continue
        
        choosen_index, traj, hand_traj, state = get_traj(args.object, hand_name, start_pos.copy(), pick_6D.copy(), place_6D.copy(), index)
        
        # Show simulation
        if use_sim:
            print("press y if trajectory ok")
            simulate(traj, hand_traj, pick_6D, place_6D, hand_name, args.object, start_event, stop_event)
            
            if stop_event.is_set():
                break
            
            start_event.clear()
        break
        # Start capture
        sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{arm_name}")
        sensors['hand'].start(f"{shared_path}/{capture_idx}/raw/{hand_name}")
        
        for i in range(len(traj)):
            if i == len(traj) - 100:
                sensors["camera"].start(os.path.join("shared_data", save_path, str(capture_idx), "place"))
                sensors["camera"].end()
                
            sensors["arm"].set_action(traj[i])
            sensors["hand"].set_target_action(hand_traj[i])
            time.sleep(0.03)  # Simulate time taken for each action
            
            
            if stop_event.is_set():
                break
        
        sensors["arm"].end()
        sensors["hand"].end()

        capture_idx += 1
        
    home_robot(sensors["arm"], end_pos.copy())
    for sensor_name, sensor in sensors.items():
        sensor.quit()