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
from paradex.inference.object_6d import get_current_object_6d, get_book_goal_position
from paradex.process.lookup import normalize

from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import run_script

from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path
from paradex.utils.env import get_pcinfo
from paradex.utils.keyboard_listener import listen_keyboard

from paradex.visualization_.renderer import BatchRenderer
from paradex.image.projection import project_mesh_nvdiff
from paradex.image.overlay import overlay_mask
from paradex.image.merge import merge_image

use_sim = True

pose_index_dict = {
    ("stand", "stand"):[4, 6, 9, 11, 19, 20],
    ("lay", "stand"):[13, 14, 16],
    ("lay", "lay"):[1, 2, 3],
    ("stand", "lay"):[17]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--object", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--marker", default=False, action="store_true")

    args = parser.parse_args()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())

    arm_name = "xarm"
    hand_name = args.hand
    
    sensors = {}
    
    run_script(f"python src/capture/camera/image_client.py", pc_list)
    sensors["camera"] = RemoteCameraController("image", None)
    
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.4],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    c2r = load_latest_C2R()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sensors["camera"].start(f"shared_data/inference/register_{timestamp}")
    sensors["camera"].end()
    
    register_dict = {}
    for img_name in os.listdir(f"{shared_dir}/inference/register_{timestamp}"):
        serial_num = img_name.split(".")[0]
        register_dict[serial_num] = cv2.imread(f"{shared_dir}/inference/register_{timestamp}/{img_name}")
    
    goal_marker = [("3", "xz"), ("1", "xz")]#, ("2", "xz"), ("4", "xz")]
    goal_position = get_book_goal_position(register_dict, goal_marker)
    print(goal_position)
    stop_event = Event()
    start_event = Event()
 
    event_dict = {"q":stop_event, "y":start_event}
    listen_keyboard(event_dict)
    
    while not stop_event.is_set():
        place_id = goal_marker[0]
        place_6D = goal_position[place_id]
        place_6D = normalize(place_6D, "book")
        
        print("press y after fixing object position")        
        while not start_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        
        if stop_event.is_set():
            break
        
        start_event.clear()
        
        sensors["camera"].start(os.path.join("shared_data", "inference", "obj_6D", "pick"))
        sensors["camera"].end()
        
        pick_img_dir = os.path.join(shared_dir, "inference", "obj_6D", "pick")       
        os.makedirs(pick_img_dir, exist_ok=True) 
        img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(pick_img_dir, img_name)) for img_name in os.listdir(pick_img_dir)}
        
        pick_6D = get_current_object_6d("book", args.marker, img_dict)
        print(pick_6D, "pick_6D")
        pick_6D = normalize(pick_6D, "book")
        print(pick_6D, "pick_asdf")

        choosen_index, traj, hand_traj = get_traj("book", hand_name, start_pos.copy(), pick_6D.copy(), place_6D.copy(), "1")
        # traj[:, :, :] = traj[0, :, :]
        # Show simulation
        if use_sim:
            print("press y if trajectory ok")
            simulate(traj, hand_traj, pick_6D, place_6D, hand_name, "book", start_event, stop_event)
            
            if stop_event.is_set():
                break
            
            start_event.clear()
        
    for sensor_name, sensor in sensors.items():
        sensor.quit()