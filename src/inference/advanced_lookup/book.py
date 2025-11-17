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

from paradex.io.camera.util import get_image
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.ssh import run_script

from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path
from paradex.utils.env import get_pcinfo
from paradex.utils.keyboard_listener import listen_keyboard

from paradex.visualization_.renderer import BatchRenderer
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.image.overlay import overlay_mask
from paradex.image.merge import merge_image

use_sim = False

pose_index_dict = {
    ("stand", "stand"):[4, 6, 9, 11, 19, 20],
    ("lay", "stand"):[13, 14, 16],
    ("lay", "lay"):[1, 2, 3],
    ("stand", "lay"):[17]
}

def get_place_position(camera, place_id_list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    camera.start(f"shared_data/inference/register_{timestamp}")
    camera.end()
    
    register_dict = {}
    for img_name in os.listdir(f"{shared_dir}/inference/register_{timestamp}"):
        serial_num = img_name.split(".")[0]
        register_dict[serial_num] = cv2.imread(f"{shared_dir}/inference/register_{timestamp}/{img_name}")
    
    place_position_dict = get_book_goal_position(register_dict, place_id_list)
    return place_position_dict

def get_object_6D(obj_name, marker, camera, path):
    img_dir = os.path.join(shared_dir, path)       
    os.makedirs(img_dir, exist_ok=True) 
    if camera is not None:
        camera.start(os.path.join("shared_data", path))
        camera.end()
    
    img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}
    
    obj_6D = get_current_object_6d(obj_name, marker, img_dict)
    obj_6D = normalize(obj_6D, obj_name)
    return obj_6D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--object", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--marker", default=False, action="store_true")

    args = parser.parse_args()
    
    ########## Load device ###########
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
                        [0, 1, 0, 0.35], 
                        [0, 0, 0, 1]])
    
    ####################################
    
    ########## Load Info #################
    stop_event = Event()
    start_event = Event()
 
    event_dict = {"q":stop_event, "y":start_event}
    listen_keyboard(event_dict)
    num_try = 2
    
    c2r = load_latest_C2R()
    place_id_list = [("1", "xz"), ("2", "xz"), ("3", "xz"), ("4", "xz")]
    try:    
        place_position_dict = get_place_position(sensors["camera"], place_id_list)
    except:
        stop_event.set()
    
    ##########################################
    process_list = []        
    for start_state in ["lay"]:
        for end_state in ["lay"]:
            for table_idx in pose_index_dict[(start_state, end_state)]:
                for place_id in place_id_list:
                    place_6D = place_position_dict[place_id]
                    place_6D = normalize(place_6D, "book")
                    for pick_id in place_id_list:
                        for demo_idx in range(num_try):
                            process_info = {"start_state": start_state,
                                            "end_state": end_state,
                                            "place_id": place_id,
                                            "pick_id": pick_id,
                                            "demo_idx": demo_idx,
                                            "place_6D": place_6D,
                                            "table_idx":table_idx}
                            process_list.append(process_info)
    
    for process_info in process_list:
        info = {"start_state": process_info["start_state"],
                "end_state": process_info["end_state"],
                "place_id": process_info["place_id"],
                "pick_id": process_info["pick_id"],
                "demo_idx": process_info["demo_idx"],
                "table_idx":process_info["table_idx"]}

        info["success"] = True
        
        save_path = os.path.join("inference", "lookup", "book", info["start_state"]+"_"+info["end_state"], str(info["table_idx"]), info["pick_id"][0] + "_" + info["place_id"][0], str(info["demo_idx"]))
        
        home_robot(sensors["arm"], start_pos)
        
        if os.path.exists(os.path.join(shared_dir, save_path, "result.json")):
            continue
        
        if stop_event.is_set():
            break
                    
        print(f"press y after fixing object position to {process_info['pick_id']} will go to {process_info['place_id']}")        
        while not start_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        
        if stop_event.is_set():
            break
        
        start_event.clear()

        ##### Load pick 6D ######
        pick_path = os.path.join(save_path, "pick")
        pick_6D = get_object_6D("book", args.marker, sensors["camera"], pick_path)
        place_6D = process_info['place_6D'].copy()
        ##########################################
                    
        ############    Load & Save traj   #############
        choosen_index, traj, hand_traj = get_traj("book", hand_name, start_pos.copy(), pick_6D.copy(), place_6D.copy(), str(process_info["table_idx"]))
        
        # Show simulation
        if use_sim:
            print("press y if trajectory ok")
            simulate(traj, hand_traj, pick_6D, place_6D, hand_name, "book", start_event, stop_event)
            
            if stop_event.is_set():
                break
            
            start_event.clear()
        
        # start the camera
        copy_calib_files(save_path)
        np.save(f'{shared_dir}/{save_path}/C2R.npy', c2r)
        np.save(f'{shared_dir}/{save_path}/pick_6D.npy', pick_6D)
        np.save(f'{shared_dir}/{save_path}/target_6D.npy', place_6D)
        np.save(f'{shared_dir}/{save_path}/traj.npy', traj)
        np.save(f'{shared_dir}/{save_path}/hand_traj.npy', hand_traj)
        ####################################################
                    
        ################ Execute ###########################
        sensors['arm'].start(f"{shared_dir}/{save_path}/raw/{arm_name}")
        sensors['hand'].start(f"{shared_dir}/{save_path}/raw/{hand_name}")
        
        for i in range(len(traj)):
            if i == len(traj) - 70:
                sensors["camera"].start(os.path.join("shared_data", save_path, "place"))
                sensors["camera"].end()
                
            sensors["arm"].set_action(traj[i])
            sensors["hand"].set_target_action(hand_traj[i])
            time.sleep(0.03)  # Simulate time taken for each action
            
            
            if stop_event.is_set():
                break
        
        sensors["arm"].end()
        sensors["hand"].end()
        
        ############## Save result ##################
        if not stop_event.is_set():
            place_path = os.path.join(save_path, "place")
            try:
                cur_6D = get_object_6D(args.object, args.marker, None, place_path)
                np.save(os.path.join(shared_dir, save_path, 'place_6D.npy', cur_6D))
            except:
                print("no place detected")
        with open(os.path.join(shared_dir, save_path, "result.json"), "w") as f:
            json.dump(info, f, indent=4)
        #############################################
        
        
    for sensor_name, sensor in sensors.items():
        sensor.quit()