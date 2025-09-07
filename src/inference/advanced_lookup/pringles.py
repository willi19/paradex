import time
import numpy as np
from threading import Event
import argparse
import os
from datetime import datetime
import cv2
import json
import chime

from paradex.inference.simulate import simulate
from paradex.inference.lookup_table import get_traj
from paradex.inference.util import home_robot
from paradex.inference.object_6d import get_current_object_6d, get_goal_position

from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import run_script

from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path
from paradex.utils.env import get_pcinfo
from paradex.utils.keyboard_listener import listen_keyboard

from paradex.process.lookup import normalize

use_sim = False
pose_index_dict = {
   ("up", "empty"): [59, 60],
   ("up", "light"): [0, 2, 3, 4],
   ("up", "heavy"): [35, 37,42],
   ("tip", "empty"): [2, 35, 36, 42],
   ("tip", "light"): [14, 16, 18],
   ("tip", "heavy"): [2, 5, 9],
   ("palm", "empty"): [0, 1, 27, 30, 38, 41],
   ("palm", "light"): [],
   ("palm", "heavy"): [39, 40],
   ("tripod", "empty"): [9, 13, 34, 44, 45, 54],
   ("tripod", "light"): [8, 9, 11, 12, 13],
   ("tripod", "heavy"): [15, 22, 27],
   ("lay", "empty"): [7, 10, 17, 24, 29, 32, 47, 48],
   ("lay", "light"): [],
   ("lay", "heavy"): [46, 49, 52]
}

def get_place_position(camera, place_id_list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    camera.start(f"shared_data/inference/register_{timestamp}")
    camera.end()
    
    register_dict = {}
    for img_name in os.listdir(f"{shared_dir}/inference/register_{timestamp}"):
        serial_num = img_name.split(".")[0]
        register_dict[serial_num] = cv2.imread(f"{shared_dir}/inference/register_{timestamp}/{img_name}")
    
    place_position_dict = get_goal_position(register_dict, place_id_list)
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
    parser.add_argument("--object", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--marker", default=False, action="store_true")

    args = parser.parse_args()
    
    ########## Load camera ###########
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())

    run_script(f"python src/capture/camera/image_client.py", pc_list)
    sensors = {}
    sensors["camera"] = RemoteCameraController("image", None)
    
    place_id_list = ["1", "4"]
    try:
        place_position_dict = get_place_position(sensors["camera"], place_id_list)
    except:
        for sensor_name, sensor in sensors.items():
            sensor.quit()
        exit("place not detected, please check the place position and camera")
        
    
    ####################################
    
    ########## Load robot #################
    arm_name = "xarm"
    hand_name = args.hand
    sensors["arm"] = get_arm(arm_name)
    sensors["hand"] = get_hand(hand_name)
    
    
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.3],
                        [0, 1, 0, 0.4], 
                        [0, 0, 0, 1]])
    c2r = load_latest_C2R()
    #####################################
    
    ########### Prepare capture ###############
    num_try = 20
    
    stop_event = Event()
    start_event = Event()
    
    event_dict = {"q":stop_event, "y":start_event}
    listen_keyboard(event_dict)
    ####################################
    
    grasp_obj_type = "empty" if args.object == "pringles" else args.object.split("_")[1]
    for grasp_type in ["up", "tip", "palm", "tripod"]:
        pose_index_list = pose_index_dict[(grasp_type, grasp_obj_type)]
        for pose_index in pose_index_list:
            for i in range(num_try):
                info = {
                    "grasp_type":grasp_type,
                    "pose_index":pose_index,
                    "index":i,
                    "success":True
                    }
                
                cur_id = place_id_list[(i+len(place_id_list)-1) % len(place_id_list)]
                place_id = place_id_list[i % len(place_id_list)]
                place_6D = place_position_dict[place_id]
                cur_6D = place_position_dict[cur_id]
                
                save_path = os.path.join("inference", "accuracy_test", args.object, grasp_type, str(pose_index), str(i))
                os.makedirs(os.path.join(shared_dir, save_path), exist_ok=True)
                
                if os.path.exists(os.path.join(shared_dir, save_path, "result.json")):
                    print(grasp_type, grasp_obj_type, str(pose_index), str(i))
                    continue
                    
                if stop_event.is_set():
                    break
                
                start_event.clear()
                ######### Prepare capture ###############
                sensors["hand"].home_robot()
                home_robot(sensors["arm"], start_pos)

                ##### Load pick 6D ######
                while not stop_event.is_set():
                    
                    pick_path = os.path.join(save_path, "pick")
                    pick_6D = get_object_6D(args.object, args.marker, sensors["camera"], pick_path)
                    if np.linalg.norm(pick_6D[:3, 3] - cur_6D[:3, 3]) < 0.05 and pick_6D[2, 2] > 0.8:
                        break 
                                     
                    print(f"press y after fixing object position to {cur_id}")        
                    chime.warning()
                    while not start_event.is_set() and not stop_event.is_set():
                        time.sleep(0.05)
                
                if stop_event.is_set():
                    break
                
                start_event.clear()
                
                #########################################
                
                ############    Load & Save traj   #############
                
                _, traj, hand_traj, state = get_traj(args.object, hand_name, start_pos.copy(), pick_6D.copy(), place_6D.copy(), index=str(pose_index))
                if use_sim:
                    print("press y if trajectory ok")
                    simulate(traj, hand_traj, pick_6D, place_6D, hand_name, "pringles", start_event, stop_event)
                    
                    if stop_event.is_set():
                        break
                    
                    start_event.clear()
                    
                copy_calib_files(os.path.join(shared_dir, save_path))
                np.save(os.path.join(shared_dir, save_path, 'C2R.npy'), c2r)
                np.save(os.path.join(shared_dir, save_path, 'pick_6D.npy'), pick_6D)
                np.save(os.path.join(shared_dir, save_path, 'target_6D.npy'), place_6D)
                np.save(os.path.join(shared_dir, save_path, 'traj.npy'), traj)
                np.save(os.path.join(shared_dir, save_path, 'hand_traj.npy'), hand_traj)
                
                ####################################################
                
                ################ Execute ###########################
                sensors['arm'].start(os.path.join(shared_dir, save_path, "raw", arm_name))
                sensors['hand'].start(os.path.join(shared_dir, save_path, "raw", hand_name))
                
                for t in range(len(traj)):
                    if t != 0 and state[t] == 4 and state[t-1] != 4:
                        sensors["camera"].start(os.path.join("shared_data", save_path, "place"))
                        sensors["camera"].end()
                        
                    sensors["arm"].set_action(traj[t])
                    sensors["hand"].set_target_action(hand_traj[t])
                    time.sleep(0.03)  # Simulate time taken for each action
                    
                    
                    if stop_event.is_set():
                        info["success"] = False
                        break
                
                sensors["arm"].end()
                sensors["hand"].end()
                ###################################
                
                ############## Save result ##################
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