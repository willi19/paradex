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
from paradex.io.capture_pc.ssh import run_script

from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path
from paradex.utils.env import get_pcinfo
from paradex.utils.keyboard_listener import listen_keyboard

from paradex.process.lookup import normalize

lookup_index = "59"
object_name = "pringles"
place_id_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
hand = "allegro"
marker = True

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
    ########## Load camera ###########
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())

    run_script(f"python src/capture/camera/image_client.py", pc_list)
    sensors = {}
    sensors["camera"] = RemoteCameraController("image", None)
    
    try:
        place_position_dict = get_place_position(sensors["camera"], place_id_list)
    except:
        for sensor_name, sensor in sensors.items():
            sensor.quit()
        exit("place not detected, please check the place position and camera")
    ####################################
    
    for sensor_name, sensor in sensors.items():
        sensor.quit()
    
    np.save("pickplace_position.npy", place_position_dict)