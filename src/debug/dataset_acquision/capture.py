from threading import Event, Thread
import time
import argparse
import os
import chime

chime.theme('pokemon')

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R

from paradex.io.robot_controller import get_arm, get_hand # XArmController, AllegroController, InspireController# , FrankaController
from paradex.io.teleop import XSensReceiver, OculusReceiver
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver

from paradex.retargetor import Unimanual_Retargetor, HandStateExtractor
from paradex.geometry.coordinate import DEVICE2WRIST

import numpy as np

def initialize_device(args):
    controller = {}
    
    controller["arm"] = get_arm(args.arm)
    # controller["hand"] = get_hand(args.hand)
    
    controller["camera"] = RemoteCameraController("video", serial_list=None, sync=True)
    controller["signal_generator"] = UTGE900()
    controller["timecode_receiver"] = TimecodeReceiver()
    
    return controller

wrist_rot = np.array([[0, 0, 1, 0.3],
                      [1, 0, 0, -0.15],
                      [0, 1, 0, 0.10], 
                      [0, 0, 0, 1]])

def move_robot(sensors):    
    start_time = time.time()
    
    while time.time() - start_time < 60:
        
        # hand_action = np.ones(12) * 1000
        wrist_pose = wrist_rot.copy()
        if time.time() - start_time < 15:
            wrist_pose[1, 3]  = -0.15 + (time.time() - start_time) / 50
        if time.time() - start_time > 15 and time.time() - start_time < 30:
            wrist_pose[1, 3]  = 0.15
            wrist_pose[0, 3]  = 0.3 + (time.time() - start_time-15) / 100
            
        if time.time() - start_time > 30 and time.time() - start_time < 45:
            wrist_pose[0, 3]  = 0.45
            wrist_pose[1, 3]  = 0.15 - (time.time() - start_time-30) / 50
            
        if time.time() - start_time > 45 and time.time() - start_time < 60:
            wrist_pose[1, 3]  = -0.15
            wrist_pose[0, 3]  = 0.45 - (time.time() - start_time-45) / 100
        # if args.hand is not None:
        #     sensors["hand"].set_target_action(hand_action)
            
        if args.arm is not None:
            sensors["arm"].set_action(wrist_pose.copy())
        
        time.sleep(0.01)
        
# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

c2r = load_latest_C2R()

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/dataset_acquision/lookup/video_client.py", pc_list)

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=['xarm', 'franka'])
# parser.add_argument("--hand", choices=['inspire', 'allegro'])

args = parser.parse_args()
sensors = initialize_device(args)

home_pose = wrist_rot.copy()# sensors["arm"].get_position().copy()
# home_pose[2, 3] += 0.1
sensors["arm"].home_robot(home_pose)
home_start_time = time.time()
while not sensors["arm"].is_ready():
    time.sleep(0.01)

save_path = os.path.join("capture_", "debug")
shared_path = os.path.join(shared_dir, save_path)
last_capture_idx = -1

if os.path.exists(shared_path) and len(os.listdir(shared_path)) > 0:
    last_capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x)))
else:
    os.makedirs(shared_path, exist_ok=True)

capture_idx = last_capture_idx + 1
while capture_idx < 1:
    # prepare for capture, move robot and object
    chime.info()
    # move_robot(sensors)
    time.sleep(2)
    
    chime.warning()
    # start
    os.makedirs(f'{shared_path}/{capture_idx}', exist_ok=True)
    copy_calib_files(f'{shared_path}/{capture_idx}')
    np.save(f'{shared_path}/{capture_idx}/C2R.npy', c2r)
        
    sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{args.arm}")
    sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
    sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
    sensors["signal_generator"].on(1)
    
    print("start")
    chime.info()
    move_robot(sensors)
        
    sensors["arm"].end()
    sensors["camera"].end()
    sensors['timecode_receiver'].end()
    
    sensors['signal_generator'].off(1)
    chime.success()
    
    capture_idx += 1

for sensor_name, sensor in sensors.items():
    print(sensor_name)
    sensor.quit()