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
    controller["hand"] = get_hand(args.hand)
    
    if args.device == "xsens":
        controller["teleop"] = XSensReceiver()
    if args.device == "occulus":
        controller["teleop"] = OculusReceiver()

    controller["camera"] = RemoteCameraController("video", serial_list=None, sync=True)
    controller["signal_generator"] = UTGE900()
    controller["timecode_receiver"] = TimecodeReceiver()
    
    return controller

def move_robot(sensors):
    
    exit_counter = 0
    stop_counter = 0
    
    state_hist = []
    state_time = []
    
    while True:
        start_time = time.time()
        data = sensors["teleop"].get_data()
        if data["Right"] is None:
            continue
        state = state_extractor.get_state(data['Left'])
        state_hist.append(state)
        state_time.append(time.time())
            
        if state == 0:
            wrist_pose, hand_action = retargetor.get_action(data)
            if args.hand is not None:
                sensors["hand"].set_target_action(hand_action)
            
            if args.arm is not None:
                sensors["arm"].set_action(wrist_pose.copy())
                
        if state == 1:
            retargetor.pause()
        
        if state == 2:
            retargetor.pause()
            stop_counter += 1
        
        else:
            stop_counter = 0
            
        if state == 3:
            exit_counter += 1
        
        else:
            exit_counter = 0
            
        if exit_counter > 90:
            return "exit", state_hist, state_time
    
        if stop_counter > 90:
            return "stop", state_hist, state_time
        time.sleep(0.01)
        
# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

c2r = load_latest_C2R()

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/dataset_acquision/hri/video_client.py", pc_list)

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=['xarm', 'franka'])
parser.add_argument("--hand", choices=['inspire', 'allegro'])
parser.add_argument("--device", choices=['xsens', 'occulus'])
parser.add_argument('--obj_name', required=True)
parser.add_argument('--grasp_type', required=True)

args = parser.parse_args()
sensors = initialize_device(args)

state_extractor = HandStateExtractor()
home_pose = sensors["arm"].get_position().copy()

sensors["arm"].home_robot(home_pose)
home_start_time = time.time()
while not sensors["arm"].is_ready():
    time.sleep(0.01)
        
retargetor = Unimanual_Retargetor(args.arm, args.hand, home_pose)

save_path = os.path.join("capture_", "hri_teleop", args.obj_name, args.grasp_type)
shared_path = os.path.join(shared_dir, save_path)
last_capture_idx = -1

if os.path.exists(shared_path) and len(os.listdir(shared_path)) > 0:
    last_capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x)))
else:
    os.makedirs(shared_path, exist_ok=True)

capture_idx = last_capture_idx + 1
while True:
    # prepare for capture, move robot and object
    chime.info()
    time.sleep(2)
    msg, state_hist, state_time = move_robot(sensors)
    if msg == "exit":
        break
    chime.warning()
    time.sleep(2)
    # start
    os.makedirs(f'{shared_path}/{capture_idx}', exist_ok=True)
    copy_calib_files(f'{shared_path}/{capture_idx}')
    np.save(f'{shared_path}/{capture_idx}/C2R.npy', c2r)
        
    sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{args.arm}")
    sensors['hand'].start(f"{shared_path}/{capture_idx}/raw/{args.hand}")
    sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
    sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
    sensors["signal_generator"].on(1)
    
    print("start")
    chime.info()
    time.sleep(2)
    
    msg, state_hist, state_time = move_robot(sensors)
        
    chime.success()
    time.sleep(2)
    
    sensors["arm"].end()
    sensors["hand"].end()
    sensors["camera"].end()
    sensors['timecode_receiver'].end()
    
    os.makedirs(f"{shared_path}/{capture_idx}/raw/state")
    np.save(f"{shared_path}/{capture_idx}/raw/state/state.npy", state_hist)
    np.save(f"{shared_path}/{capture_idx}/raw/state/time.npy", state_time)
    
    sensors['signal_generator'].off(1)
    
    if msg == "exit":
        break
    
    capture_idx += 1


wrist_rot = np.array([[0, 0, 1, 0.3],
                      [1, 0, 0, -0.15],
                      [0, 1, 0, 0.10], 
                      [0, 0, 0, 1]])

sensors["arm"].home_robot(wrist_rot)
home_start_time = time.time()
while not sensors["arm"].is_ready():
    time.sleep(0.01)

chime.info()

for sensor_name, sensor in sensors.items():
    print(sensor_name)
    sensor.quit()