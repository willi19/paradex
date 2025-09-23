from threading import Event, Thread
import time
import argparse
import os
import chime
import numpy as np

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.utils.keyboard_listener import listen_keyboard

from paradex.io.robot_controller import get_arm, get_hand # XArmController, AllegroController, InspireController# , FrankaController
from paradex.io.teleop import XSensReceiver, OculusReceiver
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver

from paradex.retargetor import Unimanual_Retargetor, HandStateExtractor
from paradex.geometry.coordinate import DEVICE2WRIST

from paradex.inference.util import home_robot

def initialize_device(args):
    controller = {}
    
    controller["hand"] = get_hand("inspire")
    
    if args.device == "xsens":
        controller["teleop"] = XSensReceiver()
    if args.device == "occulus":
        controller["teleop"] = OculusReceiver()

    controller["camera"] = RemoteCameraController("image", serial_list=None, sync=False, debug=True)
    return controller

# === SETUP ===
parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=['xsens', 'occulus'])
parser.add_argument('--obj_name', required=True)

args = parser.parse_args()

pc_info = get_pcinfo()
serial_list = get_serial_list()
pc_list = list(pc_info.keys())

run_script(f"python src/capture/camera/image_client.py", pc_list)

c2r = load_latest_C2R()

sensors = initialize_device(args)
state_extractor = HandStateExtractor()

arm_controller = get_arm("xarm")
home_pose = arm_controller.get_position().copy()
arm_controller.quit(set_break=False)
retargetor = Unimanual_Retargetor("xarm", "inspire", home_pose)

save_path = os.path.join("shared_data", "capture", "rlworld_grasp", args.obj_name)
shared_path = os.path.join(shared_dir, save_path)
last_capture_idx = -1

if os.path.exists(shared_path) and len(os.listdir(shared_path)) > 0:
    last_capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x)))
else:
    os.makedirs(shared_path, exist_ok=True)

capture_idx = last_capture_idx + 1

stop_event = Event()
start_event = Event()
listen_keyboard({"q": stop_event, "y": start_event})

while not stop_event.is_set():
    # prepare for capture, move robot and object
    # Use Ufactory to move robot to desired position
    home_pose = np.zeros(6)+1000
    home_pose[5] = 0
    
    sensors["hand"].home_robot(home_pose)
    chime.info()
    time.sleep(0.5)
       
    while not start_event.is_set() and not stop_event.is_set():
        time.sleep(0.1)
    start_event.clear()
    if stop_event.is_set():
        break

    # #####################################################

    # Save data before capture
    os.makedirs(f'{shared_path}/{capture_idx}', exist_ok=True)
    copy_calib_files(f'{shared_path}/{capture_idx}')
    np.save(f'{shared_path}/{capture_idx}/C2R.npy', c2r)
    
    
    os.makedirs(f'{shared_path}/{capture_idx}/grasp/start/image', exist_ok=True)
    os.makedirs(f'{shared_path}/{capture_idx}/grasp/end/image', exist_ok=True)

    arm_controller = get_arm("xarm")
    wristSe3 = arm_controller.get_position().copy()
    qpos = arm_controller.get_qpos().copy()
    np.save(f'{shared_path}/{capture_idx}/grasp/wristSe3.npy', wristSe3)
    np.save(f'{shared_path}/{capture_idx}/grasp/qpos.npy', qpos)

    sensors['camera'].start(f"{save_path}/{capture_idx}/grasp/start/image")
    sensors['camera'].end()

    sensors['hand'].start(f"{shared_path}/{capture_idx}/grasp/inspire")
    #####################################################
    
    # Start grasping
    chime.info()
    time.sleep(0.5)
    
    exit_counter = 0
    stop_counter = 0

    while not stop_event.is_set():
        data = sensors["teleop"].get_data()
        if data["Right"] is None:
            continue
        state = state_extractor.get_state(data['Left'])
        
        if state == 0:
            _, hand_action = retargetor.get_action(data)
            sensors["hand"].set_target_action(hand_action)
                
        if state == 1:
            retargetor.pause()
        
        if state == 2:
            retargetor.pause()
            stop_counter += 1
        
        else:
            stop_counter = 0
            
        if stop_counter > 90:
            msg =  "stop"
            break
        
        time.sleep(0.01)
        
    chime.success()
    time.sleep(0.5)
    
    sensors["hand"].end()
    sensors['camera'].start(f"{save_path}/{capture_idx}/grasp/end/image")
    sensors['camera'].end()
    
    # Stability test
    print("lift start")
    os.makedirs(f'{shared_path}/{capture_idx}/lift/image', exist_ok=True)
    arm_controller = get_arm("xarm")
    cur_pose = arm_controller.get_position().copy()
    for i in range(90):
        next_pose = cur_pose.copy()
        next_pose[2, 3] += 0.001 * (i+1)
        
        arm_controller.set_action(next_pose)
        time.sleep(0.03)
    
    lift_qpos = arm_controller.get_qpos().copy()
    lift_pose = arm_controller.get_position().copy()
    lift_hand_pose = sensors['hand'].get_qpos().copy()
    
    np.save(f'{shared_path}/{capture_idx}/lift/qpos.npy', lift_qpos)
    np.save(f'{shared_path}/{capture_idx}/lift/wristSe3.npy', lift_pose)
    np.save(f'{shared_path}/{capture_idx}/lift/hand_qpos.npy', lift_hand_pose)
    
    sensors['camera'].start(f"{save_path}/{capture_idx}/lift/image")
    sensors['camera'].end()
    
    
    os.makedirs(f'{shared_path}/{capture_idx}/lay/image', exist_ok=True)
    for i in range(90):
        next_pose = cur_pose.copy()
        next_pose[2, 3] += 0.001 * (90 - i - 1)
        
        arm_controller.set_action(next_pose)
        time.sleep(0.03)
        
    lay_qpos = arm_controller.get_qpos().copy()
    lay_pose = arm_controller.get_position().copy()
    lay_hand_pose = sensors['hand'].get_qpos().copy()
    np.save(f'{shared_path}/{capture_idx}/lay/qpos.npy', lay_qpos)
    np.save(f'{shared_path}/{capture_idx}/lay/wristSe3.npy', lay_pose)
    np.save(f'{shared_path}/{capture_idx}/lay/hand_qpos.npy', lay_hand_pose)

    sensors['camera'].start(f"{save_path}/{capture_idx}/lay/image")
    sensors['camera'].end()

    arm_controller.quit(set_break=False)
    # Release
    print("release start")
    os.makedirs(f'{shared_path}/{capture_idx}/release/image', exist_ok=True)
    os.makedirs(f'{shared_path}/{capture_idx}/release/inspire', exist_ok=True)
    
    sensors["hand"].start(f"{shared_path}/{capture_idx}/release/inspire")    
    hand_qpos = np.load(f'{shared_path}/{capture_idx}/grasp/inspire/position.npy')
    for i in range(len(hand_qpos)-1, -1, -1):
        sensors["hand"].set_target_action(hand_qpos[i])
        time.sleep(0.01)
    sensors["hand"].end()
    sensors['camera'].start(f"{save_path}/{capture_idx}/release/image")
    sensors['camera'].end()
    ###################
    
    capture_idx += 1

for sensor_name, sensor in sensors.items():
    sensor.quit()