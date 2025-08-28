import time
import numpy as np
import chime
import argparse
import json
import os

from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController

from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.utils.env import get_pcinfo
from paradex.io.capture_pc.connect import run_script
from paradex.inference.util import get_linear_path, home_robot

import random

if __name__ == "__main__":
    arm_name = "xarm"
    hand_name = "allegro"
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    sensors["hand"] = get_hand(hand_name)
    
    sensors["signal_generator"] = UTGE900()
    sensors["timecode_receiver"] = TimecodeReceiver()
    
    run_script(f"python src/capture/camera/video_client.py", pc_list)
    sensors["camera"] = RemoteCameraController("video", serial_list=None, sync=True)
    
    for _ in range(3):  # Repeat the process 3 times
        save_path = os.path.join("debug_", "inference")
        shared_path = os.path.join(shared_dir, save_path)
        os.makedirs(shared_path, exist_ok=True)
        if len(os.listdir(shared_path)) == 0:
            capture_idx = 0
        else:
            capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x))) + 1

        c2r = load_latest_C2R()
        os.makedirs(os.path.join(shared_path, str(capture_idx)))
        copy_calib_files(f'{shared_path}/{capture_idx}')
        np.save(f'{shared_path}/{capture_idx}/C2R.npy', c2r)
        
        # Set directory
        # Start capture
        sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{arm_name}")
        sensors['hand'].start(f"{shared_path}/{capture_idx}/raw/{hand_name}")
        
        sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
        sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
        sensors["signal_generator"].on(1)
        
        state_hist = []
        state_time = []
        
        last_action = np.load(f"data/eef/0_qpos.npy")
        last_hand_action = np.load(f"data/eef/0_handqpos.npy")
        
        for i in range(1, 30):
            target_action = np.load(f"data/eef/{i}_qpos.npy")
            target_hand_action = np.load(f"data/eef/{i}_handqpos.npy")
            
            target_action[0,3] += random.random() / 4
            target_action[1,3] += (random.random() - 0.5) / 1.5
            
            if i == 1:
                home_robot(sensors["arm"], last_action)
                
            length = 70
            traj, hand_traj = get_linear_path(last_action, target_action, last_hand_action, target_hand_action, length)
            
            for j in range(length):
                sensors["arm"].set_action(traj[j])
                sensors["hand"].set_target_action(hand_traj[j])
                time.sleep(0.03) 

            time.sleep(0.5)
            
            last_action = target_action
            last_hand_action = target_hand_action
            
        sensors["arm"].end()
        sensors["hand"].end()
        sensors["camera"].end()
        sensors['timecode_receiver'].end()
        sensors['signal_generator'].off(1)
        time.sleep(2)
        
        os.makedirs(f"{shared_path}/{capture_idx}/raw/state", exist_ok=True)
        np.save(f"{shared_path}/{capture_idx}/raw/state/state.npy", state_hist)
        np.save(f"{shared_path}/{capture_idx}/raw/state/time.npy", state_time)
    
    for sensor_name, sensor in sensors.items():
        sensor.quit()