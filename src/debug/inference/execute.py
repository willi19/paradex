import time
import numpy as np
import chime
import argparse
import json
import os

from paradex.inference.get_lookup_traj import get_traj
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.inference.object_6d import get_current_object_6d
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list

if __name__ == "__main__":
    arm_name = "xarm"
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    sensors["signal_generator"] = UTGE900()
    sensors["timecode_receiver"] = TimecodeReceiver()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    git_pull("merging", pc_list)
    
    start_pos= np.array([[0, 0, 1, 0.35],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    end_pos= np.array([[0, 0, 1, 0.35],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    for _ in range(3):  # Repeat the process 3 times
        sensors["arm"].home_robot(start_pos.copy())  
        home_start_time = time.time()
        while not sensors["arm"].is_ready():
            time.sleep(0.01)

        chime.info()
        
        run_script(f"python src/capture/camera/video_client.py", pc_list)
        sensors["camera"] = RemoteCameraController("video", serial_list=None, sync=True)

        # Set directory
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
        
        # Start capture
        sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{arm_name}")
        sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
        sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
        sensors["signal_generator"].on(1)
        
        state_hist = []
        state_time = []
        for i in range(150):
            target_pose = start_pos.copy()
            target_pose[1, 3] += min(i, 100 - i) * 0.005 if i < 100 else (i - 100) * 0.005
            sensors["arm"].set_action(target_pose)
            time.sleep(0.03) 
        
        
        sensors["arm"].end()
        sensors["camera"].end()
        sensors['timecode_receiver'].end()
        sensors['signal_generator'].off(1)
        time.sleep(2)
        
        os.makedirs(f"{shared_path}/{capture_idx}/raw/state", exist_ok=True)
        np.save(f"{shared_path}/{capture_idx}/raw/state/state.npy", state_hist)
        np.save(f"{shared_path}/{capture_idx}/raw/state/time.npy", state_time)
        sensors["camera"].quit()
    
    sensors["arm"].home_robot(end_pos)
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        time.sleep(0.01)

    chime.info()
    
    for sensor_name, sensor in sensors.items():
        if sensor_name == "camera":
            continue
        sensor.quit()