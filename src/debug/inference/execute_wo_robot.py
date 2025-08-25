import time
import numpy as np
import chime
import argparse
import json
import os

from paradex.inference.lookup_table import get_traj
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.inference.object_6d import get_current_object_6d
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list

if __name__ == "__main__":
    sensors = {}
    sensors["signal_generator"] = UTGE900()
    sensors["timecode_receiver"] = TimecodeReceiver()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    git_pull("merging", pc_list)
    
    for i in range(5):
        pick_6D = get_current_object_6d("pringles")
        # start the camera
        run_script(f"python src/capture/camera/video_client.py", pc_list)
        sensors["camera"] = RemoteCameraController("video", serial_list=None, sync=True)

        # Set directory
        save_path = os.path.join("debug_", "inference_wo_robot")
        shared_path = os.path.join(shared_dir, save_path)
        
        os.makedirs(shared_path, exist_ok=True)
        if len(os.listdir(shared_path)) == 0:
            capture_idx = 0
        else:
            capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x))) + 1

        time.sleep(2)
        
        # Start capture
        sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
        sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
        sensors["signal_generator"].on(1)
        
        time.sleep(20)
        
        sensors["camera"].end()
        sensors['timecode_receiver'].end()
        sensors['signal_generator'].off(1)
        
        sensors["camera"].quit()
    
    for sensor_name, sensor in sensors.items():
        if sensor_name == "camera":
            continue
        sensor.quit()