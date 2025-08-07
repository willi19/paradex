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

from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver

from paradex.retargetor import Unimanual_Retargetor, HandStateExtractor
from paradex.geometry.coordinate import DEVICE2WRIST

import numpy as np

def initialize_device():
    controller = {}
    
    controller["camera"] = RemoteCameraController("video", serial_list=None, sync=True)
    controller["signal_generator"] = UTGE900()
    controller["timecode_receiver"] = TimecodeReceiver()
    
    return controller

# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/debug/dataset_acquision/video_client.py", pc_list)

sensors = initialize_device()

save_path = os.path.join("debug_", "capture_wo_robot")
shared_path = os.path.join(shared_dir, save_path)
last_capture_idx = -1

if os.path.exists(shared_path) and len(os.listdir(shared_path)) > 0:
    last_capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x)))
else:
    os.makedirs(shared_path, exist_ok=True)

capture_idx = last_capture_idx + 1
for i in range(5):
    time.sleep(2)
    # start
    os.makedirs(f'{shared_path}/{capture_idx}', exist_ok=True)
        
    sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
    sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
    sensors["signal_generator"].on(1)
    
    print(f"start_{i}")
    time.sleep(10)
    
    sensors["camera"].end()
    sensors['timecode_receiver'].end()
    sensors['signal_generator'].off(1)
    
    
    capture_idx += 1


for sensor_name, sensor in sensors.items():
    print(sensor_name)
    sensor.quit()