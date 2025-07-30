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

def initialize_device():
    controller = {}
    
    controller["camera"] = RemoteCameraController("video", serial_list=None, sync=True)
    controller["signal_generator"] = UTGE900()
    controller["timecode_receiver"] = TimecodeReceiver()
    
    return controller

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

run_script(f"python src/dataset_acquision/lookup/video_client.py", pc_list)

sensors = initialize_device()

capture_idx = 0
while capture_idx < 1:
    sensors['camera'].start(f"erasethis/videos")
    sensors['timecode_receiver'].start(f"erasethis/{capture_idx}/raw/timestamp")
    sensors["signal_generator"].on(1)
    
    print("start")
    chime.info()
        
    sensors["camera"].end()
    sensors['timecode_receiver'].end()
    
    sensors['signal_generator'].off(1)
    chime.success()
    
    capture_idx += 1
    sensors["camera"].quit()
    
    # n = input()
    
    # # run_script(f"python src/capture/camera/image_client.py", pc_list)

    # camera_loader = RemoteCameraController("image", None)
    # camera_loader.start("shared_dir/erasethis")
    # camera_loader.end()
    # camera_loader.quit()

    capture_idx += 1
    
for sensor_name, sensor in sensors.items():
    if sensor_name == "camera":
        continue
    sensor.quit()