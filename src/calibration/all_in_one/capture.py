import threading
import numpy as np
import time
import zmq
import os
import shutil
import argparse

from paradex.utils.file_io import shared_dir, find_latest_directory, rsc_path
from paradex.utils.env import get_pcinfo
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.robot_controller import XArmController

def copy_calib_files(save_path):
    camparam_dir = os.path.join(shared_dir, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_dir, "cam_param", camparam_name)

    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))
    
# === SETUP ===
pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

dex_arm = XArmController()
git_pull("merging", pc_list)
run_script("python src/calibration/all_in_one/client.py", pc_list)

camera_loader = RemoteCameraController("image", None)
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

wrist_rot = np.array([[0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0.05], 
                      [0, 0, 0, 1]])
try:
    cnt = 0
    for x in range(30, 61, 2):
        for y in range(-30, 31, 2):
            if np.sqrt(x**2 + y**2) > 0.62:
                continue
            target_action = wrist_rot.copy()
            target_action[0, 3] = x
            target_action[1, 3] = y
            dex_arm.home_robot(target_action)
        
            time.sleep(0.5)
            
            wrist6d = dex_arm.get_position()
            os.makedirs(f"{shared_dir}/all_in_one/{filename}/{cnt}/image", exist_ok=True)
            np.save(f"{shared_dir}/all_in_one/{filename}/{cnt}/robot", wrist6d)
            
            camera_loader.start_capture(f'shared_data/all_in_one/{filename}/{cnt}/image')
            camera_loader.end_capture()
        
    copy_calib_files(f"/home/temp_id/shared_data/all_in_one/{filename}/0")

finally:
    camera_loader.quit()
    dex_arm.quit()   
    exit(0) 
