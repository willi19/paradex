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
from paradex.robot import RobotWrapper

def copy_calib_files(save_path):
    camparam_dir = os.path.join(shared_dir, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_dir, "cam_param", camparam_name)

    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))
    
# === SETUP ===
pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

camera_loader = RemoteCameraController("image", None)
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

dex_arm = XArmController()
git_pull("merging", pc_list)
run_script("python src/calibration/handeyecalibration/client.py", pc_list)

try:
    for i in range(6):
        target_action = np.load(f"data/hecalib/{i}.npy")
        dex_arm.home_robot(target_action)
        
        time.sleep(0.5)
        
        xarm_angles = dex_arm.get_position()
        os.makedirs(f"{shared_dir}/handeye_calibration/{filename}/{i}/image", exist_ok=True)
        np.save(f"{shared_dir}/handeye_calibration/{filename}/{i}/robot", xarm_angles[:6])
        
        camera_loader.start_capture(f'shared_dir/handeye_calibration/{filename}/{i}/image')
        camera_loader.end_capture()
        
    copy_calib_files(f"/home/temp_id/shared_data/handeye_calibration/{filename}/0")

finally:
    camera_loader.quit()
    dex_arm.quit()   
    exit(0) 
