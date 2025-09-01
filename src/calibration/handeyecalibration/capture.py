import numpy as np
import time
import os
import shutil

from paradex.utils.file_io import shared_dir, find_latest_directory
from paradex.utils.env import get_pcinfo
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.robot_controller import get_arm

def copy_calib_files(save_path):
    camparam_dir = os.path.join(shared_dir, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_dir, "cam_param", camparam_name)

    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))
    
# === SETUP ===
arm_name = "xarm"

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())


dex_arm = get_arm(arm_name)
git_pull("merging", pc_list)
run_script("python src/calibration/handeyecalibration/client.py", pc_list)

camera_loader = RemoteCameraController("image", None)
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())


try:
    for i in range(31):
        target_action = np.load(f"data/hecalib/{i}_qpos.npy")
        dex_arm.home_robot(target_action)
        
        time.sleep(0.5)
        
        wrist6d = dex_arm.get_position()
        qpos = dex_arm.get_qpos()
        
        os.makedirs(f"{shared_dir}/handeye_calibration/{filename}/{i}/image", exist_ok=True)
        np.save(f"{shared_dir}/handeye_calibration/{filename}/{i}/robot", wrist6d)
        np.save(f"{shared_dir}/handeye_calibration/{filename}/{i}/qpos", qpos)
        
        camera_loader.start(f'shared_data/handeye_calibration/{filename}/{i}/image')
        camera_loader.end()
        
    copy_calib_files(f"/home/temp_id/shared_data/handeye_calibration/{filename}/0")

finally:
    camera_loader.quit()
    dex_arm.quit()   
    exit(0) 
