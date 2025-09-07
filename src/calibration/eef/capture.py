import numpy as np
import time
import os
import shutil
import random
from scipy.spatial.transform import Rotation as R

from paradex.utils.file_io import shared_dir, find_latest_directory, load_latest_C2R
from paradex.utils.env import get_pcinfo
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.robot_controller import get_arm, get_hand

def copy_calib_files(save_path):
    camparam_dir = os.path.join(shared_dir, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_dir, "cam_param", camparam_name)

    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))
    
# === SETUP ===
arm_name = "xarm"
hand_name = "allegro"

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())


git_pull("merging", pc_list)
run_script("python src/calibration/eef/client.py", pc_list)

camera_loader = RemoteCameraController("image", None, debug=True)
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

start_pos= np.array([[0, 0, 1, 0.35],
                    [1, 0, 0, 0.0],
                    [0, 1, 0, 0.15], 
                    [0, 0, 0, 1]])

dex_arm = get_arm(arm_name)
hand = get_hand(hand_name)
dex_arm.home_robot(start_pos)
for i in range(30):
    
    if False:#os.path.exists(f"data/eef/{i}_handqpos.npy"):
        hand_action = np.load(f"data/eef/{i}_handqpos.npy")
    else:
        hand_action = np.zeros(16)
        for j in range(3):
            hand_action[j*4] = (random.random() - 0.5) / 2
            hand_action[j*4+1] = random.random()
            np.save(f"data/eef/{i}_handqpos.npy",hand_action)
        hand_action[12] = 1.4 - random.random() / 2
        hand_action[13] = random.random()
    hand.set_target_action(hand_action)
    
    time.sleep(0.5)
    
    wrist6d = dex_arm.get_position()
    qpos = dex_arm.get_qpos()
    
    os.makedirs(f"{shared_dir}/eef/{filename}/{i}/image", exist_ok=True)
    np.save(f"{shared_dir}/eef/{filename}/{i}/robot.npy", wrist6d)
    np.save(f"{shared_dir}/eef/{filename}/{i}/qpos.npy", qpos)
    
    camera_loader.start(f'shared_data/eef/{filename}/{i}/image')
    camera_loader.end()
    
    hand_pose = hand.get_data()
    np.save(f"{shared_dir}/eef/{filename}/{i}/hand.npy", hand_pose)
    
copy_calib_files(f"{shared_dir}/eef/{filename}/0")
c2r = load_latest_C2R()
np.save(f"{shared_dir}/eef/{filename}/0/C2R.npy", c2r)

# finally:
#     camera_loader.quit()
#     dex_arm.quit()   
#     exit(0) 

camera_loader.quit()
hand.quit()
dex_arm.quit()   
exit(0) 
