import numpy as np
import time
import os
import shutil
import random
from scipy.spatial.transform import Rotation as R

from paradex.io.camera import camera_loader
from paradex.utils.file_io import shared_dir, find_latest_directory, get_robot_urdf_path
from paradex.utils.env import get_pcinfo
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.robot_controller import get_arm
from paradex.robot.robot_wrapper import RobotWrapper

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
git_pull("paradex2", pc_list)
run_script("python src/calibration/handeyecalibration/client.py", pc_list)

camera_loader = RemoteCameraController("image", None)
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
robot = RobotWrapper(get_robot_urdf_path(arm_name, None))

start_pos= np.array([[-0.46187562,  0.59357145,  0.65904768,  0.3],
            [-0.17790253, -0.7899525,   0.58679276,  0.3],
            [ 0.86891979,  0.15377902,  0.47045767,  0.2],
            [ 0,          0,          0,          1]])
# try:
for i in range(31):
    if os.path.exists(f"data/hecalib/{i}_se3.npy"):
        target_action = np.load(f"data/hecalib/{i}_se3.npy")
    
    else:
        os.makedirs("data/hecalib", exist_ok=True)
        euler_angles = [
            (random.random() - 0.5),        # z축 회전
            (random.random() - 0.5),        # y축 회전  
            (random.random() - 0.5) / 2.5   # x축 회전 (더 작게)
        ]
        delta_transform = [
            (random.random() - 0.5) / 10,    # x축 이동
            (random.random() - 0.5) / 10,    # y축 이동
            random.random() / 20
        ]
        
        # 4x4 변환 행렬 생성
        rotation_matrix = R.from_euler('zyx', euler_angles).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        
        target_action = start_pos @ transform
        target_action[:3, 3] += delta_transform

        np.save(f"data/hecalib/{i}_se3.npy", target_action)

    target_action = np.load(f"data/hecalib/{i}_se3.npy")
    # robot.compute_forward_kinematics(target_action)
    # target_6d = robot.get_link_pose(robot.get_link_index("link6"))
    # target_6d[2, 3] += 0.15
    
    dex_arm.home_robot(target_action)
    time.sleep(0.5)
    
    wrist6d = dex_arm.get_position()
    qpos = dex_arm.get_qpos()
    
    os.makedirs(f"{shared_dir}/handeye_calibration/{filename}/{i}/image", exist_ok=True)
    np.save(f"{shared_dir}/handeye_calibration/{filename}/{i}/robot", wrist6d)
    np.save(f"{shared_dir}/handeye_calibration/{filename}/{i}/qpos", qpos)
    
    camera_loader.start(f'shared_data/handeye_calibration/{filename}/{i}/image')
    camera_loader.end()

copy_calib_files(os.path.join(shared_dir, "handeye_calibration", filename, "0"))
camera_loader.quit()
dex_arm.quit()   

# finally:
#     exit(0) 
