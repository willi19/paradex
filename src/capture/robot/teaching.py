import os
import time
from threading import Event
from paradex.utils.keyboard_listener import listen_keyboard
import argparse

from xarm.wrapper import XArmAPI
import numpy as np

stop_event = Event()
save_event = Event()
listen_keyboard({'q':stop_event, 'c':save_event})

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
     
ip = "192.168.1.221"
arm = XArmAPI(ip, is_radian=True)
if arm.has_err_warn:
    arm.clean_error()
    
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Turn on manual mode before recording
arm.set_mode(2)
arm.set_state(0)

# print(arm.get_robot_sn())
# sn = arm.get_robot_sn()
# print(arm.get_safe_level())
# print(arm.get_joint_states())
# print(arm.get_joints_torque())
arm.set_teach_sensitivity(5)

home_pose = np.array([
    [0, 1 ,0, 300],
    [0, 0, 1, -200],
    [1, 0, 0, 200],
    [0, 0, 0, 1]])
pose = np.zeros(6)
pose[:3] = home_pose[:3,3]
from scipy.spatial.transform import Rotation as R
pose[3:] = R.from_matrix(home_pose[:3,:3]).as_euler("xyz")
print(arm.get_inverse_kinematics(pose))

idx = 0
try:
    while not stop_event.is_set():
        if save_event.is_set():
            _, pos_aa = arm.get_position_aa(is_radian=True)
            _, qpos = arm.get_joint_states()
            qpos = qpos[0][:6]
            
            np.save(os.path.join(args.save_path, f'{idx}_qpos.npy'), qpos)
            np.save(os.path.join(args.save_path, f'{idx}_aa.npy'), pos_aa)
            
            print(f"Saved pose {idx}: {pos_aa}")
            idx += 1
            save_event.clear()
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("Interrupted by user.")

# Turn off manual mode after recording
arm.set_mode(0)
arm.set_state(0)
arm.motion_enable(enable=False)
arm.disconnect()
print("Recording session ended.")
