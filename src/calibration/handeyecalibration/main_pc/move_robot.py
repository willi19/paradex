import os
import time
import math

from allegro_hand.controller import AllegroController
from xarm.wrapper import XArmAPI

import numpy as np
import datetime
from dex_robot.utils.file_io import shared_path
import shutil
from paradex.utils.io import find_latest_index, find_latest_directory


ALLEGRO_HOME_VALUES = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


class DexArmControl:
    def __init__(self, xarm_ip_address="192.168.1.221"):

        self.allegro = AllegroController()
        self.arm = XArmAPI(xarm_ip_address, report_type="devlop")

        self.max_hand_joint_vel = 100.0 / 360.0 * 2 * math.pi  # 100 degree / sec
        self.last_xarm_command = None
        self.last_allegro_command = None

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.reset()

        print("init complete")

    def reset(self):
        if self.arm.has_err_warn:
            self.arm.clean_error()

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # 0: position control, 1: servo control
        self.arm.set_state(state=0)

        
        # self.home_robot()

    def move_arm(self, target_action):
        self.arm.set_position_aa(axis_angle_pose=target_action, wait=True, is_radian=True)

    def move_hand(self, allegro_angles):
        num_steps = 10
        fps = 100
        for i in range(num_steps):
            self.allegro.hand_pose(allegro_angles)
            time.sleep(1 / fps)

    def get_joint_values(self):
        is_error = 1
        while is_error != 0:
            is_error, arm_joint_states = self.arm.get_joint_states(is_radian=True)
            xarm_angles = np.array(arm_joint_states[0])

        allegro_angles = self.allegro.current_joint_pose.position
        allegro_angles = np.array(allegro_angles)

        return xarm_angles, allegro_angles

    def quit(self):
        self.arm.motion_enable(enable=False)
        self.arm.disconnect()

def copy_calib_files(save_path):
    camparam_dir = os.path.join(shared_path, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_path, "cam_param", camparam_name)

    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))

if __name__ == "__main__":
    dex_arm = DexArmControl()
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    home_pose = np.array([ 0.4917225,   0.12536007 , 0.12568464 ,-1.00128184 ,-1.49969864 ,-1.37346572])
    
    for i in range(9):
        target_action = home_pose.copy()
        if i < 3:
            target_action[i+3] -= 0.4
        elif i < 6:   
            target_action[i] += 0.4
        elif i==7:
            target_action[i-3] -= 0.8
        else:
            target_action[i-3] += 0.8
        target_action[:3] *= 1000

        hand_action = np.load("data/calibration_pose/hand_{}.npy".format(i))
        dex_arm.move_hand(allegro_angles=hand_action)
        dex_arm.move_arm(target_action)
        time.sleep(2)

        xarm_angles,allegro_angles = dex_arm.get_joint_values()
        # print(allegro_angles)
        os.makedirs(f"/home/temp_id/shared_data/handeye_calibration/{date_str}/{i}/image", exist_ok=True)
        os.makedirs(f"/home/temp_id/shared_data/handeye_calibration/{date_str}/{i}/robot", exist_ok=True)

        np.save(f"/home/temp_id/shared_data/handeye_calibration/{date_str}/{i}/robot", np.concatenate([xarm_angles[:6],allegro_angles]))

        _ = input(f"Press Enter to continue... {i}")

    dex_arm.quit()    
    copy_calib_files(f"/home/temp_id/shared_data/handeye_calibration/{date_str}/0")