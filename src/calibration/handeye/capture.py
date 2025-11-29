import argparse
import os
import numpy as np
from datetime import datetime
import tqdm
import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.calibration.utils import save_current_camparam, handeye_calib_path, get_handeye_calib_traj
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.utils.system import network_info
from paradex.utils.path import shared_dir
from paradex.utils.file_io import remove_home

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm", help="Name of the arm to save the current hand-eye calibration for.")
    args = parser.parse_args()
    
    name = network_info[args.arm]["name"]
    if name == "xarm":
        from paradex.io.robot_controller.xarm_controller import XArmController
        controller = XArmController(**network_info["xarm"]["param"])
    else:
        raise NotImplementedError(f"Robot controller for {name} is not implemented.")
    
    root_dir = os.path.join(handeye_calib_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    root_name = os.path.basename(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    
    rcc = remote_camera_controller("handeye_calibration")
    save_current_camparam(os.path.join(root_dir, "0"))
    file_list = [file_name for file_name in os.listdir(get_handeye_calib_traj(args.arm)) if "_qpos" in file_name]
    
    for idx, file_name in enumerate(file_list):
        os.makedirs(os.path.join(root_dir, str(idx)), exist_ok=True)
        action = np.load(os.path.join(get_handeye_calib_traj(args.arm), file_name))
        
        controller.move(action, is_servo=False)
        

        os.makedirs(f"{root_dir}/{idx}/images", exist_ok=True)
        rcc.start("image", False, remove_home(os.path.join(root_dir, str(idx))))
        rcc.stop()
        
        robot_data = controller.get_data()
        np.save(os.path.join(root_dir, str(idx), "robot.npy"), robot_data)
        np.save(f"{root_dir}/{idx}/eef.npy", robot_data["position"])
        np.save(f"{root_dir}/{idx}/qpos.npy", robot_data["qpos"])
        
        print(f"Saved data for step {idx}")
        
    controller.end(True)
    rcc.end()


    