import argparse
import os
import numpy as np
from datetime import datetime
import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.calibration.utils import save_current_camparam, handeye_calib_path, get_handeye_calib_traj
from paradex.utils.file_io import remove_home
from paradex.utils.system import network_info, get_pc_list

EXCLUDED_PCS = {}
pc_list = [pc for pc in get_pc_list() if pc not in EXCLUDED_PCS]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm", help="Name of the arm to save the current hand-eye calibration for.")
    args = parser.parse_args()

    name = network_info[args.arm]["name"]
    if name == "xarm":
        from paradex.io.robot_controller.xarm_controller import XArmController
        controller = XArmController(ip=network_info["xarm"]["param"]["ip"])
    else:
        raise NotImplementedError(f"Robot controller for {name} is not implemented.")
    root_dir = os.path.join(handeye_calib_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(root_dir, exist_ok=True)

    rcc = remote_camera_controller("handeye_calibration", pc_list=pc_list)
    try:
        save_current_camparam(os.path.join(root_dir, "0"))
        file_list = [
            file_name
            for file_name in os.listdir(get_handeye_calib_traj(args.arm))
            if "_qpos" in file_name
        ]
        file_list.sort(key=lambda x: int(x.split("_")[0]))

        for idx, file_name in enumerate(file_list):
            print(f"Capturing step {idx} over {len(file_list)}...")
            step_dir = os.path.join(root_dir, str(idx))
            image_dir = os.path.join(step_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            action = np.load(os.path.join(get_handeye_calib_traj(args.arm), file_name))

            controller.move(action, is_servo=False)
            time.sleep(0.5)

            rcc.start("image", False, remove_home(step_dir))
            rcc.stop()

            robot_data = controller.get_data()
            np.save(os.path.join(step_dir, "robot.npy"), robot_data)
            np.save(os.path.join(step_dir, "eef.npy"), robot_data["position"])
            np.save(os.path.join(step_dir, "qpos.npy"), robot_data["qpos"])

            print(f"Saved data for step {idx}")
    finally:
        try:
            rcc.end()
        finally:
            controller.end(set_break=False)
