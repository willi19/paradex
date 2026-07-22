import argparse
import json
import os
import numpy as np
from datetime import datetime
import tqdm
import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.calibration.utils import save_current_camparam, handeye_calib_path, get_handeye_calib_traj, EEF_LINK
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.utils.system import network_info
from paradex.utils.path import shared_dir
from paradex.utils.file_io import remove_home

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm", help="Name of the arm to save the current hand-eye calibration for.")
    parser.add_argument("--via_threshold", type=float, default=1.57,
                        help="Route through the taught via pose when any joint moves "
                             "more than this (rad) between waypoints. Needs "
                             "`via_qpos.npy` in the trajectory dir.")
    args = parser.parse_args()
    
    if args.arm == "xarm":
        from paradex.io.robot_controller.xarm_controller import XArmController
        controller = XArmController(**network_info["xarm"]["param"])
    elif args.arm == "franka":
        # franka_daemon must already be running: ./cpp/franka_daemon/run_daemon.sh
        from paradex.io.robot_controller.franka_controller import FrankaController
        controller = FrankaController(network_info["franka"])
        if not controller.ping():
            raise RuntimeError(
                f"franka_daemon not reachable at {network_info['franka']}:5555. "
                "Start it with ./cpp/franka_daemon/run_daemon.sh"
            )
    else:
        raise NotImplementedError(f"Robot controller for {args.arm} is not implemented.")
    
    root_dir = os.path.join(handeye_calib_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    root_name = os.path.basename(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    
    # Which arm produced this capture is not recoverable from the saved files alone
    # (qpos length is the only hint, and calculate.py needs the arm to pick the URDF
    # and flange link), so record it up front.
    os.makedirs(os.path.join(root_dir, "0"), exist_ok=True)
    with open(os.path.join(root_dir, "0", "meta.json"), "w") as f:
        json.dump({"arm": args.arm,
                   "timestamp": root_name,
                   "eef_link": EEF_LINK[args.arm]}, f, indent=2)

    rcc = remote_camera_controller("handeye_calibration")
    save_current_camparam(os.path.join(root_dir, "0"))
    traj_dir = get_handeye_calib_traj(args.arm)
    file_list = [file_name for file_name in os.listdir(traj_dir)
                 if "_qpos" in file_name and not file_name.startswith("via")]
    file_list.sort(key=lambda x: int(x.split("_")[0]))

    # A big joint jump between waypoints is swept through blindly — a wrist flip can
    # drag the board across the floor on the way. If a `via_qpos.npy` was taught (a
    # retracted, safe pose), pass through it first. No images are taken there.
    via_path = os.path.join(traj_dir, "via_qpos.npy")
    via_qpos = np.load(via_path) if os.path.exists(via_path) else None
    if via_qpos is None:
        print(f"[via] none taught ({via_path}) — moving directly between waypoints")

    prev_qpos = None
    for idx, file_name in enumerate(file_list):
        os.makedirs(os.path.join(root_dir, str(idx)), exist_ok=True)
        action = np.load(os.path.join(traj_dir, file_name))

        if (via_qpos is not None and prev_qpos is not None
                and np.abs(action - prev_qpos).max() > args.via_threshold):
            jump = np.degrees(np.abs(action - prev_qpos).max())
            print(f"[via] step {idx}: {jump:.0f} deg jump -> routing through via pose")
            controller.move(via_qpos, is_servo=False)

        controller.move(action, is_servo=False)
        prev_qpos = action


        os.makedirs(f"{root_dir}/{idx}/images", exist_ok=True)
        rcc.start("image", False, remove_home(os.path.join(root_dir, str(idx))))
        rcc.stop()
        
        robot_data = controller.get_data()
        np.save(os.path.join(root_dir, str(idx), "robot.npy"), robot_data)
        np.save(f"{root_dir}/{idx}/eef.npy", robot_data["position"])
        np.save(f"{root_dir}/{idx}/qpos.npy", robot_data["qpos"])
        
        print(f"Saved data for step {idx}")
        
    if args.arm == "franka":
        controller.end()          # FrankaController.end() takes no args
    else:
        controller.end(True)
    rcc.end()


    