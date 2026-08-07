import argparse
import os
import numpy as np
from datetime import datetime
import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.calibration.utils import (
    get_handeye_calib_traj,
    handeye_calib_bimanual_path,
    handeye_calib_path,
    save_current_camparam,
)
from paradex.utils.file_io import remove_home
from paradex.utils.path import shared_dir
from paradex.utils.system import network_info, get_pc_list

EXCLUDED_PCS = {}
pc_list = [pc for pc in get_pc_list() if pc not in EXCLUDED_PCS]
LEFT_XARM_IP = "192.168.1.196"
BIMANUAL_POSE_SESSION = "20260803_125942"
BIMANUAL_POSE_DIR = os.path.join(
    shared_dir,
    "handeye_pose_matching",
    BIMANUAL_POSE_SESSION,
)


def create_controller(arm, ip=None):
    name = network_info[arm]["name"]
    if name != "xarm":
        raise NotImplementedError(f"Robot controller for {name} is not implemented.")

    from paradex.io.robot_controller.xarm_controller import XArmController

    return XArmController(ip=ip or network_info["xarm"]["param"]["ip"])


def capture_sequence(root_dir, arm, ip, rcc, trajectory_dir=None):
    controller = create_controller(arm, ip)
    try:
        save_current_camparam(os.path.join(root_dir, "0"))
        if trajectory_dir is None:
            trajectory_dir = get_handeye_calib_traj(arm)
        file_list = [
            file_name
            for file_name in os.listdir(trajectory_dir)
            if file_name.endswith("_qpos.npy")
        ]
        file_list.sort(key=lambda x: int(x.split("_")[0]))
        if not file_list:
            raise FileNotFoundError(
                f"No *_qpos.npy poses found in {trajectory_dir}"
            )

        for idx, file_name in enumerate(file_list):
            print(f"Capturing step {idx} over {len(file_list)}...")
            step_dir = os.path.join(root_dir, str(idx))
            action = np.asarray(
                np.load(os.path.join(trajectory_dir, file_name)),
                dtype=np.float64,
            )
            if action.shape != (6,) or not np.all(np.isfinite(action)):
                raise ValueError(
                    f"Invalid xArm qpos in {os.path.join(trajectory_dir, file_name)}: "
                    f"expected finite shape (6,), got {action.shape}"
                )

            controller.move(action, is_servo=False)
            time.sleep(0.5)

            rcc.snapshot(remove_home(step_dir))

            robot_data = controller.get_data()
            np.save(os.path.join(step_dir, "robot.npy"), robot_data)
            np.save(os.path.join(step_dir, "eef.npy"), robot_data["position"])
            np.save(os.path.join(step_dir, "qpos.npy"), robot_data["qpos"])

            print(f"Saved data for step {idx}")
    finally:
        controller.end(set_break=False)


def wait_for_left_arm():
    while True:
        answer = input(
            "Move the marker to the left arm and move the robots to safe positions. "
            "Continue with the left xArm? [Y]: "
        )
        if answer.strip().lower() == "y":
            return
        print("Waiting for Y...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm", help="Name of the arm to save the current hand-eye calibration for.")
    parser.add_argument(
        "--bimanual",
        action="store_true",
        help="Capture the right and left xArms into one bimanual calibration session.",
    )
    args = parser.parse_args()

    name = network_info[args.arm]["name"]
    if name != "xarm":
        raise NotImplementedError(f"Robot controller for {name} is not implemented.")

    base_dir = handeye_calib_bimanual_path if args.bimanual else handeye_calib_path
    root_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(root_dir, exist_ok=True)

    rcc = remote_camera_controller("handeye_calibration", pc_list=pc_list)
    stream_started = False
    try:
        rcc.start("stream", False, fps=30)
        stream_started = True

        if args.bimanual:
            right_dir = os.path.join(root_dir, "Right")
            left_dir = os.path.join(root_dir, "Left")
            os.makedirs(right_dir, exist_ok=True)

            capture_sequence(
                right_dir,
                args.arm,
                network_info["xarm"]["param"]["ip"],
                rcc,
                os.path.join(BIMANUAL_POSE_DIR, "Right"),
            )
            rcc.stop()
            stream_started = False
            wait_for_left_arm()
            rcc.start("stream", False, fps=30)
            stream_started = True
            os.makedirs(left_dir, exist_ok=True)
            capture_sequence(
                left_dir,
                args.arm,
                LEFT_XARM_IP,
                rcc,
                os.path.join(BIMANUAL_POSE_DIR, "Left"),
            )
        else:
            capture_sequence(
                root_dir,
                args.arm,
                network_info["xarm"]["param"]["ip"],
                rcc,
            )
    finally:
        try:
            if stream_started:
                rcc.stop()
        finally:
            rcc.end()


if __name__ == "__main__":
    main()
