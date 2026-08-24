"""Capture one xArm trajectory for camera-extrinsic and hand-eye calibration."""

import argparse
import math
import os
import time
from datetime import datetime

import numpy as np

from paradex.calibration.utils import (
    extrinsic_dir,
    get_handeye_calib_traj,
    save_current_camparam,
)
from paradex.io.camera_system.remote_camera_controller import (
    remote_camera_controller,
)
from paradex.utils.file_io import remove_home
from paradex.utils.system import get_pc_list, network_info


EXCLUDED_PCS = set()

# Change only this value to tune every robot move duration.
# 1.0: default, 2.0: twice as long, 0.5: half as long.
MOVE_TIME_SCALE = 1.0


def rotation_angle(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def target_eef_pose(controller, qpos):
    from paradex.io.robot_controller.xarm_controller import cart2homo

    code, cartesian = controller.arm.get_forward_kinematics(
        qpos.tolist(),
        input_is_radian=True,
        return_is_radian=True,
    )
    if code != 0:
        raise RuntimeError(f"xArm forward kinematics failed with code {code}")
    return cart2homo(np.asarray(cartesian, dtype=np.float64))


def estimate_move_duration(
    current_qpos,
    target_qpos,
    current_eef,
    target_eef,
):
    translation_distance = float(
        np.linalg.norm(target_eef[:3, 3] - current_eef[:3, 3])
    )
    angular_distance = rotation_angle(current_eef, target_eef)
    joint_distance = float(np.max(np.abs(target_qpos - current_qpos)))
    duration = MOVE_TIME_SCALE * max(
        1.0,
        translation_distance / 0.08,
        angular_distance / np.radians(25.0),
        joint_distance / np.radians(20.0),
    )
    return duration, translation_distance, angular_distance, joint_distance


def move_smoothly(controller, current_qpos, target_qpos, duration, servo_hz):
    steps = max(1, int(math.ceil(duration * servo_hz)))
    delta = target_qpos - current_qpos
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        alpha = step / steps
        controller.move(current_qpos + alpha * delta, is_servo=True)
        deadline = start_time + step / servo_hz
        time.sleep(max(0.0, deadline - time.perf_counter()))

    controller.move(target_qpos, is_servo=True)


def create_controller(arm, ip=None):
    name = network_info[arm]["name"]
    if name != "xarm":
        raise NotImplementedError(
            f"Robot controller for {name} is not implemented."
        )

    from paradex.io.robot_controller.xarm_controller import XArmController

    return XArmController(ip=ip or network_info["xarm"]["param"]["ip"])


def load_trajectory(trajectory_dir):
    file_names = [
        name for name in os.listdir(trajectory_dir) if name.endswith("_qpos.npy")
    ]
    file_names.sort(key=lambda name: int(name.split("_", 1)[0]))
    if not file_names:
        raise FileNotFoundError(
            f"No *_qpos.npy poses found in {trajectory_dir}"
        )
    return file_names


def capture_sequence(
    root_dir,
    arm,
    ip,
    camera_controller,
    trajectory_dir,
):
    controller = create_controller(arm, ip)
    try:
        trajectory = load_trajectory(trajectory_dir)
        save_current_camparam(os.path.join(root_dir, "0"))

        for index, file_name in enumerate(trajectory):
            pose_path = os.path.join(trajectory_dir, file_name)
            action = np.asarray(np.load(pose_path), dtype=np.float64)
            if action.shape != (6,) or not np.all(np.isfinite(action)):
                raise ValueError(
                    f"Invalid xArm qpos in {pose_path}: expected finite shape "
                    f"(6,), got {action.shape}"
                )

            robot_data = controller.get_data()
            current_qpos = np.asarray(robot_data["qpos"], dtype=np.float64)
            current_eef = np.asarray(robot_data["position"], dtype=np.float64)
            target_eef = target_eef_pose(controller, action)
            (
                move_duration,
                translation_distance,
                angular_distance,
                joint_distance,
            ) = estimate_move_duration(
                current_qpos,
                action,
                current_eef,
                target_eef,
            )

            print(f"Capturing step {index} over {len(trajectory)}...")
            print(
                f"Move duration: {move_duration:.2f}s, "
                f"translation: {translation_distance * 1000:.1f}mm, "
                f"rotation: {np.degrees(angular_distance):.1f}deg, "
                f"max joint: {np.degrees(joint_distance):.1f}deg"
            )
            move_smoothly(
                controller,
                current_qpos,
                action,
                move_duration,
                50.0,
            )
            time.sleep(0.5)

            step_dir = os.path.join(root_dir, str(index))
            os.makedirs(step_dir, exist_ok=True)
            camera_controller.snapshot(remove_home(step_dir))

            robot_data = controller.get_data()
            np.save(os.path.join(step_dir, "robot.npy"), robot_data)
            np.save(os.path.join(step_dir, "eef.npy"), robot_data["position"])
            np.save(os.path.join(step_dir, "qpos.npy"), robot_data["qpos"])
            print(f"Saved data for step {index}")
    finally:
        controller.end(set_break=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Capture camera-extrinsic and xArm hand-eye calibration data in "
            "one trajectory run."
        )
    )
    parser.add_argument("--arm", default="xarm")
    parser.add_argument(
        "--ip",
        default=None,
        help="Override the configured xArm IP.",
    )
    parser.add_argument(
        "--trajectory-dir",
        default=None,
        help="Directory containing ordered *_qpos.npy calibration poses.",
    )
    args = parser.parse_args()

    if MOVE_TIME_SCALE <= 0:
        raise ValueError("MOVE_TIME_SCALE must be positive")

    name = network_info[args.arm]["name"]
    if name != "xarm":
        raise NotImplementedError(
            f"Robot controller for {name} is not implemented."
        )

    trajectory_dir = args.trajectory_dir or get_handeye_calib_traj(args.arm)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(extrinsic_dir, session_name)
    os.makedirs(root_dir, exist_ok=False)
    print(f"All-in-one calibration session: {session_name}")
    print(f"Saving to: {root_dir}")

    pc_list = [pc for pc in get_pc_list() if pc not in EXCLUDED_PCS]
    camera_controller = remote_camera_controller(
        "allinone_calibration",
        pc_list=pc_list,
    )
    stream_started = False
    try:
        camera_controller.start("stream", False, fps=30)
        stream_started = True
        capture_sequence(
            root_dir,
            args.arm,
            args.ip,
            camera_controller,
            trajectory_dir,
        )
    finally:
        try:
            if stream_started:
                camera_controller.stop()
        finally:
            camera_controller.end()

    print("Capture complete.")
    print(
        "Next: python src/calibration/allinone/calculate.py "
        f"--name {session_name} --arm {args.arm}"
    )


if __name__ == "__main__":
    main()
