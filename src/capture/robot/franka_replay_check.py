"""Dry-run the hand-eye trajectory on the Franka WITHOUT cameras.

Moves the arm through the poses collected by ``franka_teaching.py`` (read from
``get_handeye_calib_traj("franka")``) so you can verify the motion is safe
BEFORE running the full ``src/calibration/handeye/capture.py`` (which also drives
the multi-camera capture). No images, no data saved -- motion only.

Prerequisites: franka_ros2 stack up with ``fr3_arm_controller`` ACTIVE (if a
teaching session left gravity_compensation active, switch back first):

    ros2 control switch_controllers -c /controller_manager \
        --activate fr3_arm_controller --deactivate gravity_compensation_example_controller

Usage:
    python src/capture/robot/franka_replay_check.py            # auto, pause at each pose
    python src/capture/robot/franka_replay_check.py --step     # wait for Enter before each move (safest)
    python src/capture/robot/franka_replay_check.py --home     # go to home first
"""

import os
import time
import argparse
import numpy as np

from paradex.io.robot_controller.franka_controller import (
    FrankaController, FRANKA_HOME_QPOS,
)
from paradex.calibration.utils import get_handeye_calib_traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="franka")
    parser.add_argument("--step_size", type=float, default=0.15,
                        help="rad per waypoint (smaller = slower/finer)")
    parser.add_argument("--step_time", type=float, default=0.5,
                        help="seconds per waypoint (larger = slower)")
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--pause", type=float, default=1.0,
                        help="pause at each pose in auto mode (s)")
    parser.add_argument("--step", action="store_true",
                        help="wait for Enter before each move (recommended first run)")
    parser.add_argument("--home", action="store_true",
                        help="move to FR3 home first, before the trajectory")
    args = parser.parse_args()

    traj_dir = get_handeye_calib_traj(args.arm)
    files = [f for f in os.listdir(traj_dir) if "_qpos" in f]
    files.sort(key=lambda x: int(x.split("_")[0]))
    if not files:
        print(f"[error] no *_qpos.npy in {traj_dir}")
        return
    print(f"{len(files)} poses from {traj_dir}")
    print("⚠️  로봇이 움직입니다 — E-stop 준비, 처음엔 --step 권장.")

    controller = FrankaController(step_size=args.step_size, step_time=args.step_time,
                                  min_steps=args.min_steps, max_steps=args.max_steps)
    try:
        if args.home:
            print(f"[home] -> {np.round(FRANKA_HOME_QPOS, 3).tolist()}")
            if args.step:
                input("  Enter로 home 이동...")
            controller.move(np.array(FRANKA_HOME_QPOS, dtype=float), is_servo=False)
            time.sleep(args.pause)

        for i, f in enumerate(files):
            q = np.asarray(np.load(os.path.join(traj_dir, f)), dtype=float)
            print(f"[{i+1}/{len(files)}] {os.path.basename(f)}: {np.round(q, 3).tolist()}")
            if args.step:
                input("  Enter로 이동 (Ctrl+C 중단)...")
            controller.move(q, is_servo=False)
            time.sleep(args.pause)

        print("완료 — 모든 포즈 이동 확인. 문제 없으면 capture.py로 진행.")
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        controller.end()


if __name__ == "__main__":
    main()
