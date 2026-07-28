"""Jog GUI for moving a robot to known poses under manual control.

A utility for driving the arm around between captures — jog it (joint sliders / arrow
keys / cartesian buttons) and hold a "Go To Pose" button to advance toward HOME (or a
saved teaching pose) only while held, stopping the instant you release. Use it to bring
the arm to a safe start pose before a hand-eye capture, so the capture's first move
isn't a blind long jump from wherever the arm was left.

Usage:
    # franka: start the daemon first (./cpp/franka_daemon/run_daemon.sh), Execution mode
    python src/util/robot/move_robot_gui.py --arm franka
    # include the taught hand-eye poses as goto targets:
    python src/util/robot/move_robot_gui.py --arm franka --poses_dir system/current/hecalib/franka
"""
import argparse
import glob
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARADEX_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, _PARADEX_ROOT)

from paradex.io.robot_controller import get_arm
from paradex.io.robot_controller.gui_controller import RobotGUIController
from paradex.robot.utils import get_robot_urdf_path
from paradex.calibration.utils import EEF_LINK

# Franka's documented "ready" pose (radians). A neutral, well-conditioned home.
FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def build_goto_poses(arm, poses_dir):
    poses = {}

    # Prefer a rig-specific HOME saved from the live arm (home_qpos.npy in poses_dir);
    # fall back to franka's datasheet ready pose. Save one with:
    #   np.save('<poses_dir>/home_qpos.npy', arm.get_data()['qpos'])
    home = None
    if poses_dir:
        home_path = os.path.join(poses_dir, "home_qpos.npy")
        if os.path.exists(home_path):
            home = np.load(home_path)
    if home is None and arm == "franka":
        home = FRANKA_HOME
    if home is not None:
        poses["HOME"] = home

    # Optionally expose the first taught trajectory pose, so you can pre-position the
    # arm at the capture's start point (avoids the long first jump).
    if poses_dir:
        first = os.path.join(poses_dir, "0_qpos.npy")
        if os.path.exists(first):
            poses["TRAJ START (0)"] = np.load(first)
        via = os.path.join(poses_dir, "via_qpos.npy")
        if os.path.exists(via):
            poses["VIA"] = np.load(via)
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="franka")
    ap.add_argument("--poses_dir", default=None,
                    help="dir with <n>_qpos.npy / via_qpos.npy to add as goto targets")
    args = ap.parse_args()

    arm = get_arm(args.arm)

    if arm.get_data() is None:
        print("FAIL: no robot state. Start the daemon first "
              "(franka: ./cpp/franka_daemon/run_daemon.sh) and check the mode.")
        sys.exit(1)

    # Clear any latched reflex so the first move isn't silently refused.
    if hasattr(arm, "error_recovery"):
        arm.error_recovery()

    goto = build_goto_poses(args.arm, args.poses_dir)
    print(f">>> goto poses: {list(goto)}")

    rgc = RobotGUIController(
        arm, jog_only=True,
        urdf_path=get_robot_urdf_path(arm_name=args.arm),
        eef_link=EEF_LINK.get(args.arm),
        goto_poses=goto,
    )
    rgc.run()
    arm.end()


if __name__ == "__main__":
    main()
