"""Franka FR3 teaching with the jog GUI — drive the arm from buttons, save keyposes.

Unlike `franka_teaching.py` (which only reads, and needs you to free the arm from
Desk), this drives the arm over FCI. Desk keeps control of nothing, so there is no
mode juggling: leave Desk in **Execution mode** with FCI active and jog from here.

Usage:
    # 1. Desk: Execution mode, joints unlocked, FCI activated
    # 2. Start the daemon:  ./cpp/franka_daemon/run_daemon.sh
    # 3. Run:
    python franka_teaching_gui.py --save_path system/current/hecalib/franka

GUI panels:
    Joint Control      J0..J6 -/+, held down to move
    Cartesian Control  X/Y/Z and Roll/Pitch/Yaw, held down to move
    Gripper            Open / Grasp (no-op without a Franka Hand attached)
    Save Pose          writes <idx>_qpos.npy (7,) + <idx>_pose.npy (4,4)

Output matches `franka_teaching.py` and is replayed by
`src/calibration/handeye/capture.py --arm franka`. Re-running resumes after the
highest existing index; delete the .npy files to start over.

Jogging streams joint velocities (`set_joint_velocity`) rather than per-tick position
targets, because Franka's `move()` blocks and would stutter.
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARADEX_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, _PARADEX_ROOT)

from paradex.io.robot_controller import get_arm
from paradex.io.robot_controller.gui_controller import RobotGUIController
from paradex.robot.utils import get_robot_urdf_path

# Flange link the hand-eye pipeline solves against (see EEF_LINK in
# src/calibration/handeye/calculate.py).
EEF_LINK = "fr3_link8"

parser = argparse.ArgumentParser(description="Franka teaching with jog GUI")
parser.add_argument('--save_path', type=str, required=True,
                    help="Directory for <idx>_qpos.npy / <idx>_pose.npy")
args = parser.parse_args()

arm = get_arm("franka")

if arm.get_data() is None:
    print("FAIL: no state from the daemon.")
    print("      Start it with ./cpp/franka_daemon/run_daemon.sh, and check its log")
    print("      for '[STREAM] Error: Net Exception' (a Desk mode switch kills the")
    print("      libfranka session — restart the daemon after changing modes).")
    sys.exit(1)

# A previous reflex abort latches the robot into an error state where every motion
# command is silently refused — clear it before the GUI comes up.
arm.error_recovery()

rgc = RobotGUIController(arm, jog_only=True, save_path=args.save_path,
                         urdf_path=get_robot_urdf_path(arm_name="franka"),
                         eef_link=EEF_LINK)
rgc.run()

arm.end()
