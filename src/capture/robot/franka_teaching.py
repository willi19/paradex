"""Franka FR3 teaching - record keyposes while you pose the arm by hand.

This script NEVER commands the robot. It only reads the FCI state stream and writes
a pose on `c`, so you are free to move the arm however you like — jogging from Desk,
hand-guiding, whatever. Nothing here can fight you for control.

Usage:
    # 1. Set the Desk mode you want to pose the arm in.
    # 2. Start the daemon:  ./cpp/franka_daemon/run_daemon.sh
    # 3. Run:
    python franka_teaching.py --save_path /path/to/save --host 127.0.0.1

    # Commands (type and press Enter):
    #   c  - save current pose (qpos + EE pose)
    #   q  - quit

Writes `<idx>_qpos.npy` (7,) and `<idx>_pose.npy` (4,4) per `c`. Re-running resumes
after the highest existing index; delete the .npy files to start over.

⚠️ Set the Desk mode BEFORE starting the daemon, and don't switch while it runs.
   Switching modes kills the daemon's libfranka session: `ping` still answers (that
   is just the daemon's own ZMQ socket) but the state stream dies and the log fills
   with `[STREAM] Error: Net Exception`. get_data() then returns None and `c` saves
   nothing. Restart the daemon after any mode change.

Note: this script used to call `setGuidingMode` to free the arm itself. That does not
work on this rig — it returns "success" while the arm stays locked (measured: 0.0 deg
over three tries), because it only selects which axes are free *once guiding is
engaged*, and engaging it in Execution mode needs the X4 External Enabling Device.
"""

import os
import sys
import time
import argparse
from threading import Event

import numpy as np

# Add paradex to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARADEX_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, _PARADEX_ROOT)

from paradex.io.robot_controller.franka_controller import FrankaController
from paradex.utils.keyboard_listener import listen_keyboard

stop_event = Event()
save_event = Event()
listen_keyboard({'q': stop_event, 'c': save_event})

parser = argparse.ArgumentParser(description="Franka teaching mode")
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--command_port', type=int, default=5555)
parser.add_argument('--state_port', type=int, default=5556)
args = parser.parse_args()

if args.save_path is not None:
    os.makedirs(args.save_path, exist_ok=True)

# Connect to daemon
print(f"Connecting to daemon at {args.host}:{args.command_port}...")
fc = FrankaController(args.host, args.command_port, args.state_port)

if not fc.ping():
    print("FAIL: cannot connect to daemon")
    sys.exit(1)
print("[OK] Connected")

# The state stream must be live, or `c` silently saves nothing. It dies when the
# Desk mode is switched after the daemon started (see the module docstring).
if fc.get_data() is None:
    time.sleep(1.0)                      # allow for a slow ZMQ SUB join
    if fc.get_data() is None:
        print("FAIL: no state from the daemon (ping works, but the stream is dead).")
        print("      Restart it: ./cpp/franka_daemon/run_daemon.sh")
        print("      Check its log for '[STREAM] Error: Net Exception'.")
        fc.end()
        sys.exit(1)

print("Pose the arm however you like (Desk jogging / hand-guiding).")
print("This script only reads — it never commands the robot.")
print()
print("Commands:")
print("  c + Enter  = save current pose")
print("  q + Enter  = quit")
print()

# Resume after existing poses instead of overwriting them from 0. Overwriting
# only the first N would leave a mix of old and new waypoints in one trajectory.
idx = 0
if args.save_path is not None:
    existing = [int(f.split('_')[0]) for f in os.listdir(args.save_path) if '_qpos' in f]
    if existing:
        idx = max(existing) + 1
        print(f"[resume] {len(existing)} pose(s) already saved — continuing from {idx}")
        print(f"         (delete {args.save_path}/*.npy to start over)")
        print()

try:
    while not stop_event.is_set():
        if save_event.is_set() and args.save_path is not None:
            data = fc.get_data()
            if data is not None:
                qpos = data["qpos"]
                ee_pose = data["position"]  # 4x4 homogeneous
                gripper_w = data["gripper_width"]

                np.save(os.path.join(args.save_path, f'{idx}_qpos.npy'), qpos)
                np.save(os.path.join(args.save_path, f'{idx}_pose.npy'), ee_pose)

                print(f"Saved pose {idx}: qpos={np.degrees(qpos[:3]).round(1)}... "
                      f"EE=[{ee_pose[0,3]:.3f}, {ee_pose[1,3]:.3f}, {ee_pose[2,3]:.3f}] "
                      f"gripper={gripper_w:.4f}m")
                idx += 1
            save_event.clear()
        elif save_event.is_set():
            # No save path but still print pose
            data = fc.get_data()
            if data is not None:
                qpos = data["qpos"]
                ee_pose = data["position"]
                print(f"Pose (not saved): qpos={np.degrees(qpos).round(1)} "
                      f"EE=[{ee_pose[0,3]:.3f}, {ee_pose[1,3]:.3f}, {ee_pose[2,3]:.3f}]")
            save_event.clear()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

stop_event.set()
# Nothing to undo: we never took control of the arm, so we must not command it
# on the way out either — Desk still owns it.
fc.end()
print(f"Teaching session ended. {idx} poses saved.")