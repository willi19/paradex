"""Franka FR3 teaching mode - hand-guide the robot and save keyposes.

Usage:
    # 1. Start daemon in Docker first
    # 2. Run:
    python franka_teaching.py --save_path /path/to/save
    python franka_teaching.py --save_path /path/to/save --host 172.16.1.11

    # Commands (type and press Enter):
    #   c  - save current pose (qpos + EE pose)
    #   q  - quit
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

# Enable guiding mode (all axes free)
print("Enabling guiding mode...")
resp = fc.set_guiding_mode([True, True, True, True, True, True], nullspace=True)
if resp.get("type") == "error":
    print(f"FAIL: {resp.get('message')}")
    fc.end()
    sys.exit(1)
print("[OK] Guiding mode ON - move robot by hand")
print()
print("Commands:")
print("  c + Enter  = save current pose")
print("  q + Enter  = quit")
print()

idx = 0
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

# Disable guiding mode: move to current position
print("Disabling guiding mode...")
data = fc.get_data()
if data is not None:
    fc.move(data["qpos"], speed_scale=0.1)

fc.end()
print(f"Teaching session ended. {idx} poses saved.")