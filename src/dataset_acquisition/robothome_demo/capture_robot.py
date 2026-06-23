import os
import argparse
import json
import time
from collections import deque
from threading import Event, Lock, Thread

# import chime
# chime.theme('pokemon')

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_index
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.io.robot_controller.inspire_f1_tactile_plotter import InspireF1RealtimeTactilePlotter

parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'], default="xsens")
parser.add_argument('--arm', type=str, default="xarm",
                    help="Arm controller name. Use 'none' (or empty) to disable.")
parser.add_argument('--hand', type=str, default="inspire_f1")
parser.add_argument('--hand_left', type=str, default=None,
                    help="Hand controller for left side (bimanual only). Falls back to --hand if unset.")
parser.add_argument('--hand_right', type=str, default=None,
                    help="Hand controller for right side (bimanual only). Falls back to --hand if unset.")
parser.add_argument('--hand_side', choices=["right", "left", "bimanual"], default="right")
parser.add_argument('--capture_root', type=str, default="eccv2026")
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--tactile', action="store_true")
parser.add_argument('--ip', action="store_true")
parser.add_argument('--visualize-tactile-realtime', action="store_true")
parser.add_argument('--xarm-servo-api', choices=["cartesian_aa", "angle_j"], default="cartesian_aa")

args = parser.parse_args()

# Treat "none"/""/"null" as no arm.
if args.arm is not None and args.arm.strip().lower() in ("", "none", "null"):
    args.arm = None


stop_event = Event()
save_event = Event()
exit_event = Event()

events = {"save": save_event, "stop": stop_event, "exit": exit_event}

listen_keyboard(
    {
        "c": save_event,
        "q": exit_event,
        "s": stop_event,

    }
)

if args.hand_side == "bimanual":
    cs_kwargs = dict(
        hand=None,
        hand_left=args.hand_left if args.hand_left is not None else args.hand,
        hand_right=args.hand_right if args.hand_right is not None else args.hand,
    )
else:
    cs_kwargs = dict(hand=args.hand)

cs = CaptureSession(
    camera=False,
    realsense=False,
    arm=args.arm,
    teleop=args.device,
    hand_side=args.hand_side,
    events=events,
    tactile=args.tactile,
    ip=args.ip,
    arm_kwargs={"servo_api": args.xarm_servo_api} if args.arm == "xarm" else None,
    **cs_kwargs,
)

name = args.name

last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", args.capture_root, args.name)))

success_count = 0
fail_count = 0

while not exit_event.is_set():
    state = cs.teleop(session_events=events, state_policy="keyboard_control")

    if state == "exit":
        break

    if state != "start":
        continue

    last_idx += 1
    print("Prepare to record new session:", name, "episode:", last_idx)
    episode_rel_path = os.path.join("capture", args.capture_root, args.name, str(last_idx))
    episode_abs_path = os.path.join(shared_dir, episode_rel_path)
    cs.start(episode_rel_path)
    # chime.info(sync=True)
    print("Starting new recording session:", name)
    print("Capturing index:", last_idx)
    
    
    state = cs.teleop(session_events=events, state_policy="keyboard_control")
    cs.stop()
    print("Stopped recording session:", name)

    save_event.clear()
    stop_event.clear()

    
    print(f"============== episode {last_idx} done =========================")

    if state == "exit":
        break

print("Exiting teleoperation recording.")
cs.end()
