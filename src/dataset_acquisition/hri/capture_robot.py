import os
import argparse
import json
import time
import numpy as np
from collections import deque
from threading import Event, Lock, Thread

# import chime
# chime.theme('pokemon')

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_index
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.system import get_pc_list
from paradex.io.robot_controller.inspire_f1_tactile_plotter import InspireF1RealtimeTactilePlotter

EXCLUDED_PCS = {}
camera_pc_list = [pc for pc in get_pc_list() if pc not in EXCLUDED_PCS]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--device',
    choices=['xsens', 'quest3', 'occulus'],
    default="xsens",
    help="Teleoperation source. 'occulus' is kept as an alias for quest3.",
)
parser.add_argument('--camera', type=str, default=True)
parser.add_argument('--arm', type=str, default="xarm")
parser.add_argument('--hand', type=str, default="inspire_f1")
parser.add_argument('--capture_root', type=str, default="eccv2026/allegro_v5")
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--tactile', action="store_true")
parser.add_argument('--ip', action="store_true")
parser.add_argument('--visualize-tactile-realtime', action="store_true")
parser.add_argument('--xarm-servo-api', choices=["cartesian_aa", "angle_j"], default="cartesian_aa")
parser.add_argument('--quest-bind-host', default="0.0.0.0")
parser.add_argument('--quest-port', type=int, default=9000)
parser.add_argument('--quest-max-age-s', type=float, default=0.25)
parser.add_argument('--quest-extrapolation-delay-s', type=float, default=0.03)
parser.add_argument('--quest-extrapolation-horizon-s', type=float, default=0.10)
parser.add_argument('--quest-max-linear-speed-m-s', type=float, default=0.5)
parser.add_argument('--quest-max-angular-speed-rad-s', type=float, default=3.0)

args = parser.parse_args()

teleop_kwargs = None
if args.device in ("quest3", "occulus"):
    teleop_kwargs = {
        "host": args.quest_bind_host,
        "port": args.quest_port,
        "max_age_s": args.quest_max_age_s,
        "extrapolation_delay_s": args.quest_extrapolation_delay_s,
        "extrapolation_horizon_s": args.quest_extrapolation_horizon_s,
        "max_linear_speed_m_s": args.quest_max_linear_speed_m_s,
        "max_angular_speed_rad_s": args.quest_max_angular_speed_rad_s,
    }
    print(
        "Quest 3 teleoperation: listening on UDP "
        "{}:{} (max age {:.3f}s)".format(
            args.quest_bind_host,
            args.quest_port,
            args.quest_max_age_s,
        )
    )


stop_event = Event()
save_event = Event()
exit_event = Event()
grasp_yes_event = Event()
grasp_no_event = Event()
events = {"save": save_event, "stop": stop_event, "exit": exit_event}

listen_keyboard(
    {
        "c": save_event,
        "q": exit_event,
        "s": stop_event,
        "y": grasp_yes_event,
        "n": grasp_no_event,
    }
)
print("Keyboard control: c=start, s=stop, q=exit, y=grasp success, n=grasp fail")
print("In keyboard_control mode, gesture states 2/3 do not control session start/stop/exit.")


def wait_for_grasp_result():
    grasp_yes_event.clear()
    grasp_no_event.clear()
    print("Grasp success? Press y or n, then Enter.")

    while not exit_event.is_set():
        if grasp_yes_event.is_set():
            return "y"
        if grasp_no_event.is_set():
            return "n"
        time.sleep(0.01)

    return "n"

cs = CaptureSession(
    camera=args.camera,
    realsense=False,
    arm=args.arm,
    hand=args.hand,
    teleop=args.device,
    hand_side = "right",
    events=events,
    tactile=args.tactile,
    ip=args.ip,
    camera_pc_list=camera_pc_list,
    teleop_kwargs=teleop_kwargs,
)

tactile_plotter = None
if args.visualize_tactile_realtime:
    if args.hand != "inspire_f1":
        print("Realtime tactile visualization is only supported for inspire_f1. Ignoring option.")
    elif not args.tactile:
        print("Realtime tactile visualization requires --tactile. Ignoring option.")
    else:
        tactile_plotter = InspireF1RealtimeTactilePlotter(cs.hand)
        if tactile_plotter.enabled:
            tactile_plotter.start()

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

    timestamp_npy_path = os.path.join(episode_abs_path, "raw", "timestamps", "timestamp.npy")
    if os.path.exists(timestamp_npy_path):
        print(f"timestamp.npy length: {len(np.load(timestamp_npy_path))}")
    else:
        print(f"timestamp.npy not found at {timestamp_npy_path}")

    save_event.clear()
    stop_event.clear()

    grasp_input = wait_for_grasp_result()
    if grasp_input == "y":
        success_count += 1
    else:
        fail_count += 1

    os.makedirs(episode_abs_path, exist_ok=True)
    grasp_json_path = os.path.join(episode_abs_path, "grasp_result.json")
    with open(grasp_json_path, "w") as f:
        json.dump(
            {
                "episode": last_idx,
                "grasp_success": grasp_input == "y",
            },
            f,
            indent=2,
        )
    grasp_yes_event.clear()
    grasp_no_event.clear()
    
    
    paired_human_episode = int(input(f"Enter the episode number of paired human sequence for {args.name}: "))
    paired_info_json_path = os.path.join(shared_dir, episode_rel_path, "paired_human_episode.json")
    
    with open(paired_info_json_path, "w") as f:
        json.dump(
            {
                "human hand episode": last_idx,
                "paired human episode": paired_human_episode,
            },
            f,
            indent=2,
        )
        
    
        
    # print(f"Saved grasp result: {grasp_json_path}")
    print(f"Current Success count: {success_count} / Failure count: {fail_count}")
    print("===================================================")
    
    
    print(f"============== episode {last_idx} done =========================")

    if state == "exit":
        break

print("Exiting teleoperation recording.")
cs.end()
if tactile_plotter is not None:
    tactile_plotter.close()
