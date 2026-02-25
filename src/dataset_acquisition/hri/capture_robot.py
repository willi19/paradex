import os
import argparse
import json
from threading import Event

import chime
chime.theme('pokemon')

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_index
from paradex.utils.keyboard_listener import listen_keyboard



parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'], default="xsens")
parser.add_argument('--arm', type=str, default="xarm")
parser.add_argument('--hand', type=str, default="inspire_f1")
parser.add_argument('--capture_root', type=str, default="eccv2026/inspire_f1")
parser.add_argument('--name', type=str, required=True)

args = parser.parse_args()

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

cs = CaptureSession(
    camera=True,
    realsense=False,
    arm=args.arm,
    hand=args.hand,
    teleop=args.device,
    hand_side = "right",
    events=events,
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
    chime.info(sync=True)
    print("Starting new recording session:", name)
    print("Capturing index:", last_idx)
    
    
    state = cs.teleop(session_events=events, state_policy="keyboard_control")
    cs.stop()
    print("Stopped recording session:", name)

    grasp_yes_event.clear()
    grasp_no_event.clear()
    print("Grasp success? Press y or n, then Enter.")
    while not exit_event.is_set():
        if grasp_yes_event.is_set():
            grasp_input = "y"
            success_count += 1
            break
        if grasp_no_event.is_set():
            grasp_input = "n"
            fail_count += 1
            break
    else:
        grasp_input = "n"

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
    # print(f"Saved grasp result: {grasp_json_path}")
    print(f"Current Success count: {success_count} / Failure count: {fail_count}")
    print("===================================================")
    grasp_yes_event.clear()
    grasp_no_event.clear()
    save_event.clear()
    stop_event.clear()

    if state == "exit":
        break

print("Exiting teleoperation recording.")
cs.end()
