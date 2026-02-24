import datetime
import os
import argparse
from threading import Event
import time

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_index

from paradex.utils.keyboard_listener import listen_keyboard
stop_event = Event()
save_event = Event()
exit_event = Event()

parser = argparse.ArgumentParser()


parser.add_argument('--name', type=str, required=True)


args = parser.parse_args()

cs = CaptureSession(
    camera=True,
    realsense=True,
    arm="xarm",
    hand="inspire_f1",
    teleop="xsens",
    hand_side="right",
    events={"save": save_event, "stop": stop_event, "exit": exit_event},
)

listen_keyboard({"c": save_event, "q": exit_event, "s": stop_event})


name = args.name

last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", "hri_xarm_f1", args.name)))

while not exit_event.is_set():
    if not save_event.is_set():
        # state = cs.teleop()
        stop_event.clear()
        time.sleep(0.01)
        continue
    
    # index = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    last_idx += 1
    print(last_idx)
    # state = cs.teleop()
    # cs.start(os.path.join("capture", "hri_bimanual", name, str(last_idx)))
    cs.start(os.path.join("capture", "hri_xarm_f1", args.name, str(last_idx)))

    print("Starting new recording session:", name)
    # state = cs.teleop()

    while not stop_event.is_set() and not exit_event.is_set():
        time.sleep(0.02)
    cs.stop()
    print("Stopped recording session:", name)
    save_event.clear()
    stop_event.clear()

print("Exiting teleoperation recording.")
cs.end()