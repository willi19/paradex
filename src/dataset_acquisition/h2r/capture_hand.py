import os
import argparse
from threading import Event
import time

from paradex.utils.file_io import find_latest_index
from paradex.utils.path import shared_dir
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.calibration.utils import save_current_camparam, save_current_C2R

parser = argparse.ArgumentParser()
# parser.add_argument('--hand_side', type=str, default='right')
parser.add_argument('--name', type=str)
parser.add_argument('--capture_root', default="h2r/")
args = parser.parse_args()

save_event = Event()
exit_event = Event()

listen_keyboard({"c": save_event, "q": exit_event})

rcc = remote_camera_controller("capture_hand.py")

name = args.name

last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", args.capture_root, name)))

try:
    while not exit_event.is_set():
        if not save_event.is_set():
            time.sleep(0.01)
            continue

        last_idx += 1
        save_path = os.path.join("capture", args.capture_root, name, str(last_idx))

        print("Capturing image:", name)
        rcc.start("image", True, f'shared_data/{save_path}/raw')
        rcc.stop()

        save_current_camparam(os.path.join(shared_dir, save_path))
        save_current_C2R(os.path.join(shared_dir, save_path))

        save_event.clear()

        print(f"============== episode {last_idx} done =========================")

finally:
    print("Exiting recording.")
    rcc.end()
