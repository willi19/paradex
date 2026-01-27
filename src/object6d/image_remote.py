from threading import Event
import time
import argparse
import os
import datetime

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir

from paradex.calibration.utils import save_current_camparam, save_current_C2R

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

rcc = remote_camera_controller("image_main.py")

stop_event = Event()
save_event = Event()

listen_keyboard({"c":save_event, "q":stop_event})

save_dir = os.path.join(shared_dir, args.save_path)

try:
    while not stop_event.is_set():
        if not save_event.is_set():
            time.sleep(0.01)
            continue

        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_current_camparam(os.path.join(shared_dir, args.save_path, date_str))
        save_current_C2R(os.path.join(shared_dir, args.save_path, date_str))
        print(f"Capturing image to {args.save_path}/{date_str}")
        rcc.start("image", False, f'shared_data/{args.save_path}/{date_str}/raw')
        rcc.stop()
        save_event.clear()
        
finally:
    rcc.end()