from threading import Event
import time
import argparse
import os
import datetime

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir

from paradex.calibration.utils import save_current_camparam

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
parser.add_argument('--sync_mode', default=False, action='store_true')
parser.add_argument('--fps', default=30, type=int)

args = parser.parse_args()

rcc = remote_camera_controller("video_main.py")

stop_event = Event()
save_event = Event()
exit_event = Event()

listen_keyboard({"c":save_event, "q":exit_event, "s":stop_event})

try:
    while not exit_event.is_set():
        if not save_event.is_set():
            time.sleep(0.01)
            continue
        
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rcc.start("video", args.sync_mode, f'{args.save_path}/{date_str}/raw', fps=args.fps)
        print(f"Capturing video to {args.save_path}/{date_str}/raw")
        while not stop_event.is_set() and not exit_event.is_set():
            time.sleep(0.02)
            
        rcc.stop()
        save_event.clear()
        stop_event.clear()
        
        save_current_camparam(os.path.join(shared_dir, args.save_path, date_str))

        
finally:
    rcc.end()