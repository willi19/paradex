from threading import Event
import time
import argparse
import os

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.file_io import find_latest_index, shared_dir

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

rcc = remote_camera_controller("image_main.py")

stop_event = Event()
save_event = Event()

listen_keyboard({"c":save_event, "q":stop_event})

save_dir = os.path.join(shared_dir, args.save_path)
last_idx = int(find_latest_index(save_dir)) if os.path.exists(save_dir) else -1

try:
    while not stop_event.is_set():
        
        if not save_event.is_set():
            time.sleep(0.01)
            continue
        
        last_idx += 1
        print(f"Capturing image to {args.save_path}/{last_idx}/image")
        rcc.start("image", False, f'{save_dir}/{last_idx}/image')
        save_event.clear()
        
finally:
    rcc.stop()