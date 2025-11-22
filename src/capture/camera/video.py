from threading import Event
import argparse
import os
import time

from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.utils.file_io import find_latest_index
from paradex.utils.path import shared_dir
from paradex.utils.keyboard_listener import listen_keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
parser.add_argument('--sync_mode', default=False, action='store_true')
parser.add_argument('--frame_rate', default=30, type=int)

args = parser.parse_args()

stop_event = Event()
save_event = Event()
exit_event = Event()    

listen_keyboard({"q":exit_event, "c":save_event, "s":stop_event})

last_idx = int(find_latest_index(os.path.join(shared_dir, args.save_path)))

camera = CameraLoader() # video, image, stream


while not exit_event.is_set():
    if not save_event.is_set():
        time.sleep(0.01)
        continue
    last_idx += 1
    save_path = f"{args.save_path}/{last_idx}/raw"

    os.makedirs(os.path.join(shared_dir, save_path), exist_ok=True)
    camera.start("video", args.sync_mode, save_path=save_path, frame_rate=args.frame_rate)
    print(f"Capturing video to {save_path}")
    
    while not stop_event.is_set() and not exit_event.is_set():
        time.sleep(0.02)
        
    camera.stop()
    save_event.clear()
    stop_event.clear()

camera.end()
