from threading import Event
import argparse
import os
import time

from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.utils.file_io import find_latest_index
from paradex.utils.path import shared_dir
from paradex.utils.keyboard_listener import listen_keyboard

camera = CameraLoader() # video, image, stream
stop_event = Event()
save_event = Event()

listen_keyboard({"q":stop_event, "c":save_event})

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)

args = parser.parse_args()

save_dir = os.path.join("shared_data", args.save_path)

last_idx = int(find_latest_index(os.path.join(shared_dir, args.save_path)))

while not stop_event.is_set():
    if not save_event.is_set():
        time.sleep(0.01)
        continue
    last_idx += 1
    save_path = f"{save_dir}/{last_idx}/raw"
    
    print(f"Capturing image to {save_path}")
    camera.start("image", False, save_path=save_path)
    camera.stop()
    save_event.clear()

camera.end()