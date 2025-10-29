from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.utils.file_io import find_latest_directory, shared_dir
from paradex.utils.keyboard_listener import listen_keyboard
from threading import Event
import argparse
import os
import time

camera = CameraLoader() # video, image, stream

stop_event = Event()
save_event = Event()

listen_keyboard({"q":stop_event, "c":save_event})

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)

args = parser.parse_args()

save_dir = os.path.join(shared_dir, args.save_path)
last_idx = int(find_latest_directory(save_dir)) if os.path.exists(save_dir) else 0

while not stop_event.is_set():
    if not save_event.is_set():
        time.sleep(0.01)
        continue
    last_idx += 1
    save_path = f"{save_dir}/{last_idx}"

    camera.start("image", False, save_path=save_path)
    camera.stop()
camera.end()