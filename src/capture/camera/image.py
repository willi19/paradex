from paradex.io.camera.camera_loader import CameraManager
from paradex.utils.file_io import find_latest_directory, shared_dir
from paradex.utils.keyboard_listener import listen_keyboard
from threading import Event
import argparse
import os
import time

camera = CameraManager("image")
num_cam = camera.num_cameras

stop_event = Event()
save_event = Event()

listen_keyboard({"q":stop_event, "c":save_event})

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)

args = parser.parse_args()

save_dir = os.path.join(shared_dir, args.save_path)
last_idx = find_latest_directory(save_dir) if os.path.exists(save_dir) else -1

while not stop_event.is_set():
    if not save_event.is_set():
        time.sleep(0.01)
        continue
    
    save_path = f"{save_dir}/{last_idx}/image"
    camera.set_save_dir(save_path)

    camera.start()
    camera.wait_for_capture_end()
    save_event.clear()

camera.quit()