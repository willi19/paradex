from paradex.io.camera.camera_loader import CameraManager
import time
from paradex.utils.file_io import find_latest_index, capture_path_list
import argparse
import os
from threading import Event
from paradex.utils.keyboard_listener import listen_keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

stop_event = Event()
listen_keyboard({"q":stop_event})

save_dir = os.path.join(capture_path_list[0], args.save_path)
last_idx = find_latest_index(save_dir) if os.path.exists(save_dir) else -1

camera = CameraManager("video", syncMode=True)
num_cam = camera.num_cameras

save_path = f"{args.save_path}/{last_idx+1}/video"
camera.set_save_dir(save_path)
camera.start()

while not stop_event.is_set():
    time.sleep(0.01)
    
camera.end()
camera.quit()