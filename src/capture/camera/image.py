from threading import Event
import argparse
import os
import time
import datetime

from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.utils.file_io import find_latest_directory
from paradex.utils.path import shared_dir
from paradex.utils.keyboard_listener import listen_keyboard

stop_event = Event()
save_event = Event()

listen_keyboard({"q":stop_event, "c":save_event})

camera = CameraLoader()

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)

args = parser.parse_args()

save_dir = os.path.join(shared_dir, args.save_path)

while not stop_event.is_set():
    if not save_event.is_set():
        time.sleep(0.01)
        continue
    
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    save_path = f"{save_dir}/{date_str}"

    camera.start("image", False, save_path=save_path)
    camera.stop()
    
camera.end()