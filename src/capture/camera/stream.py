from threading import Event
import argparse
import os
import time

from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.utils.file_io import find_latest_index
from paradex.utils.path import shared_dir
from paradex.utils.keyboard_listener import listen_keyboard

parser = argparse.ArgumentParser()
parser.add_argument('--sync_mode', default=False, action='store_true')
parser.add_argument('--frame_rate', default=30, type=int)

args = parser.parse_args()

stop_event = Event()
start_event = Event()
exit_event = Event()    

listen_keyboard({"q":exit_event, "c":start_event, "s":stop_event})

camera = CameraLoader() # video, image, stream


while not exit_event.is_set():
    if not start_event.is_set():
        time.sleep(0.01)
        continue

    camera.start("stream", args.sync_mode, fps=args.frame_rate)
    
    while not stop_event.is_set() and not exit_event.is_set():
        time.sleep(0.02)
        
    camera.stop()
    start_event.clear()
    stop_event.clear()

camera.end()
