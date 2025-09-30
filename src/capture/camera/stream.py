from paradex.io.camera.camera_loader import CameraManager
import time
from paradex.utils.file_io import find_latest_index, capture_path_list
import argparse
import os
from threading import Event
from paradex.utils.keyboard_listener import listen_keyboard

stop_event = Event()
listen_keyboard({"q":stop_event})

camera = CameraManager("stream", syncMode=False) # SyncMode = True : follow signal generator
num_cam = camera.num_cameras

camera.start() # captures1/save_path captures2/save_path 

while not stop_event.is_set():
    time.sleep(0.01)
    
camera.end()
camera.quit()