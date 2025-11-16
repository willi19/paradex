from threading import Event
import time
import argparse
import os

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.file_io import find_latest_index, shared_dir

rcc = remote_camera_controller("stream_main.py")

stop_event = Event()
save_event = Event()

listen_keyboard({"c":save_event, "q":stop_event})

try:
    while not stop_event.is_set():
        
        if not save_event.is_set():
            time.sleep(0.01)
            continue
        
        rcc.start("stream", False, fps=10)
        tmp = input("Press Enter to stop streaming...")
        rcc.stop()
        save_event.clear()
        
finally:
    rcc.end()