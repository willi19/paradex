import datetime
import os
import argparse
from threading import Event
import time

from paradex.utils.file_io import find_latest_index
from paradex.utils.path import shared_dir
from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.keyboard_listener import listen_keyboard

parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', type=str, default=None)
parser.add_argument('--hand', type=str, default=None)
parser.add_argument('--name', type=str)

args = parser.parse_args()

stop_event = Event()
save_event = Event()
exit_event = Event()

listen_keyboard({"c": save_event, "q": exit_event, "s": stop_event})
cs = CaptureSession(
    camera=True
)

name = args.name


last_idx = int(find_latest_index(os.path.join(shared_dir, args.name)))
while not exit_event.is_set():
    if not save_event.is_set():
        stop_event.clear()
        continue
    
    # index = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    last_idx += 1
    
    cs.start(os.path.join("capture", "hri", name, str(last_idx)))
    print("Starting new recording session:", name)
    while not stop_event.is_set() and not exit_event.is_set():
        time.sleep(0.02)
        
    cs.stop()
    print("Stopped recording session:", name)

    save_event.clear()
    stop_event.clear()

print("Exiting teleoperation recording.")
cs.end()