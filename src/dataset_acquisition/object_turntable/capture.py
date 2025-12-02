import datetime
import os
import argparse
from threading import Event
import time

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.keyboard_listener import listen_keyboard

parser = argparse.ArgumentParser()

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
while not exit_event.is_set():
    if not save_event.is_set():
        stop_event.clear()
        continue
    
    index = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    cs.start(os.path.join("capture", "object_turntable", name, index))
    print("Starting new recording session:", name)
    while not stop_event.is_set() and not exit_event.is_set():
        time.sleep(0.02)
        
    cs.stop()
    print("Stopped recording session:", name)
    save_event.clear()
    stop_event.clear()

print("Exiting teleoperation recording.")
cs.end()