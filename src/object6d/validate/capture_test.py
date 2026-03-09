import datetime
import os
import argparse
from threading import Event
import time

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir
from paradex.calibration.utils import save_current_camparam, save_current_C2R

parser = argparse.ArgumentParser()

parser.add_argument('--arm', type=str, default=None)
parser.add_argument('--hand', type=str, default=None)
parser.add_argument('--name', type=str)

args = parser.parse_args()

stop_event = Event()
save_event = Event()
exit_event = Event()

listen_keyboard({"c": save_event, "q": exit_event, "s": stop_event})

# Initialize robot data capture (arm/hand)
cs = CaptureSession(
    camera=False,
    arm=args.arm,
    hand=args.hand,
    hand_ip=True
)

# Initialize image capture
rcc = remote_camera_controller("image_main.py")

name = args.name
try:
    while not exit_event.is_set():
        if not save_event.is_set():
            stop_event.clear()
            continue

        index = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(name, index)

        # Start robot data recording
        cs.start(save_path)
        print("Starting new recording session:", name)

        # Capture initial image with camera parameters
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_current_camparam(os.path.join(shared_dir, save_path))
        print(f"Capturing camera parameters to {save_path}")
        save_current_C2R(os.path.join(shared_dir, save_path))
        print(f"Capturing C2R to {save_path}")
        print(f"Capturing image to {save_path}")
        rcc.start("image", False, f'shared_data/{save_path}/raw')
        rcc.stop()

        cs.stop()
        print("Stopped recording session:", name)
        save_event.clear()
        stop_event.clear()

finally:
    print("Exiting teleoperation recording.")
    cs.end()
    rcc.end()