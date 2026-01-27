import datetime
import os
import argparse

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_index



parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', type=str, default=None)
parser.add_argument('--hand', type=str, default=None)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--tactile', action='store_true', help='Whether to record tactile data from the Inspire hand.')
parser.add_argument('--ip', action='store_true', help='Use IP connection for Inspire hand controller.')

args = parser.parse_args()

cs = CaptureSession(
    camera=True,
    arm=args.arm,
    hand=args.hand,
    teleop=args.device,
    tactile=args.tactile,
    ip=args.ip
)

name = args.name

last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", "hri_inspire_left", args.name)))

while True:
    # index = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")   
    last_idx += 1

    print("Prepare to record new session:", name, "episode:", last_idx)
    state = cs.teleop()
    if state == "exit":
        break
    
    cs.start(os.path.join("capture", "hri_inspire_left", args.name, str(last_idx)))
    print("Starting new recording session:", name)
    print("Capturing index:", last_idx)
    state = cs.teleop()
    cs.stop()
    print("Stopped recording session:", name)
    
    if state == "exit":
        break

print("Exiting teleoperation recording.")
cs.end()