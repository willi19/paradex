import datetime
import os
import argparse

from paradex.dataset_acqusition.capture import CaptureSession


parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', type=str, default=None)
parser.add_argument('--hand', type=str, default=None)
parser.add_argument('--name', type=str, required=True)

args = parser.parse_args()

cs = CaptureSession(
    camera=True,
    arm=args.arm,
    hand=args.hand,
    teleop=args.device
)

name = args.name
while True:
    index = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Prepare to record new session:", name)
    state = cs.teleop()
    if state == "exit":
        break
    
    cs.start(os.path.join("capture", "miyungpa", args.name, index))
    print("Starting new recording session:", name)
    state = cs.teleop()
    cs.stop()
    print("Stopped recording session:", name)
    
    if state == "exit":
        break

print("Exiting teleoperation recording.")
cs.end()