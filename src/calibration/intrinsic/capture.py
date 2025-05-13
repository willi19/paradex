
import argparse
import os
import json
from paradex.utils.file_io import config_dir, home_path
import zmq
from paradex.io.capture_pc.connect import reset_and_run
import os
import threading
import sys
import time

parser = argparse.ArgumentParser(description="Capture intrinsic camera calibration.")
parser.add_argument(
    "--serial",
    type=str,
    required=True,
    help="Directory to save the video.",
)

pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
args = parser.parse_args()
serial_num = int(args.serial)

pc_name = None
for pc in pc_info.keys():
    if serial_num in pc_info[pc]['cam_list']:
        pc_name = pc
        break

if pc_name is None:
    raise ValueError(f"Serial number {serial_num} not found in PC list.")

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://*:5556")

reset_and_run(os.path.join("src/calibration/intrinsic/client.py"), [pc_name])  # 명령 수신 대기

def send_commands():
    while True:
        key = sys.stdin.read(1)
        if key == 'c':
            print("[Server] Sending command to all clients...")
            # ROUTER는 클라이언트 ident를 알아야 보낼 수 있음 → 저장해둬야 함
            socket.send_multipart([pc_name.encode(), b"capture"])

def receive_replies():
    while True:
        print("[Server] Waiting for message...")
        ident, reply = socket.recv_multipart()
        ident = ident.decode("utf-8")
        reply = reply.decode("utf-8")

        print(f"[Server] Received reply from {ident}: {reply}")
        print(f"[{pc_name}] Replied: {reply}")
        print(time.time())
            
threading.Thread(target=send_commands, daemon=True).start()
receive_replies()  # main thread는 계속 수신 담당

# Get serial number

# Connect pc with according serial number

# Get and plot current marker position

# Capture when it is inside the image
