
import argparse
import os
import json
from paradex.utils.file_io import config_dir, home_path
import zmq
from paradex.io.capture_pc.connect import git_pull, run_script
import os
import threading
import sys
import time
import numpy as np
import cv2

send_time = time.time()

def wait_for_keypress(socket):
    while True:
        key = sys.stdin.read(1)
        if key == 'q':
            print("[Server] Quitting...")
            socket.send_string("quit")
            break

        if key == 'c':
            print("[Server] Capturing image...")
            socket.send_string("capture")
            send_time = time.time()
            

pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
pc_name = pc_info.keys()[0]
ip = pc_info[pc_name]["ip"]

print("PC Name:", pc_name)

git_pull("merging", [pc_name]) 
print(f"[{pc_name}] Git pull complete.")

run_script(os.path.join(f"python src/debug/io/ping_test.py"), [pc_name])  # 명령 수신 대기
print(f"[{pc_name}] Client script started.")

context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.identity = b"server"
socket.connect(f"tcp://{ip}:5556")  # 서버 IP로 연결

ind = 0
send_time_list = []
recv_time_list = []

for i in range(100):
    socket.send_string(f"send_{ind}")
    send_time_list.append(time.time())
    msg = socket.recv_string()
    recv_time_list.append(time.time())

    if msg != f"send_{ind}":
        print(f"[Server] Error: {msg}")
        break

    else:
        ind += 1
        print(f"[Server] Received: {recv_time_list[-1] - send_time_list[-1]:.4f} seconds")

socket.send_string("quit")
        