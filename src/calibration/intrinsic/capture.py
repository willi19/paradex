
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

def get_pc_info(serial_num):
    pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))

    pc_name = None
    for pc in pc_info.keys():
        if serial_num in pc_info[pc]['cam_list']:
            pc_name = pc
            break

    if pc_name is None:
        raise ValueError(f"Serial number {serial_num} not found in PC list.")
    
    return pc_name, pc_info[pc_name]['ip']

def wait_for_keypress(socket):
    while True:
        key = sys.stdin.read(1)
        if key == 'q':
            print("[Server] Quitting...")
            socket.send_string("quit")
            break
            
parser = argparse.ArgumentParser(description="Capture intrinsic camera calibration.")
parser.add_argument(
    "--serial",
    type=str,
    required=True,
    help="Directory to save the video.",
)

args = parser.parse_args()
serial_num = args.serial

pc_name, ip = get_pc_info(serial_num)
print("PC Name:", pc_name)

git_pull("merging", [pc_name]) 
print(f"[{pc_name}] Git pull complete.")

run_script(os.path.join(f"python src/calibration/intrinsic/client.py --serial {serial_num}"), [pc_name])  # 명령 수신 대기
print(f"[{pc_name}] Client script started.")

context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.identity = b"server"
socket.connect(f"tcp://{ip}:5556")  # 서버 IP로 연결

socket.send_string("register")
msg = socket.recv_string()

if msg == "registered":
    print("[Client] Registration complete, starting detection loop...")
else:
    print("[Client] Registration failed.")

threading.Thread(target=wait_for_keypress, args=(socket,), daemon=True).start()  # 키 입력 대기 스레드
while True:
    msg = socket.recv_string()
    if msg == "terminated":
        print("[Client] Quitting...")
        break
    else:
        data = json.loads(msg)
        if data.get("type") == "charuco":
            corners = np.array(data["corners"], dtype=np.float32)
            ids = np.array(data["ids"], dtype=np.int32).reshape(-1, 1)

            if corners.shape[0] == 0:
                print("[Client] No corners detected.")
                continue

            # 기본 크기의 검정 배경 이미지 (예: 1280x1080)
            image = np.zeros((1536, 2048, 3), dtype=np.uint8)

            # 코너 시각화
            cv2.aruco.drawDetectedCornersCharuco(image, corners, ids)

            # 시각화
            cv2.imshow("Charuco Detection", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Client] Local quit requested.")
                socket.send_string("quit")
                break
        else:
            print(f"[Client] Unknown JSON type: {data.get('type')}")
        