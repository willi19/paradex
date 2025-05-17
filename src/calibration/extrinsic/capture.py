
import argparse
import os
import json
from paradex.utils.file_io import config_dir, home_path, shared_dir
import zmq
from paradex.io.capture_pc.connect import git_pull, run_script
import os
import threading
import sys
import time
import numpy as np
import cv2

BOARD_COLORS = [
    (255, 0, 0),     # 빨강
    (0, 255, 0),     # 초록
    (0, 0, 255),     # 파랑
    (255, 255, 0),   # 시안
    (255, 0, 255),   # 마젠타
    (0, 255, 255),   # 노랑
    (128, 128, 255)  # 연보라
]
current_index = 0
save_finish = True

def wait_for_keypress(socket_dict):
    while True:
        key = sys.stdin.read(1)
        for pc_name, socket in socket_dict.items():
            if key == 'q':
                print("[Server] Quitting...")
                socket.send_string("quit")
                break
            
            if key == 'c':
                global current_index, save_finish

                print("[Server] Capture command received.")
                if not save_finish:
                    print("[Server] Capture command already in progress.")
                    continue
                
                socket.send_string(f"capture:{current_index}")
                current_index += 1

pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))

for pc_name in pc_info.keys():
    git_pull("merging", [pc_name]) 
    print(f"[{pc_name}] Git pull complete.")

for pc_name in pc_info.keys():
    run_script(os.path.join(f"python src/calibration/extrinsic/client.py"), [pc_name])  # 명령 수신 대기

print(f"[{pc_name}] Client script started.")

socket_dict = {}
for pc_name in pc_info.keys():
    ip = pc_info[pc_name]["ip"]
    context = zmq.Context()
    socket_dict[pc_name] = context.socket(zmq.DEALER)
    socket_dict[pc_name].identity = b"server"
    socket_dict[pc_name].connect(f"tcp://{ip}:5556")  # 서버 IP로 연결

    socket_dict[pc_name].send_string("register")
    msg = socket_dict[pc_name].recv_string()

    if msg == "registered":
        print("[Client] Registration complete, starting detection loop...")
    else:
        print("[Client] Registration failed.")
    filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    socket_dict[pc_name].send_string("filename:"+filename)

threading.Thread(target=wait_for_keypress, args=(socket_dict,), daemon=True).start()  
saved_board_corners = []

def draw_charuco_corners_custom(image, corners, color=(0, 255, 255), radius=4, thickness=2, ids=None):

    for i in range(len(corners)):
        corner = tuple(int(x) for x in corners[i][0])
        cv2.circle(image, corner, radius, color, thickness)

        if ids is not None:
            cv2.putText(
                image,
                str(int(ids[i])),
                (corner[0] + 5, corner[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                lineType=cv2.LINE_AA
            )

while True:
    msg = socket.recv_string()
    if msg == "terminated":
        print("[Client] Quitting...")
        break
    else:
        data = json.loads(msg)
        if data.get("type") == "charuco":
            image = np.zeros((1536, 2048, 3), dtype=np.uint8)
            for board_id, result in data["detect_result"].items():
                corners = np.array(result["checkerCorner"], dtype=np.float32)
                ids = np.array(result["checkerIDs"], dtype=np.int32).reshape(-1, 1)

                save = result["save"]
                if save:
                    saved_board_corners.append(corners)
                if corners.shape[0] == 0:
                    print("[Client] No corners detected.")
                    continue

                draw_charuco_corners_custom(image, corners, BOARD_COLORS[int(board_id)], 5, -1, ids)

            for saved_corner in saved_board_corners:
                draw_charuco_corners_custom(image, saved_corner, BOARD_COLORS[0], 5, -1)

            cv2.imshow("Charuco Detection", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                socket.send_string("quit")
                break
        else:
            print(f"[Client] Unknown JSON type: {data.get('type')}")

socket.send_string("quit")
        