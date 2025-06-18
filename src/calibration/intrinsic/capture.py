
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

BOARD_COLORS = [
    (255, 0, 0),     # 빨강
    (0, 255, 0),     # 초록
    (0, 0, 255),     # 파랑
    (255, 255, 0),   # 시안
    (255, 0, 255),   # 마젠타
    (0, 255, 255),   # 노랑
    (128, 128, 255)  # 연보라
]

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

threading.Thread(target=wait_for_keypress, args=(socket,), daemon=True).start()  
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

saved_image = np.zeros((1536, 2048, 3), dtype=np.uint8)
while True:
    msg = socket.recv_string()
    if msg == "terminated":
        print("[Client] Quitting...")
        break
    else:
        data = json.loads(msg)
        plot_img = saved_image.copy()
        if data.get("type") == "charuco":
            for board_id, result in data["detect_result"].items():
                corners = np.array(result["checkerCorner"], dtype=np.float32)
                ids = np.array(result["checkerIDs"], dtype=np.int32).reshape(-1, 1)

                save = result["save"]
                if save:
                    saved_board_corners.append(corners)
                    draw_charuco_corners_custom(saved_image, corners, BOARD_COLORS[0], 5, -1, ids)

                if corners.shape[0] == 0:
                    print("[Client] No corners detected.")
                    continue
            
                draw_charuco_corners_custom(plot_img, corners, BOARD_COLORS[int(board_id)], 5, -1, ids)

            # for saved_corner in saved_board_corners:
            #     draw_charuco_corners_custom(image, saved_corner, BOARD_COLORS[0], 5, -1)

            plot_img = cv2.resize(plot_img, (2048 // 2, 1536 // 2))
            cv2.imshow("Charuco Detection", plot_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                socket.send_string("quit")
                break
        else:
            print(f"[Client] Unknown JSON type: {data.get('type')}")

socket.send_string("quit")
        