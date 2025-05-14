import zmq
from paradex.utils.file_io import config_dir
import json
import os
from paradex.io.camera.camera_loader import CameraManager
import cv2
import time
import argparse
from paradex.image.aruco import detect_charuco
import threading
import numpy as np

should_exit = False  # 공유 변수로 종료 제어
client_ident = None  # 메인 PC에서 온 ident 저장용
board_corner_list = []

def listen_for_commands():
    global should_exit, client_ident
    while True:
        ident, msg = socket.recv_multipart()
        msg = msg.decode()

        if msg == "register":
            client_ident = ident  # ← bytes 그대로 저장
            socket.send_multipart([client_ident, b"registered"])
            print(f"[Server] Client registered: {ident.decode()}")
        elif msg == "quit":
            print(f"[Server] Received quit from client")
            should_exit = True
            break

def should_save(result):
    corner = result["checkerCorner"]
    ids = result["markerIDs"]
    
    if len(ids) != 70:
        return False
    
    for board_corner in board_corner_list:
        dist = np.linalg.norm(corner - board_corner, axis=1).sum()
        if dist < 0.1:
            return False

    return True

parser = argparse.ArgumentParser(description="Capture intrinsic camera calibration.")
parser.add_argument(
    "--serial",
    type=str,
    required=True,
    help="Directory to save the video.",
)

args = parser.parse_args()
serial_num = args.serial

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://*:5556")

board_info = json.load(open(os.path.join(config_dir, "environment", "charuco_info.json"), "r"))

camera = CameraManager("stream", path=None, serial_list=[serial_num], syncMode=False)
num_cam = camera.num_cameras

camera.start()
last_frame = -1
last_image = None

selected_frame = []

threading.Thread(target=listen_for_commands, daemon=True).start()
while not should_exit:
    if camera.frame_num[0] != last_frame:
        last_frame = camera.frame_num[0]
        with camera.locks[0]:
            last_image = camera.image_array[0].copy()
        detect_result = detect_charuco(last_image, board_info)

        for board_id, result in detect_result.items():
            is_save = should_save(result)
            if is_save:
                board_corner_list.append(result["checkerCorner"])
            detect_result[board_id]["save"] = is_save
        
            for data_name in detect_result[board_id].keys():
                detect_result[board_id][data_name] = detect_result[board_id][data_name].tolist()
        # print(detect_result)
        msg_dict = {
            "frame": int(last_frame),
            "detect_result": detect_result,
            # "image": last_image.tolist(),
            "type": "charuco"
        }
        msg_json = json.dumps(msg_dict)

        if client_ident is not None:
            socket.send_multipart([client_ident, msg_json.encode()])

        time.sleep(0.01)

camera.end()
camera.quit()

if client_ident is not None:
    socket.send_multipart([client_ident, b"terminate"])