import threading
import numpy as np
import cv2
import json
import time
import os

from paradex.utils.file_io import shared_dir
from paradex.utils.env import get_pcinfo, get_serial_list

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.util import get_client_socket
from paradex.io.capture_pc.connect import git_pull, run_script

from paradex.image.aruco import draw_charuco
from paradex.image.merge import merge_image

BOARD_COLORS = [
    (0, 0, 255), 
    (0, 255, 0)
]

pc_info = get_pcinfo()
serial_list = get_serial_list()

saved_corner_img = {serial_num:np.zeros((1536, 2048, 3), dtype=np.uint8) for serial_num in serial_list}
cur_state = {serial_num:(np.array([]), np.array([]), 0) for serial_num in serial_list}

capture_idx = 0
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

def listen_socket(pc_name, socket):
    while True:
        msg = socket.recv_string()
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            print(f"[{pc_name}] Non-JSON message: {msg}")
            continue
        
        if data.get("type") == "charuco":
            serial_num = data["serial_num"]
            result = data["detect_result"]
            corners = np.array(result["checkerCorner"], dtype=np.float32)
            ids = np.array(result["checkerIDs"], dtype=np.int32).reshape(-1, 1)
            frame = data["frame"]
            cur_state[serial_num] = (corners, ids, frame)                

        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/calibration/extrinsic/client.py --save_path {filename}", pc_list)

camera_controller = RemoteCameraController("stream", None, sync=False)
camera_controller.start_capture()

try:
    socket_dict = {name:get_client_socket(pc_info["ip"], 5564) for name, pc_info in pc_info.items()}

    for pc_name, sock in socket_dict.items():
        threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()
        
    while True:
        img_dict = {}
        for serial_num in serial_list:
            img = saved_corner_img[serial_num].copy()
            corners, ids, frame = cur_state[serial_num]
            if corners.shape[0] > 0:
                draw_charuco(img, corners[:,0], BOARD_COLORS[0], 5, -1, ids)
            img = cv2.putText(img, f"{serial_num} {frame}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 0), 12)
            img_dict[serial_num] = img   
        
        grid_image = merge_image(img_dict)
        grid_image = cv2.resize(grid_image, (int(2048//1.5), int(1536//1.5)))
        
        cv2.imshow("Grid", grid_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            for socket in socket_dict.values():
                socket.send_string("quit")
            break
        
        elif key == ord('c'):
            os.makedirs(os.path.join(shared_dir, "extrinsic", filename, str(capture_idx)), exist_ok=True)
            for serial_num in serial_list:
                corners, ids, frame = cur_state[serial_num]
                np.save(os.path.join(shared_dir, "extrinsic", filename, str(capture_idx), serial_num + "_cor.npy"), corners)
                np.save(os.path.join(shared_dir, "extrinsic", filename, str(capture_idx), serial_num + "_id.npy"), ids)
                if corners.shape[0] > 0:
                    draw_charuco(saved_corner_img[serial_num], corners[:,0], BOARD_COLORS[1], 5, -1, ids)
            capture_idx += 1

finally:
    camera_controller.end_capture()
    camera_controller.quit()        