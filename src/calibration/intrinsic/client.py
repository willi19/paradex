from paradex.utils.file_io import config_dir, shared_dir
import json
import os
from paradex.io.capture_pc.camera_local import CameraCommandReceiver
from paradex.io.capture_pc.util import get_server_socket

import time
from paradex.image.aruco import detect_charuco
import numpy as np

board_corner_list = []
board_info = json.load(open(os.path.join(config_dir, "environment", "charuco_info.json"), "r"))

def should_save(result):
    corner = result["checkerCorner"]
    ids = result["checkerIDs"]
    
    if len(ids) != 70:
        return False
    
    for board_corner in board_corner_list:
        dist = np.linalg.norm(corner - board_corner, axis=1).mean()
        if dist < 10:
            return False

    return True

camera_loader = CameraCommandReceiver()
last_frame = -1

ident = camera_loader.ident
serial_num = camera_loader.serial_list[0]
socket = get_server_socket(5564)

while not camera_loader.exit:
    frame_id = camera_loader.camera.get_frameid(0)
    if frame_id == last_frame:
        continue
        
    data = camera_loader.camera.get_data(0)
    last_frame = data["frameid"]
    last_image = data["image"]
        
    detect_result = detect_charuco(last_image, board_info)

    for board_id, result in detect_result.items():
        is_save = should_save(result)
        if is_save:
            board_corner_list.append(result["checkerCorner"])
        detect_result[board_id]["save"] = is_save
    
        for data_name in ["checkerCorner", "checkerIDs"]:
            detect_result[board_id][data_name] = detect_result[board_id][data_name].tolist()
    
    msg_dict = {
        "type": "charuco",
        "frame": int(last_frame),
        "detect_result": detect_result
    }
    msg_json = json.dumps(msg_dict)
    socket.send_multipart([ident, msg_json.encode()])

    time.sleep(0.01)

board_corner_list = np.array(board_corner_list)
datetime_str = time.strftime("%Y%m%d_%H%M%S")

os.makedirs(os.path.join(shared_dir, "intrinsic", serial_num, "keypoint"), exist_ok=True)
np.save(os.path.join(shared_dir, "intrinsic", serial_num, "keypoint", datetime_str + ".npy"), board_corner_list)
