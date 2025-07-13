import json
import os
import time

from paradex.utils.file_io import config_dir
from paradex.image.aruco import detect_charuco, merge_charuco_detection
from paradex.io.capture_pc.util import get_server_socket
from paradex.io.capture_pc.camera_local import CameraCommandReceiver

camera_loader = CameraCommandReceiver()

ident = camera_loader.ident
socket = get_server_socket(5564)

board_info = json.load(open(os.path.join(config_dir, "environment", "charuco_info.json"), "r"))
serial_list = camera_loader.camera.serial_list
num_cam = len(serial_list)
last_frame_ind = [-1 for _ in range(num_cam)]

while not camera_loader.exit:
    for i, serial_num in enumerate(serial_list):
        frame_id = camera_loader.camera.get_frameid(i)
        
        if frame_id == last_frame_ind[i]:
            continue
        
        data = camera_loader.camera.get_data(i)
        last_frame_ind[i] = data["frameid"]
        last_image = data["image"]
        
        detect_result = detect_charuco(last_image, board_info)
        merged_detect_result = merge_charuco_detection(detect_result, board_info)
        
        for data_name in ["checkerCorner", "checkerIDs"]:
            merged_detect_result[data_name] = merged_detect_result[data_name].tolist()

        msg_dict = {
            "frame": int(last_frame_ind[i]),
            "detect_result": merged_detect_result,
            "type": "charuco",
            "serial_num": serial_num,
        }
        msg_json = json.dumps(msg_dict)
        socket.send_multipart([ident, msg_json.encode()])
        
    time.sleep(0.01)