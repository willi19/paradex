import zmq
from paradex.utils.file_io import config_dir, shared_dir
import json
import os
from paradex.io.camera.camera_loader import CameraManager
import time
from paradex.image.aruco import detect_charuco, merge_charuco_detection
import threading
import numpy as np
import sys

from yolo_world_module import YOLO_MODULE
print("yolo module loaded")

should_exit = False 
client_ident = None 
current_index = 0 
cur_filename = "asdf"

def listen_for_commands():
    global should_exit, client_ident
    while True:
        ident, msg = socket.recv_multipart()
        msg = msg.decode()

        if msg == "quit":
            print(f"[Server] Received quit from client")
            should_exit = True
            break

        elif msg.startswith("capture"):
            global current_index, num_cam, save_flag, save_finish
            _, index = msg.split(":")
            index = int(index)
            for i in range(num_cam):
                save_flag[i] = True
            current_index = index
            save_finish = False
        
        else:
            print(f"[Server] Unknown command: {msg}")
            continue
        

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://*:5556")

ident, msg = socket.recv_multipart()
msg = msg.decode()

if msg == "register":
    client_ident = ident  # ← bytes 그대로 저장
    socket.send_multipart([client_ident, b"registered"])
    print(f"[Server] Client registered: {ident.decode()}")

else:
    sys.exit(1)

ident, msg = socket.recv_multipart()
msg = msg.decode()

if msg.startswith("filename"):
    _, cur_filename = msg.split(":")

else:
    sys.exit(1)

board_info = json.load(open(os.path.join(config_dir, "environment", "charuco_info.json"), "r"))

try:
    camera = CameraManager("stream", path=None, serial_list=None, syncMode=False)
except:
    socket.send_multipart([client_ident, b"camera_error"])
    sys.exit(1)
    
socket.send_multipart([client_ident, b"camera_ready"])
num_cam = camera.num_cameras

camera.start()
last_frame_ind = [-1 for _ in range(num_cam)]
save_flag = [False for _ in range(num_cam)]
save_finish = True

yolo_module = YOLO_MODULE(categories="pringles")
socket.send_multipart([client_ident, b"yolo module initialized"])

threading.Thread(target=listen_for_commands, daemon=True).start()

while not should_exit:
    print("waiting for camera frame")
    for i in range(num_cam):
        if camera.frame_num[i] == last_frame_ind[i]:
            continue

        last_frame_ind[i] = camera.frame_num[i]
        with camera.locks[i]:
            last_image = camera.image_array[i].copy()
        print(last_frame_ind[i])

        
        detections = yolo_module.process_img(last_image, with_segmentation=False)
        if detections.xyxy.size > 0:
            bbox = detections.xyxy[0]
            # detections.bbox_center = bbox[:2] + (bbox[2:] - bbox[:2]) / 2
            detections.mask = np.zeros((1, last_image.shape[0], last_image.shape[1]), dtype=bool)
            detections.mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = True

        serial_num = camera.serial_list[i]

        save_flag[i] = False

        msg_dict = {
            "frame": int(last_frame_ind[i]),
            "detect_result": detections,
            "type": "demo",
            "serial_num": serial_num,
        }
        print(detections)
        msg_json = json.dumps(msg_dict)

        if client_ident is not None:
            socket.send_multipart([client_ident, msg_json.encode()])
    
    all_saved = True
    for i in range(num_cam):
        if save_flag[i]:
            all_saved = False
            break
    if all_saved and not save_finish:
        socket.send_multipart([client_ident, b"save_finish"])
        save_finish = True

    time.sleep(0.01)

camera.end()
camera.quit()

if client_ident is not None:
    socket.send_multipart([client_ident, b"terminate"])