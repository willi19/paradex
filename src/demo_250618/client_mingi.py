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
import cv2
from paradex.image.undistort import undistort_img

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

try:
    camera = CameraManager("stream", path=None, serial_list=None, syncMode=True)
except:
    socket.send_multipart([client_ident, b"camera_error"])
    sys.exit(1)
    
socket.send_multipart([client_ident, b"camera_ready"])
num_cam = camera.num_cameras

camera.start()
last_frame_ind = [-1 for _ in range(num_cam)]
save_flag = [False for _ in range(num_cam)]
save_finish = True

threading.Thread(target=listen_for_commands, daemon=True).start()

while not should_exit:
    intrinsic = json.load(open(f"{shared_dir}/demo_250618/pringles/0/cam_param/intrinsics.json", "r"))
    


    for i in range(num_cam):
        if camera.frame_num[i] == last_frame_ind[i]:
            continue

        last_frame_ind[i] = camera.frame_num[i]
        with camera.locks[i]:
            last_image = camera.image_array[i].copy()
        
        # cv2.imwrite(os.path.join(shared_dir, str(cur_filename), str(last_frame_ind[i]), "images", f"{camera.serial_list[i]}.jpg"), last_image)
        undistorted_img = undistort_img(last_image, intrinsic[cam_id])
        cv2.imwrite(os.path.join(shared_dir, str(cur_filename), str(last_frame_ind[i]), "images_undistorted", f"{camera.serial_list[i]}.jpg"), undistorted_img)
    
        serial_num = camera.serial_list[i]
        msg_dict = {
            "frame": int(last_frame_ind[i]),
            "type": "charuco",
            "serial_num": serial_num,
        }
        msg_json = json.dumps(msg_dict)

        if client_ident is not None:
            socket.send_multipart([client_ident, msg_json.encode()])
    
    time.sleep(0.001)

camera.end()
camera.quit()

if client_ident is not None:
    socket.send_multipart([client_ident, b"terminate"])