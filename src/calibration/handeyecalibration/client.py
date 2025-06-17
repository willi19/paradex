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
            print(f"[Server] Capture command received, index: {index}")
        
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

# board_info = json.load(open(os.path.join(config_dir, "environment", "charuco_info.json"), "r"))

try:
    camera = CameraManager("image", path=None, serial_list=None)
except:
    socket.send_multipart([client_ident, b"camera_error"])
    sys.exit(1)
    
socket.send_multipart([client_ident, b"camera_ready"])
num_cam = camera.num_cameras

threading.Thread(target=listen_for_commands, daemon=True).start()

last_frame_ind = [-1 for _ in range(num_cam)]
save_flag = [False for _ in range(num_cam)]
save_finish = True

while not should_exit:
    if not save_finish:
        time.sleep(0.1)
        continue
    
    camera.set_save_dir(os.path.join(shared_dir, "handeye_calibration", cur_filename, str(current_index)))
    camera.start()
    camera.wait_for_capture_end()

    socket.send_multipart([client_ident, b"save_finish"])
    print(current_index, "save finish")
    save_finish = True

    time.sleep(0.01)

camera.end()
camera.quit()

if client_ident is not None:
    socket.send_multipart([client_ident, b"terminate"])