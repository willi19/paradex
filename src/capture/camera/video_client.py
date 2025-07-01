from paradex.io.camera.camera_loader import CameraManager
import time
from paradex.utils.file_io import find_latest_index, capture_path_list
import argparse
import os
from threading import Event
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.io.capture_pc.client import get_socket, register
import sys

def listen_for_commands():
    while True:
        ident, msg = socket.recv_multipart()
        msg = msg.decode()

        if msg == "quit":
            print(f"[Server] Received quit from client")
            stop_event.set()
            break

        else:
            print(f"[Server] Unknown command: {msg}")
            continue

socket = get_socket(5556)
client_ident = register(socket)

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

stop_event = Event()

try:
    camera = CameraManager("video")
except:
    socket.send_multipart([client_ident, b"camera_error"])
    sys.exit(1)
socket.send_multipart([client_ident, b"camera_ready"])

num_cam = camera.num_cameras

save_path = f"{args.save_path}/video"
camera.set_save_dir(save_path)
camera.start()

while not stop_event.is_set():
    time.sleep(0.01)
    
camera.end()
camera.quit()