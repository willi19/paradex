from paradex.io.camera.camera_loader import CameraManager
from paradex.utils.file_io import shared_dir
from threading import Event, Thread
import argparse
import os
import sys
from paradex.io.capture_pc.client import get_socket, register
import time

def listen_for_commands():
    while True:
        ident, msg = socket.recv_multipart()
        msg = msg.decode()

        if msg == "quit":
            print(f"[Server] Received quit from client")
            stop_flag.set()
            break

        elif msg.startswith("capture"):
            global current_index
            _, index = msg.split(":")
            index = int(index)
            save_flag.set()
            current_index = index
        
        else:
            print(f"[Server] Unknown command: {msg}")
            continue

socket = get_socket(5556)
client_ident = register(socket)
current_index = 0

stop_flag = Event()
save_flag = Event()

Thread(target=listen_for_commands, daemon=True).start()
try:
    camera = CameraManager("image")
except:
    socket.send_multipart([client_ident, b"camera_error"])
    sys.exit(1)
socket.send_multipart([client_ident, b"camera_ready"])
    
num_cam = camera.num_cameras

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()
save_path = os.path.join(shared_dir, args.save_path)

while not stop_flag.is_set():
    if not save_flag.is_set():
        time.sleep(0.01)
        continue
    save_path = f"{save_path}/{current_index}/image"
    camera.set_save_dir(save_path)

    camera.start()
    camera.wait_for_capture_end()
    save_flag.clear()
    
camera.quit()