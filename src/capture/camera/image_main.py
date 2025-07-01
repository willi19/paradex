from threading import Event, Thread
import json
import time
import zmq
import os
from paradex.utils.file_io import config_dir, shared_dir
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.keyboard_listener import listen_keyboard
import argparse

# === SETUP ===
pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
serial_list = []
for pc in pc_info.keys():
    serial_list.extend(pc_info[pc]['cam_list'])

context = zmq.Context()
socket_dict = {}
terminate_dict = {pc: False for pc in pc_info.keys()}
start_dict = {pc: False for pc in pc_info.keys()}

capture_idx = 0
capture_state = {pc: False for pc in pc_info.keys()}

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

def listen_socket(pc_name, socket):
    while True:
        msg = socket.recv_string()
        if msg == "terminated":
            terminate_dict[pc_name] = True
            print(f"[{pc_name}] Terminated.")
            break
        
        elif msg == "camera_ready":
            start_dict[pc_name] = True
            print(f"[{pc_name}] Camera ready.")
            continue
        
        elif msg == "camera_error":
            print(f"[{pc_name}] Camera error.")
            terminate_dict[pc_name] = True
            continue

        elif msg == "save_finish":
            capture_state[pc_name] = False
            print(f"[{pc_name}] Save finished.")
            continue

        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

def wait_for_camera_ready():
    while True:
        all_ready = True
        for pc_name, ready in start_dict.items():
            if not ready:
                all_ready = False
                break
        if all_ready:
            break
        time.sleep(0.1)
        
def wait_for_capture():
    while True:
        all_captured = True
        for pc_name, in_capture in capture_state.items():
            if in_capture:
                all_captured = False
                break
        if all_captured:
            break
        time.sleep(0.1)

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/capture/camera/image_client.py --save_path {args.save_path}", pc_list)

stop_event = Event()
save_event = Event()

listen_keyboard({"c":save_event, "q":stop_event})
try:
    for pc_name, info in pc_info.items():
        ip = info["ip"]
        sock = context.socket(zmq.DEALER)
        sock.identity = b"server"
        sock.connect(f"tcp://{ip}:5556")
        socket_dict[pc_name] = sock

    for pc_name, info in pc_info.items():
        socket_dict[pc_name].send_string("register")
        if socket_dict[pc_name].recv_string() == "registered":
            print(f"[{pc_name}] Registered.")
        
    # Start per-socket listener
    for pc_name, sock in socket_dict.items():
        Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()
    wait_for_camera_ready()
    
    while not stop_event.is_set():
        if not save_event.is_set():
            time.sleep(0.01)
            continue
        send_capture = True
        for pc in pc_info.keys():
            if capture_state[pc]:
                send_capture = False
                break
            
        if send_capture:
            os.makedirs(os.path.join(shared_dir, args.save_path, str(capture_idx), "image"), exist_ok=True)
            for pc, socket in socket_dict.items():
                socket.send_string(f"capture:{capture_idx}")
                capture_state[pc] = True
            capture_idx += 1
        wait_for_capture()
        save_event.clear()

    for pc_name, sock in socket_dict.items():
        sock.send_string("quit")
        sock.close()
        
except Exception as e:
    print(e)
    for pc_name, sock in socket_dict.items():
        sock.send_string("quit")
        sock.close()


    