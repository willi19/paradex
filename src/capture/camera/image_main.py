from threading import Event, Thread
import time
import argparse
import os

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.file_io import find_latest_index, shared_dir

# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

pc_list = list(pc_info.keys()) # list of capture pc
git_pull("paradex2", pc_list)
run_script(f"python src/capture/camera/image_client.py", pc_list)

camera_loader = RemoteCameraController("image", None, debug=True)

stop_event = Event()
save_event = Event()

listen_keyboard({"c":save_event, "q":stop_event})

save_dir = os.path.join(shared_dir, args.save_path)
last_idx = int(find_latest_index(save_dir)) if os.path.exists(save_dir) else -1

try:
    while not stop_event.is_set():
        if not save_event.is_set():
            time.sleep(0.01)
            continue
        
        last_idx += 1
        print(f"Capturing image to {args.save_path}/{last_idx}/image")
        camera_loader.start(f'shared_data/{args.save_path}/{last_idx}/image')
        camera_loader.end()
        save_event.clear()
        
finally:
    camera_loader.quit()