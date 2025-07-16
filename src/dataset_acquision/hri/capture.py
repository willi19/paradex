from threading import Event, Thread
import time
import argparse
import os

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.file_io import shared_dir, copy_calib_files
# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

parser = argparse.ArgumentParser()
parser.add_argument('--obj_name', required=True)
args = parser.parse_args()

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/dataset_acquision/hri/video_client.py", pc_list)

camera_loader = RemoteCameraController("video", None)

stop_event = Event()
start_capture = Event()
end_capture = Event()

listen_keyboard({"s":start_capture, "q":stop_event, "e" : end_capture})

save_path = os.path.join(shared_dir, "capture_", "hri", args.save_path)
last_capture_idx = -1

if os.path.exists(save_path):
    last_capture_idx = max(os.listdir(save_path), key=lambda x:int(x))
try:
    capture_idx = last_capture_idx + 1
    while not stop_event.is_set():
        if not start_capture.is_set():
            time.sleep(0.01)
            continue
        copy_calib_files(f'{save_path}/{capture_idx}')
        
        end_capture.clear()
        camera_loader.start_capture(f'{save_path}/{capture_idx}')
        
        while not end_capture.is_set():
            time.sleep(0.01)
            continue
        
        camera_loader.end_capture()
        print("end_capture")
        start_capture.clear()
        
        capture_idx += 1
        
finally:
    camera_loader.quit()