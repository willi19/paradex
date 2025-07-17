from threading import Event, Thread
import time
import argparse
import os
import chime
chime.theme('pokemon')

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.signal_generator.UTGE900 import UTGE900

from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.utils.file_io import shared_dir, copy_calib_files
from paradex.utils.keyboard_listener import listen_keyboard


# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="xarm")
parser.add_argument("--hand")
parser.add_argument('--obj_name', required=True)
args = parser.parse_args()

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/dataset_acquision/hri/video_client.py", pc_list)

camera_loader = RemoteCameraController("video", None, sync=True)
signal_generator = UTGE900()

stop_event = Event()
start_capture = Event()
end_capture = Event()

listen_keyboard({"s":start_capture, "e":end_capture, "q":stop_event})

save_path = os.path.join("capture_", "hri", args.obj_name)
shared_path = os.path.join(shared_dir, save_path)
last_capture_idx = -1

if os.path.exists(shared_path):
    last_capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x)))
else:
    os.makedirs(shared_path, exist_ok=True)
    
try:
    capture_idx = last_capture_idx + 1
    while not stop_event.is_set():
        if not start_capture.is_set():
            time.sleep(0.01)
            continue
        os.makedirs(f'{shared_path}/{capture_idx}', exist_ok=True)
        copy_calib_files(f'{shared_path}/{capture_idx}')
        
        end_capture.clear()
        camera_loader.start_capture(f'{save_path}/{capture_idx}/videos')
        print("start_capture")
        signal_generator.on(1)
        chime.info()
        
        while not end_capture.is_set():
            time.sleep(0.01)
            continue
        
        camera_loader.end_capture()
        print("end_capture")
        start_capture.clear()
        signal_generator.off(1)
        chime.success()
        capture_idx += 1
        
finally:
    camera_loader.quit()
    