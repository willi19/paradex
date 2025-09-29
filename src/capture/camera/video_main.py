from threading import Event, Thread
import time
import argparse
import os

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.file_io import copy_calib_files, shared_dir
from paradex.io.signal_generator.UTGE900 import UTGE900

# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
# run_script(f"python src/capture/camera/video_client.py", pc_list)

camera_loader = RemoteCameraController("video", None, sync=True, debug=True)

stop_event = Event()
start_capture = Event()
end_capture = Event()

listen_keyboard({"s":start_capture, "e" : end_capture, "q":stop_event})
gSgen = UTGE900()

try:
    capture_idx = 0
    while not stop_event.is_set():
        if not start_capture.is_set():
            time.sleep(0.01)
            continue
        
        save_dir = os.path.join(shared_dir, args.save_path, str(capture_idx))
        os.makedirs(save_dir, exist_ok=True)
        copy_calib_files(save_dir)
        
        end_capture.clear()
        camera_loader.start(f'{args.save_path}/{capture_idx}')
        gSgen.on(1)
        
        while not end_capture.is_set():
            time.sleep(0.01)
            continue
        
        camera_loader.end()
        print("end_capture")
        start_capture.clear()
        gSgen.off(1)
        time.sleep(0.5)
        capture_idx += 1
        
finally:
    camera_loader.quit()
    gSgen.quit()