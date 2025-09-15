from threading import Event, Thread
import time
import argparse

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.utils.keyboard_listener import listen_keyboard

# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
args = parser.parse_args()

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/capture/camera/video_client.py", pc_list)

camera_loader = RemoteCameraController("video", None,debug=True)

stop_event = Event()
start_capture = Event()
end_capture = Event()

listen_keyboard({"s":start_capture, "e" : end_capture, "q":stop_event})

try:
    capture_idx = 0
    while not stop_event.is_set():
        if not start_capture.is_set():
            time.sleep(0.01)
            continue
        
        end_capture.clear()
        camera_loader.start(f'{args.save_path}/{capture_idx}')
        
        while not end_capture.is_set():
            time.sleep(0.01)
            continue
        
        camera_loader.end()
        print("end_capture")
        start_capture.clear()
        
        capture_idx += 1
        
finally:
    camera_loader.quit()