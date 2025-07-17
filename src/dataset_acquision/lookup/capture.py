from threading import Event, Thread
import time
import argparse
import os
import chime

chime.theme('pokemon')

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list
from paradex.utils.file_io import shared_dir, copy_calib_files

from paradex.io.robot_controller import XArmController, AllegroController, InspireController, FrankaController
from paradex.io.teleop import XSensReceiver, OculusReceiver
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver

from paradex.retargetor import Unimanual_Retargetor, HandStateExtractor
from paradex.geometry.coordinate import DEVICE2WRIST

def initialize_device(args):
    controller = {}
    
    if args.arm == "xarm":
        controller["arm"] = XArmController()

    if args.arm == "franka":
        controller["franka"] = FrankaController()
        
    if args.hand == "allegro":
        controller["hand"] = AllegroController()
        
    elif args.hand == "inspire":
        controller["hand"] = InspireController()
    
    if args.device == "xsens":
        controller["teleop"] = XSensReceiver()
    if args.device == "occulus":
        controller["teleop"] = OculusReceiver()

    controller["camera"] = RemoteCameraController("image", serial_list=None, sync=True)
    controller["signal_generator"] = UTGE900()
    controller["timecode_receiver"] = TimecodeReceiver()
    
    return controller

# === SETUP ===
pc_info = get_pcinfo()
serial_list = get_serial_list()

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=['xarm', 'franka'])
parser.add_argument("--hand", choices=['inspire', 'allegro'])
parser.add_argument("--device", choices=['xsens', 'occulus'])
parser.add_argument('--obj_name', required=True)
parser.add_argument('--grasp_type', required=True)

args = parser.parse_args()
sensors = initialize_device(args)

state_extractor = HandStateExtractor()
home_pose = sensors["arm"].get_position().copy()

sensors["arm"].home_robot(home_pose)
home_start_time = time.time()
while not sensors["arm"].is_ready():
    time.sleep(0.01)
        
retargetor = Unimanual_Retargetor(args.arm, args.hand, home_pose)

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script(f"python src/dataset_acquision/lookup/video_client.py", pc_list)

save_path = os.path.join("capture_", "lookup", args.obj_name, args.grasp_type)
shared_path = os.path.join(shared_dir, save_path)
last_capture_idx = -1

if os.path.exists(shared_path):
    last_capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x)))
else:
    os.makedirs(shared_path, exist_ok=True)

exit = False
start = False

while not exit:
    capture_idx = last_capture_idx + 1
    # prepare for capture, move robot and object
    while not start:
        pass
    
    # start
    chime.info()
    
    os.makedirs(f'{shared_path}/{capture_idx}', exist_ok=True)
    copy_calib_files(f'{shared_path}/{capture_idx}')
        
    sensors['arm'].start()
    sensors['hand'].start()
    sensors['camera'].start()
    sensors['timecode_receiver'].start()
    sensors["signal_generator"].on(1)
    
    while start:
        pass
    # Do capture
    
    sensors['signal_generator'].off(1)

for sensor_name, sensor in sensors.items():
    sensor.quit()
    



























stop_event = Event()
start_capture = Event()
end_capture = Event()

    
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
        
        while not end_capture.is_set():
            time.sleep(0.01)
            continue
        
        camera_loader.end_capture()
        print("end_capture")
        start_capture.clear()
        
        capture_idx += 1
        
finally:
    camera_loader.quit()
    