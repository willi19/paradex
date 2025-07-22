import time
import numpy as np
import chime
import argparse
import json
import os

from paradex.inference.get_lookup_traj import get_traj
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.inference.object_6d import get_current_object_6d
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo, get_serial_list

def initialize_device(arm_name, hand_name):
    controller = {}
    
    controller["arm"] = get_arm(arm_name)
    controller["hand"] = get_hand(hand_name)
    
    controller["camera"] = RemoteCameraController("video", serial_list=None, sync=True)
    controller["signal_generator"] = UTGE900()
    controller["timecode_receiver"] = TimecodeReceiver()
    
    return controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--obj_name", required=True)
    parser.add_argument("--grasp_type", required=True)
    parser.add_argument("--place", required=True, type=str)

    args = parser.parse_args()
    
    pick_6D = get_current_object_6d(args.obj_name)
    pick_6D[:3,:3] = np.eye(3)
    
    place_position = json.load(open(f"data/lookup/{args.obj_name}/obj_pose.json"))
    place_6D = np.array(place_position[args.place])
    
    arm_name = "xarm"
    hand_name = "inspire"
    
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    git_pull("merging", pc_list)
    run_script(f"python src/capture/camera/video_client.py", pc_list)

    sensors = initialize_device(arm_name, hand_name)
    
    demo_idx = args.index
    demo_path = os.path.join("data", "lookup", args.obj_name, args.grasp_type, str(demo_idx))

    pick_traj = np.load(f"{demo_path}/pick.npy")
    place_traj = np.load(f"{demo_path}/place.npy")
    pick_hand_traj = np.load(f"{demo_path}/pick_hand.npy")
    place_hand_traj = np.load(f"{demo_path}/place_hand.npy")

    traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)
    
    save_path = os.path.join("inference_", "lookup", args.obj_name, args.grasp_type)
    shared_path = os.path.join(shared_dir, save_path)
    os.makedirs(shared_path, exist_ok=True)
    if len(os.listdir(shared_path)) == 0:
        capture_idx = 0
    else:
        capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x))) + 1

    c2r = load_latest_C2R()
    os.makedirs(os.path.join(shared_path, str(capture_idx)))
    copy_calib_files(f'{shared_path}/{capture_idx}')
    np.save(f'{shared_path}/{capture_idx}/C2R.npy', c2r)
        
        
    sensors["arm"].home_robot(traj[0])
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        if time.time() - home_start_time > 0.5:
            chime.warning()
            home_start_time = time.time()
        time.sleep(0.01)
    
    chime.success()
    
    
    sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{arm_name}")
    sensors['hand'].start(f"{shared_path}/{capture_idx}/raw/{hand_name}")
    sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
    sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
    sensors["signal_generator"].on(1)
    
    for i in range(len(traj)):
        sensors["arm"].set_action(traj[i])
        sensors["hand"].set_target_action(hand_traj[i])
        time.sleep(0.06)  # Simulate time taken for each action
    
    sensors["arm"].end()
    sensors["hand"].end()
    sensors["camera"].end()
    sensors['timecode_receiver'].end()
    
    sensors['signal_generator'].off(1)
    
    wrist_rot = np.array([[0, 0, 1, 0.3],
                      [1, 0, 0, -0.15],
                      [0, 1, 0, 0.10], 
                      [0, 0, 0, 1]])

    sensors["arm"].home_robot(wrist_rot)
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        time.sleep(0.01)

    chime.info()

    for sensor_name, sensor in sensors.items():
        print(sensor_name)
        sensor.quit()