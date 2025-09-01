import time
import numpy as np
from threading import Event
import argparse
import json
import os
from datetime import datetime

from paradex.inference.simulate import simulate
from paradex.inference.lookup_table import get_traj
from paradex.inference.util import home_robot
from paradex.inference.object_6d import get_current_object_6d, normalize_cylinder, get_goal_position

from paradex.io.camera.util import get_image
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.io.camera.timecode_receiver import TimecodeReceiver
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import run_script

from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R
from paradex.utils.env import get_pcinfo
from paradex.utils.keyboard_listener import listen_keyboard

use_sim = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--marker", default=False, action="store_true")
    parser.add_argument("--simple", default=False, action="store_true")

    args = parser.parse_args()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())

    arm_name = "xarm"
    hand_name = args.hand
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    sensors["hand"] = get_hand(hand_name)
    
    if not args.simple:
        sensors["signal_generator"] = UTGE900()
        sensors["timecode_receiver"] = TimecodeReceiver()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.35],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    end_pos= np.array([[0, 0, 1, 0.25],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    
    c2r = load_latest_C2R()
    
    save_path = os.path.join("inference", "lookup", args.object, args.hand)
    
    register_dict = get_image(f"inference/register/{timestamp}")
    place_id_list = ["1", "4"]
    place_position_dict = get_goal_position(register_dict, place_id_list)
    
    shared_path = os.path.join(shared_dir, save_path)
    os.makedirs(shared_path, exist_ok=True)
    if len(os.listdir(shared_path)) == 0:
        capture_idx = 0
    else:
        capture_idx = int(max(os.listdir(shared_path), key=lambda x:int(x))) + 1
    
    stop_event = Event()
    start_event = Event()
 
    event_dict = {"q":stop_event, "y":start_event}
    listen_keyboard(event_dict)
    
    while not stop_event.is_set():
        place_id = place_id_list[int(capture_idx) % len(place_id_list)]
        place_6D = place_position_dict[place_id]
        
        home_robot(sensors["arm"], start_pos)

        print("press y after fixing object position")        
        while not start_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        
        if stop_event.is_set():
            break
        
        start_event.clear()
        
        pick_6D = get_current_object_6d(args.object, args.marker)
        pick_6d = normalize_cylinder(pick_6D)

        choosen_index, traj, hand_traj = get_traj(args.object, hand_name, start_pos.copy(), pick_6D.copy(), place_6D.copy())
        
        # Show simulation
        if use_sim:
            print("press y if trajectory ok")
            simulate(traj, hand_traj, pick_6D, place_6D, hand_name, args.object, start_event, stop_event)
            
            if stop_event.is_set():
                break
            
            start_event.clear()
        
        # start the camera
        if not args.simple:
            run_script(f"python src/capture/camera/video_client.py", pc_list)
            sensors["camera"] = RemoteCameraController("video", serial_list=None, sync=True)

            os.makedirs(os.path.join(shared_path, str(capture_idx)))
            copy_calib_files(f'{shared_path}/{capture_idx}')
            np.save(f'{shared_path}/{capture_idx}/C2R.npy', c2r)
            np.save(f'{shared_path}/{capture_idx}/pick_6D.npy', pick_6D)
            np.save(f'{shared_path}/{capture_idx}/target_6D.npy', place_6D)
            np.save(f'{shared_path}/{capture_idx}/traj.npy', traj)
            np.save(f'{shared_path}/{capture_idx}/hand_traj.npy', hand_traj)
            np.save(f'{shared_path}/{capture_idx}/choosen_index.npy', np.array(choosen_index))
        

        # Start capture
        sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{arm_name}")
        sensors['hand'].start(f"{shared_path}/{capture_idx}/raw/{hand_name}")
        
        if not args.simple:
            sensors['camera'].start(f"{save_path}/{capture_idx}/videos")
            sensors['timecode_receiver'].start(f"{shared_path}/{capture_idx}/raw/timestamp")
            sensors["signal_generator"].on(1)
        
        state_hist = []
        state_time = []
        for i in range(len(traj)):
            sensors["arm"].set_action(traj[i])
            sensors["hand"].set_target_action(hand_traj[i])
            state_hist.append(i)
            state_time.append(time.time())
            time.sleep(0.03)  # Simulate time taken for each action
            
            if stop_event.is_set():
                break
        
        sensors["arm"].end()
        sensors["hand"].end()
        place_6D = get_current_object_6d(args.object, args.marker)
        
        if not args.simple:
            sensors["camera"].end()
            sensors['timecode_receiver'].end()
            sensors['signal_generator'].off(1)
            
            os.makedirs(f"{shared_path}/{capture_idx}/raw/state", exist_ok=True)
            np.save(f"{shared_path}/{capture_idx}/raw/state/state.npy", state_hist)
            np.save(f"{shared_path}/{capture_idx}/raw/state/time.npy", state_time)
            np.save(f'{shared_path}/{capture_idx}/place_6D.npy', place_6D)
            sensors["camera"].quit()
            time.sleep(1) # Need distributor to stop
        
        capture_idx += 1
        
    
    home_robot(sensors["arm"], end_pos.copy())
    for sensor_name, sensor in sensors.items():
        if sensor_name == "camera":
            continue
        sensor.quit()