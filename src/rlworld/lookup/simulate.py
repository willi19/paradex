import time
import numpy as np
from threading import Event
import argparse
import json
import os
from datetime import datetime

from paradex.inference.lookup_table import get_traj
from paradex.inference.util import home_robot
from paradex.inference.simulate import simulate
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--marker", default=False, action="store_true")

    args = parser.parse_args()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())

    arm_name = "xarm"
    hand_name = args.hand
    
    place_position_list = json.load(open(f"data/lookup/{args.object}/obj_pose.json"))
    place_position_list = {i:np.array(p) for i, p in place_position_list.items()}
    
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.35],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    end_pos= np.array([[0, 0, 1, 0.25],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    place_id_list = ["1", "5"]
    
    c2r = load_latest_C2R()
    
    save_path = os.path.join("inference", "lookup", args.object, args.hand)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    register_dict = get_image(f"inference/register/{timestamp}")
    print(time.time()-start_time)
    place_id_list = ["1", "5"]
    
    get_goal_position(register_dict, place_id_list)
    
    stop_event = Event()
    start_event = Event()
 
    event_dict = {"q":stop_event, "y":start_event}
    listen_keyboard(event_dict)
    
    index = 0
    while not stop_event.is_set():
        place_id = place_id_list[int(index) % len(place_id_list)]
        place_6D = place_position_list[place_id]
        
        pick_6D = get_current_object_6d(args.object, args.marker)
        
        pick_6d = normalize_cylinder(pick_6D)
            
        choosen_index, traj, hand_traj = get_traj(args.object, hand_name, start_pos.copy(), pick_6D.copy(), place_6D.copy())
        print(choosen_index, traj.shape, hand_traj.shape)
        # Show simulation
        simulate(traj, hand_traj, pick_6D, place_6D, hand_name, args.object, start_event, stop_event)
        
        if stop_event.is_set():
            break
        
        start_event.clear()
        index += 1
    