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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--obj_name", required=True)
    parser.add_argument("--grasp_type", required=True)

    args = parser.parse_args()
    
    path_planning = [(1, 2), (2, 4), (4, 2), (2, 3), (3, 1), (1, 3), (3, 4), (4, 1), (1, 4), (4, 3), (3, 2), (2, 1)]
    
    arm_name = "xarm"
    hand_name = "allegro"
    
    sensors = {}
    sensors["arm"] = get_arm(arm_name)
    sensors["hand"] = get_hand(hand_name)
    sensors["signal_generator"] = UTGE900()
    sensors["timecode_receiver"] = TimecodeReceiver()
    
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    git_pull("merging", pc_list)
    
    demo_idx = args.index
    demo_path = os.path.join("data", "lookup", args.obj_name, args.grasp_type, str(demo_idx))

    pick_traj = np.load(f"{demo_path}/pick.npy")
    place_traj = np.load(f"{demo_path}/place.npy")
    pick_hand_traj = np.load(f"{demo_path}/pick_hand.npy")
    place_hand_traj = np.load(f"{demo_path}/place_hand.npy")
    
    place_position_list = json.load(open(f"data/lookup/{args.obj_name}/obj_pose.json"))
    start_pos= np.array([[0, 0, 1, 0.3],
                        [1, 0, 0, -0.35],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    
    end_pos= np.array([[0, 0, 1, 0.25],
                        [1, 0, 0, 0.0],
                        [0, 1, 0, 0.10], 
                        [0, 0, 0, 1]])
    while True:
        sensors["arm"].home_robot(start_pos.copy())  
        home_start_time = time.time()
        while not sensors["arm"].is_ready():
            time.sleep(0.01)

        chime.info()
        
        place_id = input(f"place the object to")
        if place_id == "-1":
            break
        
        # retister object
        pick_6D = get_current_object_6d(args.obj_name)
        if "lay" in args.grasp_type:
            z = pick_6D[:3, 2]
            pick_6D[:3,2] = np.array([z[0], z[1], 0])
            pick_6D[:3,2] /= np.linalg.norm(pick_6D[:3,2])

            pick_6D[:3,0] = np.array([0,0,1])
            pick_6D[:3,1] = np.array([z[1], -z[0], 0])
            pick_6D[:3,1] /= np.linalg.norm(pick_6D[:3,2])
        else:
            pick_6D[:3,:3] = np.eye(3)
            
        place_6D = np.array(place_position_list[place_id])
        
        traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)
        
        # start the camera
        run_script(f"python src/capture/camera/video_client.py", pc_list)
        sensors["camera"] = RemoteCameraController("video", serial_list=None, sync=True)

        # Set directory
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
        np.save(f'{shared_path}/{capture_idx}/pick_6D.npy', pick_6D)
        np.save(f'{shared_path}/{capture_idx}/place_6D.npy', place_6D)
        np.save(f'{shared_path}/{capture_idx}/traj.npy', traj)
        np.save(f'{shared_path}/{capture_idx}/hand_traj.npy', hand_traj)
        
        # Prepare execution
        sensors["arm"].home_robot(traj[0])
        home_start_time = time.time()
        while not sensors["arm"].is_ready():
            if time.time() - home_start_time > 0.5:
                chime.warning()
                home_start_time = time.time()
            time.sleep(0.01)
        
        chime.success()
        
        # Start capture
        sensors['arm'].start(f"{shared_path}/{capture_idx}/raw/{arm_name}")
        sensors['hand'].start(f"{shared_path}/{capture_idx}/raw/{hand_name}")
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
        
        
        sensors["arm"].end()
        sensors["hand"].end()
        sensors["camera"].end()
        sensors['timecode_receiver'].end()
        sensors['signal_generator'].off(1)
        
        os.makedirs(f"{shared_path}/{capture_idx}/raw/state", exist_ok=True)
        np.save(f"{shared_path}/{capture_idx}/raw/state/state.npy", state_hist)
        np.save(f"{shared_path}/{capture_idx}/raw/state/time.npy", state_time)
        sensors["camera"].quit()
    
    sensors["arm"].home_robot(end_pos)
    home_start_time = time.time()
    while not sensors["arm"].is_ready():
        time.sleep(0.01)

    chime.info()
    
    for sensor_name, sensor in sensors.items():
        if sensor_name == "camera":
            continue
        sensor.quit()