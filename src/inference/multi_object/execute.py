import numpy as np
import trimesh
import os
import time
from threading import Event

from paradex.visualization.viser_viewer import ViserViewer
from paradex.utils.file_io import rsc_path, get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.io.robot_controller import get_arm, get_hand
from paradex.inference.util import home_robot
from paradex.utils.keyboard_listener import listen_keyboard

start_pos= np.array([[0, 0, 1, 0.3],
                    [1, 0, 0, -0.3],
                    [0, 1, 0, 0.4], 
                    [0, 0, 0, 1]])

stop_event = Event()
start_event = Event()

event_dict = {"q":stop_event, "y":start_event}
listen_keyboard(event_dict)

sensors = {}
arm_name = "xarm"
hand_name = "allegro"
sensors["arm"] = get_arm(arm_name)
sensors["hand"] = get_hand(hand_name)

# sensors["hand"].home_robot()
home_robot(sensors["arm"], start_pos)

obj_list = os.listdir("pickplace/traj")
obj_list.sort()
prev_pose = start_pos.copy()
for obj_id in obj_list:
    start_pos = np.load(f"pickplace/traj/{obj_id}/start.npy")
    end_pos = np.load(f"pickplace/traj/{obj_id}/end.npy")
    pick_pos = np.load(f"pickplace/traj/{obj_id}/pick.npy")
    place_pos = np.load(f"pickplace/traj/{obj_id}/place.npy")
    
    state = []
    for _ in range(len(start_pos)):
        state.append("start")
    for _ in range(len(pick_pos)):
        state.append("pick")
    for _ in range(len(end_pos)):
        state.append("end")
    for _ in range(len(place_pos)):
        state.append("place")
    
    start_hand = np.load(f"pickplace/traj/{obj_id}/start_hand.npy")
    end_hand = np.load(f"pickplace/traj/{obj_id}/end_hand.npy")
    pick_hand = np.load(f"pickplace/traj/{obj_id}/pick_hand.npy")
    place_hand = np.load(f"pickplace/traj/{obj_id}/place_hand.npy")
    
    tot_pos = np.concatenate([start_pos, pick_pos, end_pos, place_pos], axis=0)
    tot_hand = np.concatenate([start_hand, pick_hand, end_hand, place_hand],
                              axis=0)
    for t in range(len(tot_pos)):
        print(t, state[t], obj_id, np.linalg.norm(tot_pos[t][:3, 3]-prev_pose[:3, 3]))
        prev_pose = tot_pos[t].copy()
        sensors["arm"].set_action(tot_pos[t])
        sensors["hand"].set_target_action(tot_hand[t])
        # if obj_id == "1":
        #     while not start_event.is_set():
        #         time.sleep(0.05)
        #         continue
        # else:
        time.sleep(0.06)
        start_event.clear()

        if stop_event.is_set():
            break
    
    if stop_event.is_set():
        break

for sensor_name, sensor in sensors.items():
    sensor.quit()