from paradex.simulator.isaac import Simulator
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.retargetor.unimanual import Retargetor
from paradex.retargetor.state import HandStateExtractor

from threading import Event
import argparse
import numpy as np
import time
import os
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', type=str, nargs="+", default=["franka", "xarm"])
parser.add_argument('--hand', type=str, nargs="+", default=["inspire", "allegro"])
parser.add_argument('--object', type=str, nargs="+", default=[])

args = parser.parse_args()

if args.device == 'xsens': 
    from paradex.io.teleop import XSensReceiver
    teleop_device = XSensReceiver()

if args.device =='occulus':
    from paradex.io.teleop import OculusReceiver
    teleop_device = OculusReceiver()

save_video = False
save_state = False
headless = False
num_envs = 1

retargetor_list = []

qpos_init = np.zeros(6)
init_pose = np.zeros(6)

home_wrist_pose = np.array([[1,0,0,300],[0,1,0,400], [0,0,1,200],[0,0,0,1]])
arm_list = [None if s.lower() == "none" else s for s in args.arm]
hand_list = [None if s.lower() == "none" else s for s in args.hand]
    
for arm_name in arm_list:
    for hand_name in hand_list:        
        if arm_name == None and hand_name == None:
            continue
        
        retargetor_list.append(Retargetor(arm_name, hand_name, home_wrist_pose))

state_extractor = HandStateExtractor()

sim = Simulator(
                headless
                )

for arm_name in arm_list:
    for hand_name in hand_list:
        if arm_name == None and hand_name == None:
            continue
        
        sim.load_robot_asset(arm_name, hand_name)

obj_list = []
for obj_name in obj_list:
    sim.load_object_asset(obj_name)

if save_video:
    sim.load_camera() # load camparam if desired
    sim.set_videopath("teleop_sim/video")

if save_state:
    sim.set_savepath("teleop_sim/state")

for arm_name in arm_list:
    for hand_name in hand_list:
        if arm_name == None and hand_name == None:
            continue
        env_info = {
            "robot":{},
            "robot_vis":{"right":(arm_name, hand_name)}, 
            "object":{},
            "object_vis":{}
        }

        sim.add_env(env_info)

stop_event = Event()
listen_keyboard({"q":stop_event})

robot_list = {}
for robot_name in arm_list + hand_list:
    if robot_name == None:
        continue
    robot_list[robot_name] = RobotWrapper(os.path.join(rsc_path, "robot", f"{robot_name}.urdf"))

def get_action(arm_name, hand_name, wrist_pose, hand_action):
    if arm_name != None:
        last_link = robot_list[arm_name].get_end_links()[0]
        arm_action, success = robot_list[arm_name].solve_ik(
            wrist_pose.copy(),
            last_link,
            q_init= qpos_init.copy(),
            max_iter=1000,
            tol=1e-4,
            alpha=1e-1
        )
        if hand_name != None:
            return np.concatenate([arm_action, hand_action]).astype(np.float32)
        else:
            return arm_action
    
    xyz = wrist_pose[:3,3]
    rpy = R.from_matrix(wrist_pose[:3,:3]).as_euler("zyx")
                
    return np.concatenate([xyz, rpy, hand_action]).astype(np.float32)

env_idx = 0
for arm_name in arm_list:
    for hand_name in hand_list:
        if arm_name == None and hand_name == None:
            continue
        hand_dof = 1
        if hand_name != None:
            hand_dof = robot_list[hand_name].dof
        robot_action = get_action(arm_name, hand_name, home_wrist_pose, np.zeros(hand_dof))
        # sim.reset(env_idx, {"robot":{"right":robot_action},"robot_vis":{"right":robot_action}, "object":{}})    
        sim.reset(env_idx, {"robot":{},"robot_vis":{"right":robot_action}, "object":{}})    
        env_idx += 1
    
while not stop_event.is_set():
    sim.tick()
    hand_pose = teleop_device.get_data() #{'Left':{}, 'Right':{}}
    print(hand_pose)
    if hand_pose["Right"] == None:
        continue
    env_idx = 0
    state = state_extractor.get_state(hand_pose['Left'])
    for arm_name in arm_list:
        for hand_name in hand_list:
            if arm_name == None and hand_name == None:
                continue
        
            if state == 1:
                retargetor_list[env_idx].pause()
                continue
            
            wrist_pose, hand_action = retargetor_list[env_idx].get_action(hand_pose)
            print(wrist_pose)
            robot_action = get_action(arm_name, hand_name, wrist_pose, hand_action).astype(np.float32)
            
            sim.step(env_idx, {"robot":{},"robot_vis":{"right":robot_action}, "object_vis":{}})
            env_idx += 1
    time.sleep(0.01)
            
    
sim.save()
sim.terminate()
teleop_device.quit()