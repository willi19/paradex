from paradex.simulator.isaac import Simulator
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.retargetor.unimanual import Retargetor

from threading import Event
import argparse
import numpy as np
import time
import os
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.utils.file_io import rsc_path

parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus'])
parser.add_argument('--arm', type=str, nargs="+", default=["franka", "xarm"])
parser.add_argument('--hand', type=str, nargs="+", default=["inspire", "allegro"])
parser.add_argument('--object', type=str, nargs="+", default=[])

args = parser.parse_args()

if args.device == 'xsens': 
    from paradex.io.xsens.receiver import XSensReceiver
    teleop_device = XSensReceiver()

if args.device =='occulus':
    from paradex.io.occulus.receiver import OculusReceiver
    teleop_device = OculusReceiver()

save_video = False
save_state = False
headless = False
num_envs = 1

retargetor_list = []

home_wrist_pose = np.eye(4)
for arm_name in args.arm:
    for hand_name in args.hand:
        retargetor_list.append(Retargetor(arm_name, hand_name, home_wrist_pose))

sim = Simulator(
                headless
                )

for arm_name in args.arm:
    for hand_name in args.hand:
        sim.load_robot_asset(arm_name, hand_name)

obj_list = []
for obj_name in obj_list:
    sim.load_object_asset(obj_name)

if save_video:
    sim.load_camera() # load camparam if desired
    sim.set_videopath("teleop_sim/video")

if save_state:
    sim.set_savepath("teleop_sim/state")

for arm_name in args.arm:
    for hand_name in args.hand:
        env_info = {
            "robot":{"right":(arm_name, hand_name)},
            "robot_vis":{"right":(arm_name, hand_name)}, 
            "object":{},
            "object_vis":{}
        }

        sim.add_env(env_info)

stop_event = Event()
listen_keyboard({"q":stop_event})

robot = RobotWrapper(
    os.path.join(rsc_path, "robot", "xarm_allegro.urdf")
)
link_index = robot.get_link_index("link6")

while not stop_event.is_set():
    sim.tick()
    hand_pose = teleop_device.get_data() #{'Left':{}, 'Right':{}}
    if hand_pose["Right"] == None:
        continue
    env_idx = 0
    for arm_name in args.arm:
        for hand_name in args.hand:
            arm_action, hand_action = retargetor_list[env_idx].get_action(hand_pose)
            arm_action, success = robot.solve_ik(
                arm_action,
                "link6",
                q_init=np.zeros(22),
                max_iter=100,
                tol=1e-4,
                alpha=1e-1
            )
            robot_action = np.concatenate([arm_action]).astype(np.float32)
            sim.step(env_idx, {"robot":{"right":robot_action},"robot_vis":{"right":robot_action}, "object_vis":{}})
            env_idx += 1
    time.sleep(0.01)
            
    
sim.save()
sim.terminate()