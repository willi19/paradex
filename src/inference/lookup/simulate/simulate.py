import numpy as np
import os
import argparse
import json
import cv2

from paradex.inference.get_lookup_traj import get_traj
from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path

from paradex.inference.object_6d import get_current_object_6d
from paradex.robot.mimic_joint import parse_inspire

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, type=int)
parser.add_argument("--obj_name", required=True)
parser.add_argument("--grasp_type", required=True)
parser.add_argument("--place", required=True, type=str)

args = parser.parse_args()

arm_name = "xarm"
hand_name = "allegro"
robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf"))

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

place_position = json.load(open(f"data/lookup/{args.obj_name}/obj_pose.json"))
place_6D = np.array(place_position[args.place])

demo_idx = args.index
demo_path = os.path.join("data", "lookup", args.obj_name, args.grasp_type, str(demo_idx))

pick_traj = np.load(f"{demo_path}/pick.npy")
place_traj = np.load(f"{demo_path}/place.npy")
pick_hand_traj = np.load(f"{demo_path}/pick_hand.npy")
place_hand_traj = np.load(f"{demo_path}/place_hand.npy")

traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)

if hand_name == "inspire":
    hand_traj = parse_inspire(hand_traj)

sim = IsaacSimulator(headless=False)
sim.load_robot_asset("xarm", hand_name)
sim.load_object_asset("bottle")
sim.add_env("asdf", env_info = {"robot":{},
                        "robot_vis":{"right":(arm_name, hand_name)},
                        "object":{"bottle":"bottle"},
                        "object_vis":{"bottle_start":"bottle", "bottle_end":"bottle"}})

while True:
    init_action, success = robot.solve_ik(traj[0], "link6")
    last_q = init_action.copy()

    sim.reset("asdf", {"robot":{},
            "robot_vis":{"right":init_action.copy()},
            "object":{"bottle":pick_6D.copy()},
            "object_vis":{"bottle":pick_6D.copy()}
            })

    for i in range(len(traj)):
        sim.tick()
        action, success = robot.solve_ik(traj[i], "link6", last_q)
        
        last_q = action.copy()
        
        action[6:] = hand_traj[i]
        
        sim.step("asdf", {"robot":{},
            "robot_vis":{"right":action.copy()},
            "object_vis":{
                "bottle_start":pick_6D.copy(),
                "bottle_end":place_6D.copy()}
            })

sim.terminate()