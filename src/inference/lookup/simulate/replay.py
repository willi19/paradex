import numpy as np
import os
import argparse
import json
import cv2

from paradex.inference.get_lookup_traj import get_traj
from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path, shared_dir

from paradex.inference.object_6d import get_current_object_6d
from paradex.robot.mimic_joint import parse_inspire

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, type=int)
parser.add_argument("--inf_index", default=0, type=int)
parser.add_argument("--obj_name", required=True)
parser.add_argument("--grasp_type", required=True)

args = parser.parse_args()

arm_name = "xarm"
hand_name = "allegro"
robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf"))

# pick_6D = get_current_object_6d(args.obj_name)
inf_path = os.path.join(shared_dir, "inference_", "lookup", args.obj_name, args.grasp_type, str(args.inf_index))
pick_6D = np.load(os.path.join(inf_path, "pick_6D.npy"))
place_6D = np.load(os.path.join(inf_path, "place_6D.npy"))
# pick_6D[:3, :3] = np.eye(3)
real_arm_qpos = np.load(os.path.join(inf_path, arm_name, "qpos.npy"))
real_hand_qpos = np.load(os.path.join(inf_path, hand_name, "qpos.npy"))
real_qpos = np.concatenate([real_arm_qpos, real_hand_qpos], axis=1)
real_action = np.load(os.path.join(inf_path, arm_name, "action.npy"))

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
                        "robot_vis":{"sim":(arm_name, hand_name), "real":(arm_name, hand_name)},
                        "object":{"bottle":"bottle"},
                        "object_vis":{"bottle_start":"bottle", "bottle_end":"bottle"}})

while True:
    init_action, success = robot.solve_ik(real_action[0], "link6")
    last_q = init_action.copy()

    sim.reset("asdf", {"robot":{},
            "robot_vis":{"sim":init_action.copy(), "real":real_qpos[0].copy()},
            "object":{"bottle":pick_6D.copy()},
            "object_vis":{"bottle":pick_6D.copy()}
            })

    for i in range(len(real_action)):
        sim.tick()
        action, success = robot.solve_ik(real_action[i], "link6", last_q)
        
        last_q = action.copy()
        
        action[6:] = real_hand_qpos[i]
        
        sim.step("asdf", {"robot":{},
            "robot_vis":{"sim":action.copy(),"real":real_qpos[min(real_qpos.shape[0]-1, i)].copy()},
            "object_vis":{
                "bottle_start":pick_6D.copy(),
                "bottle_end":place_6D.copy()}
            })

sim.terminate()