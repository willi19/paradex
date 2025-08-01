import numpy as np
import os
import time
import argparse
import tqdm

from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path, shared_dir
from paradex.robot.mimic_joint import parse_inspire
from paradex.inference.get_lookup_traj import get_traj

parser = argparse.ArgumentParser()
parser.add_argument("--obj_name")
parser.add_argument("--grasp_type")
parser.add_argument("--arm", default="xarm")
parser.add_argument("--hand", default="inspire")

args = parser.parse_args()

def load_goal_action(arm_action, hand_qpos):
    ret = []
    goal_action = np.zeros(18)
    for i in tqdm.tqdm(range(len(arm_action))):
        goal_action, _ = robot.solve_ik(arm_action[i], "link6", goal_action.copy())
        goal_action[6:] = parse_inspire(hand_qpos[i:i+1])[0]
        ret.append(goal_action)

    return np.array(ret)

arm_name = args.arm
hand_name = args.hand

robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}.urdf"))

sim = IsaacSimulator(headless=False)
sim.load_robot_asset("xarm", "inspire")
sim.load_object_asset("bottle")

demo_path = os.path.join(shared_dir,"inference_", "lookup", "pringles", "stand_free")

action_dict = {}

lookup_idx = 0

for demo_name in tqdm.tqdm(os.listdir(demo_path)):
    arm_qpos = np.load(os.path.join(demo_path, str(demo_name), arm_name, "qpos.npy"))
    arm_action = np.load(os.path.join(demo_path, str(demo_name), arm_name, "action.npy"))
    
    hand_qpos = np.load(os.path.join(demo_path, str(demo_name), hand_name, "action.npy"))
    obj_T = np.load(os.path.join(demo_path, str(demo_name), "obj_T.npy"))
    c2r = np.load(os.path.join(demo_path, str(demo_name), "C2R.npy"))
    
    pick_6D = np.linalg.inv(c2r) @ obj_T[0]
    place_6D = np.linalg.inv(c2r) @ obj_T[-1]

    if not os.path.exists(os.path.join(demo_path, str(demo_name), "goal_qpos.npy")):
        goal_action = load_goal_action(arm_action, hand_qpos)
        np.save(os.path.join(demo_path, str(demo_name), "goal_qpos.npy"), goal_action)
    else:
        goal_action = np.load(os.path.join(demo_path, str(demo_name), "goal_qpos.npy"))
    action_dict[demo_name] = {
        "real":[],
        "goal":[],
        "obj_T":[]
    }

    print(goal_action.shape, len(arm_qpos))
    for i in range(len(goal_action)// 2):
        real_action = np.zeros(18)
        real_action[:6] = arm_qpos[i*2]
        real_action[6:] = parse_inspire(hand_qpos[i*2:i*2+1])[0]
        action_dict[demo_name]["real"].append(real_action)
        action_dict[demo_name]["goal"].append(goal_action[i*2])
        action_dict[demo_name]["obj_T"].append(np.linalg.inv(c2r) @ obj_T[i*2].copy())

    sim.add_env(demo_name,env_info = {"robot":{},
                            "robot_vis":{"real":(arm_name, hand_name), "goal":(arm_name, hand_name)},
                            "object":{"bottle":"bottle"},
                            "object_vis":{"bottle":"bottle"}})

idx = 0
while True:
    for demo_name in os.listdir(demo_path):
        length = len(action_dict[demo_name]["real"])
        sim.step(demo_name, {"robot":{},
                            "robot_vis":{"real":action_dict[demo_name]["real"][idx % length], "goal":action_dict[demo_name]["goal"][idx % length]},
                            "object_vis":{"bottle":action_dict[demo_name]["obj_T"][idx % length]}})

    sim.tick()
    idx += 1

sim.terminate()