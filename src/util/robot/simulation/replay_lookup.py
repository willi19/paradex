import numpy as np
from paradex.inference.lookup_table import get_traj
from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path, shared_dir
from paradex.robot.mimic_joint import parse_inspire
import os
import pickle
import time

arm_name = "xarm"
hand_name = "inspire"
robot = RobotWrapper(os.path.join(rsc_path, "robot", "xarm_inspire.urdf"))
link_id = robot.get_link_index("palm_link")

demo_path = os.path.join(shared_dir,"capture_", "lookup", "pringles", "stand")
demo_name = "0"

arm_qpos = np.load(os.path.join(demo_path, str(demo_name), arm_name, "qpos.npy"))
hand_qpos = np.load(os.path.join(demo_path, str(demo_name), hand_name, "qpos.npy"))
obj_T = np.load(os.path.join(demo_path, str(demo_name), "obj_T.npy"))
c2r = np.load(os.path.join(demo_path, str(demo_name), "C2R.npy"))
# arm_time = np.load(os.path.join(demo_path, str(demo_name), arm_name, "time.npy"))
# pick_T = np.load(os.path.join(demo_path, str(demo_name), "start", "obj_6D.npy"))
# place_T = np.load(os.path.join(demo_path, str(demo_name), "end", "obj_6D.npy"))

sim = IsaacSimulator(headless=False)
sim.load_robot_asset("xarm", "inspire")
sim.load_object_asset("bottle")
sim.add_env(env_info = {"robot":{},
                        "robot_vis":{"new":(arm_name, "inspire")},
                        "object":{"bottle":"bottle"},
                        "object_vis":{"bottle":"bottle"}})

while True:
    action = np.zeros(18)
    action[:6] = arm_qpos[0]
    action[6:] = parse_inspire(hand_qpos[0:1])[0]
    sim.reset(0, {"robot":{},
            "robot_vis":{"new":action},
            "object":{"bottle":np.linalg.inv(c2r) @ obj_T[0].copy()},
            "object_vis":{"bottle":np.linalg.inv(c2r) @ obj_T[0].copy()}
            })
    
    T = len(arm_qpos)
    delta = 125
    print(obj_T.shape[0] - T)
    
    for i in range(T):
        action[:6] = arm_qpos[i]
        action[6:] = parse_inspire(hand_qpos[i:i+1])[0]
        sim.tick()
        sim.step(0, {"robot":{},
            "robot_vis":{"new":action.copy()},
            "object_vis":{"bottle":np.linalg.inv(c2r) @ obj_T[i+delta].copy()}
            })
        time.sleep(0.01)
        
    sim.tick()
    

sim.terminate()