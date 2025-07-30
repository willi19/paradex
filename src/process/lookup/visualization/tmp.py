import numpy as np
from paradex.inference.get_lookup_traj import get_traj
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
link_id = robot.get_link_index("link6")

demo_path = os.path.join(shared_dir,"capture_", "lookup", "pringles", "stand_free")
demo_name = "2"

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
sim.add_env("tmp", env_info = {"robot":{},
                        "robot_vis":{"new":(arm_name, "inspire")},
                        "object":{},
                        "object_vis":{"bottle_start":"bottle", "bottle_end":"bottle"}})
print(np.linalg.inv(c2r) @ obj_T[0], np.linalg.inv(c2r) @ obj_T[-1])
print(obj_T[0], obj_T[-1])

while True:
    action = np.zeros(18)
    action[:6] = arm_qpos[229]
    action[6:] = parse_inspire(hand_qpos[229:230])[0]
    
    
    # print(np.linalg.inv(obj_T[30]) @ c2r @ wrist_pose)
    sim.reset("tmp", {"robot":{},
            "robot_vis":{"new":action},
            "object":{},
            "object_vis":{"bottle_start":np.linalg.inv(c2r) @ obj_T[0].copy(), "bottle_end":np.linalg.inv(c2r) @ obj_T[-1].copy()}
            })
    
    T = min(len(arm_qpos), obj_T.shape[0])
    
    for i in range(229, 230):
        action[:6] = arm_qpos[i]
        action[6:] = parse_inspire(hand_qpos[i:i+1])[0]

        # if i == 227:
        #     robot.compute_forward_kinematics(action)
        #     wrist_pose = robot.get_link_pose(robot.get_link_index("wrist"))
        #     asdf = np.linalg.inv(c2r) @ obj_T[0]
        #     asdf[:3,:3] = np.eye(3)
            
        #     print(np.linalg.inv(asdf) @ wrist_pose)
            
        
        # obj_T[i, :3,:3] = np.eye(3)
        sim.tick()
        sim.step("tmp", {"robot":{},
            "robot_vis":{"new":action},
            "object_vis":{"bottle_start":np.linalg.inv(c2r) @ obj_T[0].copy(), "bottle_end":np.linalg.inv(c2r) @ obj_T[-1].copy()}
            })
        time.sleep(0.03)
        
    sim.tick()
    

sim.terminate()