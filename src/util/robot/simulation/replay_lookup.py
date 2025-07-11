import numpy as np
from paradex.inference.get_lookup_traj import get_traj
from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path, shared_dir
import os
import pickle
import time

arm_name = "xarm"
robot = RobotWrapper(os.path.join(rsc_path, "robot", "xarm_inspire.urdf"))
link_id = robot.get_link_index("palm_link")

demo_path = os.path.join(shared_dir, "lookup", "pringles")
demo_name = "0"

arm_qpos = np.load(os.path.join(demo_path, str(demo_name), "arm", "position.npy"))
arm_time = np.load(os.path.join(demo_path, str(demo_name), "arm", "time.npy"))
pick_T = np.load(os.path.join(demo_path, str(demo_name), "start", "obj_6D.npy"))
place_T = np.load(os.path.join(demo_path, str(demo_name), "end", "obj_6D.npy"))

sim = IsaacSimulator(headless=False)
sim.load_robot_asset("xarm", "inspire")
sim.load_object_asset("bottle")
sim.add_env(env_info = {"robot":{},
                        "robot_vis":{"new":(arm_name, "inspire")},
                        "object":{"bottle":"bottle"},
                        "object_vis":{"bottle":"bottle"}})

while True:
    action = np.zeros(18)
    action[:6] = arm_qpos[920]
    sim.reset(0, {"robot":{},
            "robot_vis":{"new":action},
            "object":{"bottle":place_T.copy()},
            "object_vis":{"bottle":place_T.copy()}
            })
    
    T = len(arm_qpos)
    for i in range(920, 1260):
        action[:6] = arm_qpos[i]
        sim.tick()
        sim.step(0, {"robot":{},
            "robot_vis":{"new":action.copy()},
            "object_vis":{"bottle":place_T.copy()}
            })
        print(arm_time[i], i)
        time.sleep(0.01)
        
    sim.tick()
    

sim.terminate()