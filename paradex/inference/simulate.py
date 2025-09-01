import numpy as np
import os
import argparse
import json
import cv2

from paradex.inference.lookup_table import get_traj
from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path

from paradex.inference.object_6d import get_current_object_6d
from paradex.robot.mimic_joint import parse_inspire

import numpy as np
import os
import argparse
import json
import cv2
import time

from paradex.inference.lookup_table import get_traj
from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path

from paradex.inference.object_6d import get_current_object_6d
from paradex.robot.mimic_joint import parse_inspire

def simulate(traj, hand_traj, pick_6D, place_6D, hand_name, obj_name, start_event, stop_event):
    arm_name = "xarm"
    sim = IsaacSimulator(headless=False)
    sim.load_robot_asset(arm_name, hand_name)
    sim.load_object_asset(obj_name)
    sim.add_env("inference", env_info = {"robot":{},
                            "robot_vis":{"right":(arm_name, hand_name)},
                            "object":{},
                            # "object":{obj_name:obj_name},
                            "object_vis":{f"{obj_name}_start":obj_name, f"{obj_name}_end":obj_name}})

    robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf"))

    while not start_event.is_set() and not stop_event.is_set():
        init_action, success = robot.solve_ik(traj[0], "link6")
        last_q = init_action.copy()

        sim.reset("inference", {"robot":{},
                "robot_vis":{"right":init_action.copy()},
                "object":{},
                "object_vis":{f"{obj_name}":pick_6D.copy()}
                })
        
        for i in range(0, len(traj), 3):
            sim.tick()
            time.sleep(1/30)
            action, success = robot.solve_ik(traj[i], "link6", last_q)
            
            last_q = action.copy()
            
            action[6:] = hand_traj[i]
            
            sim.step("inference", {"robot":{},
                "robot_vis":{"right":action.copy()},
                "object_vis":{
                    f"{obj_name}_start":pick_6D.copy(),
                    f"{obj_name}_end":place_6D.copy()}
                })
            
            if start_event.is_set() or stop_event.is_set():
                break

    sim.terminate()
    
    
def simulate_temp(traj, hand_traj, pick_6D, place_6D, hand_name, obj_name, start_event, stop_event):
    arm_name = "xarm"
    sim = IsaacSimulator(headless=False)
    sim.load_robot_asset(arm_name, hand_name)
    sim.load_object_asset(obj_name)
    sim.add_env("inference", env_info = {"robot":{},
                            "robot_vis":{"right":(arm_name, hand_name)},
                            "object":{},
                            # "object":{obj_name:obj_name},
                            "object_vis":{f"{obj_name}_start":obj_name, f"{obj_name}_end":obj_name}})

    robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf"))

    while not start_event.is_set() and not stop_event.is_set():
        # init_action, success = robot.solve_ik(traj[0], "link6")
        init_action = traj[0]
        last_q = init_action.copy()

        sim.reset("inference", {"robot":{},
                "robot_vis":{"right":init_action.copy()},
                "object":{},
                "object_vis":{f"{obj_name}":pick_6D.copy()}
                })
        
        for i in range(0, len(traj), 3):
            sim.tick()
            time.sleep(1/30)
            # action, success = robot.solve_ik(traj[i], "link6", last_q)
            action = traj[i]
            last_q = action.copy()
            
            action[6:] = hand_traj[i]
            
            sim.step("inference", {"robot":{},
                "robot_vis":{"right":action.copy()},
                "object_vis":{
                    f"{obj_name}_start":pick_6D.copy(),
                    f"{obj_name}_end":place_6D.copy()}
                })
            
            if start_event.is_set() or stop_event.is_set():
                break

    sim.terminate()