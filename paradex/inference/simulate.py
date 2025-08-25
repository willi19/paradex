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
                            "object":{"bottle":"bottle"},
                            "object_vis":{"bottle_start":"bottle", "bottle_end":"bottle"}})

    robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf"))

    while not start_event.is_set() and stop_event.is_set():
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