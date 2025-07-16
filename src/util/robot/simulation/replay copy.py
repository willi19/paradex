import numpy as np
from paradex.inference.get_lookup_traj import get_traj
from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path
import os
import pickle
import time

def get_action(palm_pose, hand_pose, last_qpos=None):
    if last_qpos is None:
        success = False
        while not success:
            qpos, success = robot.solve_ik(
                palm_pose,"palm_link", None, 1000
            )
    else:
        qpos, success = robot.solve_ik(
            palm_pose,"palm_link",last_qpos.copy(),1000
        )
        while not success:
            qpos, success = robot.solve_ik(
                palm_pose,"palm_link", None, 1000
            )
    
    return np.concatenate([qpos[:6], hand_pose]).astype(np.float32)

arm_name = "xarm"
hand_name = "allegro"
# robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf"))
robot = RobotWrapper(os.path.join(rsc_path, "robot", "xarm_allegro.urdf"))
link_id = robot.get_link_index("palm_link")

prev_urdf_file = os.path.join(rsc_path, "robot", "xarm_allegroprev.urdf")
robot_prev = RobotWrapper(prev_urdf_file)
link_id_prev = robot_prev.get_link_index("palm_link")

demo_path = f"{shared_dir}/data_Icra/teleoperation/bottle"
demo_name = "1"

obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

sim = IsaacSimulator(headless=False)
sim.load_robot_asset("xarm", "allegro")
sim.load_robot_asset("xarm", "allegroprev")
sim.load_object_asset("bottle")
sim.add_env(env_info = {"robot":{},
                        "robot_vis":{"new":(arm_name, hand_name), "prev":(arm_name, "allegroprev")},
                        "object":{"bottle":"bottle"},
                        "object_vis":{"bottle":"bottle"}})

while True:
    robot_prev.compute_forward_kinematics(robot_traj[0].copy())
    palm_pose = robot_prev.get_link_pose(link_id_prev)
    
    hand_pose = target_traj[0][6:]
        
    action = get_action(palm_pose, hand_pose)
    robot.compute_forward_kinematics(action)
    palm_pose2 = robot.get_link_pose(link_id)
    
    sim.reset(0, {"robot":{},
            "robot_vis":{"new":action.copy(), "prev":robot_traj[0].copy()},
            "object":{"bottle":obj_T[0].copy()},
            "object_vis":{"bottle":obj_T[0].copy()}
            })
    T = len(target_traj)
    for i in range(T):
        sim.tick()
        robot_prev.compute_forward_kinematics(robot_traj[i].copy())
        palm_pose = robot_prev.get_link_pose(link_id_prev)
        
        hand_pose = target_traj[i][6:]
        
        action = get_action(palm_pose, hand_pose, action.copy())
        
        sim.step(0, {"robot":{},
            "robot_vis":{"new":action.copy(), "prev":robot_traj[i].copy()},
            "object_vis":{"bottle":obj_T[i].copy()}
            })
    sim.tick()

sim.terminate()