import numpy as np
from paradex.inference.get_lookup_traj import get_traj
from paradex.simulator import IsaacSimulator
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path
import os

arm_name = "xarm"
hand_name = None#"allegro"
robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf"))
link_id = robot.get_link_index("palm_link")

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
            
    robot.compute_forward_kinematics(qpos)
    return np.concatenate([qpos[:6], hand_pose]).astype(np.float32)

bottle_height = 0.039
pick_tx = 0.4
pick_ty = -0.1

place_tx = 0.4
place_ty = 0.2

pick_6D = np.eye(4)
place_6D = np.eye(4)

pick_6D[:3,3] = np.array([pick_tx, pick_ty, bottle_height])
place_6D[:3,3] = np.array([place_tx, place_ty, bottle_height])

demo_idx = 1

pick_traj = np.load(f"lookup/bottle/{demo_idx}/pick.npy")
place_traj = np.load(f"lookup/bottle/{demo_idx}/place.npy")

pick_hand_traj = np.load(f"lookup/bottle/{demo_idx}/pick_hand.npy")
place_hand_traj = np.load(f"lookup/bottle/{demo_idx}/place_hand.npy")

traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)

sim = IsaacSimulator(headless=False)
sim.load_robot_asset("xarm", "allegro")
sim.load_object_asset("bottle")
sim.add_env(env_info = {"robot":{},
                        "robot_vis":{"right":(arm_name, hand_name)},
                        "object":{"bottle":"bottle"},
                        "object_vis":{"bottle":"bottle"}})

init_action = get_action(traj[0], hand_traj[0])
last_q = init_action.copy()

sim.reset(0, {"robot":{},
        "robot_vis":{"right":init_action.copy()},
        "object":{"bottle":pick_6D.copy()},
        "object_vis":{"bottle":pick_6D.copy()}
        })

for i in range(len(traj)):
    sim.tick()
    
    action = get_action(traj[i], hand_traj[i], last_q)
    last_q = action.copy()
    
    sim.step(0, {"robot":{},
        "robot_vis":{"right":action.copy()},
        "object_vis":{"bottle":place_6D.copy()}
        })

sim.terminate()