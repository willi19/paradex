import numpy as np
import os
import time
from scipy.spatial.transform import Rotation

from paradex.utils.file_io import shared_dir, download_dir, eef_calib_path, load_latest_eef
from paradex.simulator import IsaacSimulator
from paradex.robot.mimic_joint import parse_inspire
from paradex.geometry.coordinate import DEVICE2WRIST

hand_name = "allegro"
arm_name = "xarm"
obj_name = "pringles_light"

# LINK2WRIST = np.linalg.inv(DEVICE2WRIST["xarm"]) @ DEVICE2WRIST[hand_name]
LINK2WRIST = load_latest_eef()
demo_name = "13"

demo_path = os.path.join(shared_dir, "capture", "lookup", obj_name, demo_name)

sim = IsaacSimulator(headless=False, plane=True)

sim.load_robot_asset(None, hand_name)
sim.load_robot_asset(arm_name, hand_name)

sim.load_object_asset("bottle")

env_list = []
action_dict = {}
object_dict = {}

# hand_qpos = np.load(os.path.join(demo_path, "raw", hand_name, f"qpos.npy"))
wrist_pos = np.load(os.path.join(demo_path, "raw", arm_name, "action.npy"))
arm_qpos = np.load(os.path.join(demo_path, "raw", arm_name, "position.npy"))

hand_qpos = np.load(os.path.join(demo_path, "hand", "qpos.npy"))
wrist_pos = np.load(os.path.join(demo_path, "raw", arm_name, "action.npy"))
arm_qpos = np.load(os.path.join(demo_path, "raw", arm_name, "position.npy"))

T = wrist_pos.shape[0]
wrist_qpos = np.zeros((T, 6))
hand_qpos = np.zeros((T, 16))

for i in range(T):
    wrist_pos[i] = wrist_pos[i] @ LINK2WRIST
    euler = Rotation.from_matrix(wrist_pos[i,:3,:3]).as_euler('zyx')
    
    wrist_qpos[i, 5] = euler[0]
    wrist_qpos[i, 4] = euler[1]
    wrist_qpos[i, 3] = euler[2]
wrist_qpos[:,:3] = wrist_pos[:, :3, 3]

action = np.concatenate([wrist_qpos, hand_qpos], axis=1)
state = np.concatenate([arm_qpos, hand_qpos], axis=1)

object_T = np.load(os.path.join(demo_path, f"obj_T.npy"), allow_pickle=True).item()[obj_name]
idx = 0
env_name = "fuck"

sim.add_env(env_name,env_info = {"robot":{},
                                "robot_vis":{"hand":(None, hand_name), "arm":(arm_name, hand_name)},
                                "object":{},
                                "object_vis":{}#"bottle":"bottle"}
                                })
        
sim.reset(env_name, {"robot":{},
        "robot_vis":{"hand":action[0], "arm":state[0]},
        "object":{},
        "object_vis":{}#"bottle":object_T[0].copy()}
        })

for idx in range(0, T, 10):
    sim.step(env_name, {"robot":{},
            "robot_vis":{"hand":action[idx].copy(), "arm":state[idx].copy()},
            "object_vis":{}#"bottle":object_T[idx].copy()}
            })

    sim.tick()
    time.sleep(1/120)
    
sim.terminate()
