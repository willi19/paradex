import numpy as np
import os
import time
from scipy.spatial.transform import Rotation

from paradex.simulator import IsaacSimulator
from paradex.robot.mimic_joint import parse_inspire
from paradex.geometry.coordinate import DEVICE2WRIST

hand_name = "inspire"
arm_name = "xarm"

LINK2WRIST = np.linalg.inv(DEVICE2WRIST["xarm"]) @ DEVICE2WRIST["inspire"]
height_offset = 0.096-0.0525

demo_path = os.path.join("data","lookup", "pringles", "stand_free")
demo_name_list = os.listdir(demo_path)

sim = IsaacSimulator(headless=False)
sim.load_robot_asset(None, "inspire")
sim.load_object_asset("bottle")

env_list = []
action_dict = {}

obj_pose = np.eye(4)
obj_pose[2,3] += height_offset

for demo_name in demo_name_list:
    for type in ["pick", "place"]:
        env_list.append((demo_name, type))
        env_name = f"{demo_name}/{type}"
        
        hand_qpos = np.load(os.path.join(demo_path, str(demo_name), f"{type}_hand.npy"))
        wrist_pos = np.load(os.path.join(demo_path, str(demo_name), f"{type}.npy"))
        
        T = wrist_pos.shape[0]
        action = np.zeros((T, 18))
        
        for i in range(T):
            wrist_pos[i] = wrist_pos[i] @ LINK2WRIST
            euler = Rotation.from_matrix(wrist_pos[i,:3,:3]).as_euler('zyx')
            
            action[i, 5] = euler[0]
            action[i, 4] = euler[1]
            action[i, 3] = euler[2]
            
        action[:,:3] = wrist_pos[:, :3, 3]
        action[:,2] += height_offset
        action[:, 6:] = parse_inspire(hand_qpos)
        
        action_dict[env_name] = action.copy()
        
cur_env_idx = 0
max_env_num = 50
cur_idx = {}

while True:
    start_time = time.time()
    while cur_env_idx < len(env_list) and len(cur_idx) < max_env_num:
        # add environment
        env_data = env_list[cur_env_idx]
        env_name = f"{env_data[0]}/{env_data[1]}"
        sim.add_env(env_name,env_info = {"robot":{},
                                "robot_vis":{"new":(None, "inspire")},
                                "object":{},
                                "object_vis":{"bottle":"bottle"}})
        
        sim.reset(env_name, {"robot":{},
                "robot_vis":{"new":action_dict[env_name][0]},
                "object":{},
                "object_vis":{"bottle":obj_pose}
                })
        
        sim.load_camera(env_name)
        sim.set_videopath(env_name, "test")
        
        cur_env_idx += 1
        cur_idx[env_name] = 0
    
    if cur_env_idx >= len(env_list) and len(cur_idx) == 0:
        break
    
    for env_name in list(cur_idx.keys()):
        print(action_dict[env_name].shape[0], cur_idx[env_name])
        if action_dict[env_name].shape[0] <= cur_idx[env_name]:
            sim.destroy_env(env_name)
            del cur_idx[env_name]
            continue
        
    for env_name, idx in cur_idx.items():
        sim.step(env_name, {"robot":{},
                "robot_vis":{"new":action_dict[env_name][idx].copy()},
                "object_vis":{"bottle":obj_pose}
                })

        cur_idx[env_name] += 1        
                
    sim.tick()
    time.sleep(max(0, 1/30 - (time.time()-start_time)))
    print(cur_env_idx, len(cur_idx))
    
sim.terminate()
