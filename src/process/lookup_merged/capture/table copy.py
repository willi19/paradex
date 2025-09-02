import numpy as np
import os
import time
from scipy.spatial.transform import Rotation

from paradex.utils.file_io import shared_dir, get_robot_urdf_path
from paradex.simulator import IsaacSimulator
from paradex.robot.mimic_joint import parse_inspire
from paradex.geometry.coordinate import DEVICE2WRIST
from paradex.robot.robot_wrapper import RobotWrapper

hand_name = "allegro"
arm_name = "xarm"
obj_name = "pringles_light"

robot = RobotWrapper(get_robot_urdf_path(arm_name, hand_name))
robot.compute_forward_kinematics(np.zeros(22))
wrist_pos = robot.get_link_pose(robot.get_link_index("wrist"))
last_link_pos = robot.get_link_pose(robot.get_link_index("link6"))
LINK2WRIST = np.linalg.inv(last_link_pos) @ wrist_pos

demo_path = os.path.join(shared_dir, "capture", "lookup", obj_name)
demo_name_list = os.listdir(demo_path)

sim = IsaacSimulator(headless=False, plane=True)
sim.load_robot_asset(None, hand_name)
sim.load_object_asset("bottle")

env_list = []
action_dict = {}
object_dict = {}


for demo_name in ["18"]:# demo_name_list[:2]:
    # if os.path.exists(os.path.join(shared_dir, "capture", "lookup", obj_name, demo_name, "pick.mp4")):
    #     continue
    for type in ["place"]:
        env_list.append((demo_name, type))
        env_name = f"{demo_name}/{type}"
        
        hand_qpos = np.load(os.path.join(demo_path, str(demo_name), f"{type}_hand.npy"))
        wrist_pos = np.load(os.path.join(demo_path, str(demo_name), f"{type}.npy"))
        obj_pos = np.load(os.path.join(demo_path, str(demo_name), f"{type}_obj_T.npy"))
        obj_pose = obj_pos[0] if type == "pick" else obj_pos[-1]
        #obj_pos[:,:2,3] -= obj_pos[0,:2,3]
        # print(obj_pos[:10,2,3])
        T = wrist_pos.shape[0]
        action = np.zeros((T, 22 if hand_name == "allegro" else "inspire"))
        
        for i in range(T):
            print(wrist_pos[i], "asdf")
            wrist_pos[i] = obj_pose @ wrist_pos[i] @ LINK2WRIST
            euler = Rotation.from_matrix(wrist_pos[i,:3,:3]).as_euler('zyx')
            
            action[i, 5] = euler[0]
            action[i, 4] = euler[1]
            action[i, 3] = euler[2]
            
        action[:,:3] = wrist_pos[:, :3, 3]
        
        if hand_name == "inspire":
            hand_qpos = parse_inspire(hand_qpos)
        action[:, 6:] = hand_qpos
        
        action_dict[env_name] = action.copy()
        object_dict[env_name] = obj_pos.copy()
        
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
                                "robot_vis":{"new":(None, hand_name)},
                                "object":{},
                                "object_vis":{"bottle":"bottle"}})
        
        sim.reset(env_name, {"robot":{},
                "robot_vis":{"new":action_dict[env_name][0]},
                "object":{},
                "object_vis":{"bottle":object_dict[env_name][0].copy()}
                })
        
        sim.load_camera(env_name)
        sim.set_videopath(env_name, os.path.join("capture","lookup", obj_name))
        
        cur_env_idx += 1
        cur_idx[env_name] = 0
    
    if cur_env_idx >= len(env_list) and len(cur_idx) == 0:
        cur_env_idx = 0
    
    for env_name in list(cur_idx.keys()):
        if action_dict[env_name].shape[0] <= cur_idx[env_name]:
            sim.destroy_env(env_name)
            del cur_idx[env_name]
            continue
        
    for env_name, idx in cur_idx.items():
        sim.step(env_name, {"robot":{},
                "robot_vis":{"new":action_dict[env_name][idx].copy()},
                "object_vis":{"bottle":object_dict[env_name][idx].copy()}
                })

        cur_idx[env_name] += 1        
                
    sim.tick()
    time.sleep(max(0, 1/30 - (time.time()-start_time)))
    print(cur_env_idx, cur_idx)
    

sim.terminate()
