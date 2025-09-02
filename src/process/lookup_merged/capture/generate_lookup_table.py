import os
import numpy as np
import open3d as o3d
import argparse

from paradex.utils.file_io import rsc_path, shared_dir, load_c2r, get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper

def normalize(obj_6D):
    ret = obj_6D.copy()
    
    if obj_6D[2, 2] < 0.7:
        z = np.array([obj_6D[0,2], obj_6D[1, 2], 0])
        if z[0] == -1:
            z *= -1
        z /= np.linalg.norm(z)
        ret[:3, 2] = z
        ret[:3, 0] = np.array([0,0,1])
        ret[:3, 1] = np.array([z[1],-z[0],0])
    else:
        ret[:3, :3] = np.eye(3)
        
    return ret

def load_demo(demo_path):
    result = {}
    
    last_link_pose = np.load(os.path.join(demo_path, "arm", "action.npy"))
    arm_qpos = np.load(os.path.join(demo_path, "arm", "qpos.npy"))
    arm_state = []
    
    robot = RobotWrapper(get_robot_urdf_path("xarm"))
    link_id = robot.get_link_index("link6")
    
    for i in range(arm_qpos.shape[0]):
        robot.compute_forward_kinematics(arm_qpos[i])
        arm_state.append(robot.get_link_pose(link_id))
    
    arm_state = np.array(arm_state)
    hand_qpos = np.load(os.path.join(demo_path, "hand", "action.npy"))
    
    obj_T_dict = np.load(os.path.join(demo_path, "obj_T.npy"), allow_pickle=True).item()
    obj_T = list(obj_T_dict.values())[0]
    
    T = min(obj_T.shape[0], hand_qpos.shape[0])
    split_t = -1
    max_h = -1
    orig_pick_6D = np.zeros((4,4))
    for step in range(T):
        if np.linalg.norm(obj_T[step]) < 0.1:
            continue
        if np.linalg.norm(orig_pick_6D) < 0.1:
            orig_pick_6D = obj_T[step].copy()
        place_6D_orig = obj_T[step].copy()
        if obj_T[step, 2, 3] > max_h:
            max_h = obj_T[step, 2, 3]
            split_t = step
    pick_6D = normalize(orig_pick_6D.copy())
    place_6D = normalize(place_6D_orig.copy())
        
    if np.linalg.norm(obj_T[0]) < 0.1:        
        obj_T[0] = orig_pick_6D.copy()
    
    if np.linalg.norm(obj_T[-1]) < 0.1:
        obj_T[-1] = place_6D_orig.copy()
        
    pick_6D_diff = np.linalg.inv(orig_pick_6D) @ pick_6D
    place_6D_diff = np.linalg.inv(place_6D_orig) @ place_6D
    
    place_T = min(last_link_pose.shape[0], obj_T.shape[0])
    
    pick_hand_action = hand_qpos[:split_t]
    place_hand_action = hand_qpos[split_t:place_T]
    
    pick = last_link_pose[:split_t]
    place = last_link_pose[split_t:place_T]
    
    pick_state = arm_state[:split_t]
    place_state = arm_state[split_t:place_T]
    
    pick_objT = obj_T[:split_t]
    place_objT = obj_T[split_t:place_T]
    
    for i in range(len(pick_hand_action)):
        pick[i] = np.linalg.inv(pick_6D) @ pick[i]
        pick_state[i] = np.linalg.inv(pick_6D) @ pick_state[i]
        
        if np.linalg.norm(pick_objT[i]) < 0.1:
            pick_objT[i] = pick_objT[i-1].copy()
        else:
            pick_objT[i] = pick_objT[i] @ pick_6D_diff
    
    for i in range(len(place_hand_action)):
        place[i] = np.linalg.inv(place_6D) @ place[i]
        
        
        if np.linalg.norm(place_objT[i]) < 0.1:
            place_objT[i] = place_objT[i-1].copy()
        else:
            place_objT[i] = place_objT[i] @ place_6D_diff
    
    result["pick"] = {"state":pick_state, "action":pick, "objT":pick_objT}
    result["place"] = {"state":place_state, "action":place, "objT":place_objT}
    
    return result

def plot_table():
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--obj', nargs="+", type=str, default=None)
args = parser.parse_args()

process_list = []

if args.obj == None:
    name_list = os.listdir(os.path.join(shared_dir, 'capture', "lookup"))
    name_list.sort()

else:
    name_list = args.obj

print(name_list)
for name in name_list:
    vis_list = []
    index_list = os.listdir(os.path.join(shared_dir, 'capture', "lookup", name))
    for index in index_list:
        demo_path = os.path.join(shared_dir, 'capture', "lookup", name, index)
        # if os.path.exists(os.path.join(demo_path, "pick.npy")) and os.path.exists(os.path.join(demo_path, "place.npy")):
        #     continue
        print(demo_path)
        result = load_demo(demo_path)
        
        for type in ["pick", "place"]:
            for data_name, data in result[type].items():
                np.save(f"{demo_path}/{type}_{data_name}.npy", data)