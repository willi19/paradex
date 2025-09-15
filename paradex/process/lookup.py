import numpy as np
import os
from scipy.spatial.transform import Rotation
import copy
import trimesh
import time

from paradex.robot.robot_wrapper import RobotWrapper
from paradex.utils.file_io import get_robot_urdf_path, load_eef, load_latest_eef, shared_dir, rsc_path, download_dir
from paradex.robot.mimic_joint import parse_inspire
from paradex.visualization.open3d_viewer import Open3DVideoRenderer
from paradex.utils.upload_file import copy_file
from paradex.video.convert_codec import change_to_h264

obj_type = {
    "pringles":"cylinder",
    "pringles_heavy":"cylinder",
    "pringles_light":"cylinder",
    "book":"box"
    }

RENDER_CONFIG = {
    "width": 1280,
    "height": 720, 
    "fps": 30,
    "camera_eye": [-0.3, 0.0, 0.2],  # 카메라 위치
}

def normalize_cylinder(obj_6D):
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

def normalize_box(obj_6D):
    ret = obj_6D.copy()
    if obj_6D[2, 1] < -0.8:
        ret[:, 0] *= -1
        ret[:, 1] *= -1
    if ret[2, 1] > 0.8:
        if ret[:3, 2] @ ret[:3, 3] < 0:
            ret[:,0] *= -1
            ret[:,2] *= -1
    
    else:
        if obj_6D[2, 0] < 0:
            ret[:,0] *= -1
            ret[:,2] *= -1
    return ret

def normalize(obj_6D, obj_name):
    if obj_name not in list(obj_type.keys()):
        raise NotImplementedError
    
    type = obj_type[obj_name]
    if type == "cylinder":
        return normalize_cylinder(obj_6D)
    elif type == "box":
        return normalize_box(obj_6D)

def get_robot_name(demo_path):
    arm_name = None
    hand_name = None
    
    raw_path = os.path.join(demo_path, "raw")
    
    for an in ["xarm", "franka"]:
        if an in os.listdir(raw_path):
            arm_name = an
            break
    
    for hn in ["allegro", "inspire"]:
        if hn in os.listdir(raw_path):
            hand_name = hn
            break
    
    return arm_name, hand_name

def get_obj_name(demo_path):
    return os.path.basename(os.path.dirname(demo_path))

def visualize_lookup_table(demo_path, logger=[], overwrite=False):
    arm_name, hand_name = get_robot_name(demo_path)
    obj_name = get_obj_name(demo_path)
    demo_name = os.path.basename(demo_path)
    mesh = trimesh.load(os.path.join(rsc_path, "object", obj_name, obj_name+".obj"))
    
    try:
        LINK2WRIST = load_eef(demo_path)
    except:
        LINK2WRIST = load_latest_eef()
        np.save(os.path.join(demo_path, "eef.npy"), LINK2WRIST)
    
    for demo_type in ["pick", "place"]:
        logger.append({"root_dir":demo_path, "time":time.time(), "state":"processing", "msg":f"visualize lookup table {demo_type}", "type":"process_msg"})
        if not overwrite and os.path.exists(os.path.join(shared_dir, f"{demo_type}.mp4")):
            continue
        
        hand_qpos = np.load(os.path.join(demo_path, f"{demo_type}_hand.npy"))
        wrist_pos = np.load(os.path.join(demo_path, f"{demo_type}_action.npy"))
        obj_pos = np.load(os.path.join(demo_path, f"{demo_type}_objT.npy"))
        
        obj_pose = obj_pos[0].copy() if demo_type == "pick" else obj_pos[-1].copy()
        
        obj_pos[:,:2, 3] -= obj_pose[:2, 3]
        obj_pose[:2, 3] = 0
        
        T = wrist_pos.shape[0]
        action = np.zeros((T, 22 if hand_name == "allegro" else "inspire"))
        
        # 액션 데이터 변환
        for i in range(T):
            wrist_pos[i] = obj_pose @ wrist_pos[i] @ LINK2WRIST
            euler = Rotation.from_matrix(wrist_pos[i,:3,:3]).as_euler('zyx')
            
            action[i, 5] = euler[0]
            action[i, 4] = euler[1] 
            action[i, 3] = euler[2]
            
        action[:,:3] = wrist_pos[:, :3, 3]
        
        if hand_name == "inspire":
            hand_qpos = parse_inspire(hand_qpos)
        action[:, 6:] = hand_qpos
        try:
            renderer = Open3DVideoRenderer(
                obj_mesh=copy.deepcopy(mesh),
                obj_T=obj_pos,
                urdf_path=get_robot_urdf_path(None, hand_name),
                qpos=action,
                width=RENDER_CONFIG["width"],
                height=RENDER_CONFIG["height"],
                fps=RENDER_CONFIG["fps"]
            )
            
            download_root_dir = demo_path.replace(shared_dir, download_dir)
            output_tmp_path = os.path.join(download_root_dir, f"{demo_type}_tmp.mp4")
            output_path = os.path.join(download_root_dir, f"{demo_type}.mp4")
            os.makedirs(download_root_dir, exist_ok=True)
            renderer.render_video(
                output_path=output_tmp_path,
                camera_eye=RENDER_CONFIG["camera_eye"],
                logger=logger
            )
            renderer = None
            
            print(logger)

        except Exception as e:
            error_msg = f"Error rendering {demo_path}/{demo_type}: {str(e)}"
            print(f"❌ {error_msg}")
        
        change_to_h264(output_tmp_path, output_path)
        copy_file(output_path, os.path.join(demo_path, f"{demo_type}.mp4"))
            
def generate_lookup_table(demo_path):
    result = {}
    obj_name = os.path.basename(os.path.dirname(demo_path))
    
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
            
    pick_6D = normalize(orig_pick_6D.copy(), obj_name)
    place_6D = normalize(place_6D_orig.copy(), obj_name)
        
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
        place_state[i] = np.linalg.inv(place_6D) @ place_state[i]

        
        if np.linalg.norm(place_objT[i]) < 0.1:
            place_objT[i] = place_objT[i-1].copy()
        else:
            place_objT[i] = place_objT[i] @ place_6D_diff
    
    result["pick"] = {"state":pick_state, "action":pick, "objT":pick_objT, "hand":pick_hand_action}
    result["place"] = {"state":place_state, "action":place, "objT":place_objT, "hand":place_hand_action}
    for type in ["pick", "place"]:
        for data_name, data in result[type].items():
            np.save(f"{demo_path}/{type}_{data_name}.npy", data)
    return result

    