from datetime import datetime
import os
import numpy as np
import trimesh

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import detect_aruco
from paradex.utils.path import shared_dir
from paradex.calibration.utils import save_current_camparam, load_current_C2R
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.io.robot_controller.gui_controller_prev import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.path import rsc_path
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.robot.utils import get_robot_urdf_path

rcc = remote_camera_controller("object6d")

def get_object_6d(obj_name, filename):
    img_dict = ImageDict.from_path(os.path.join(shared_dir, "inference", "grasp_eval", filename))
    marker_2d, marker_3d = img_dict.triangulate_markers()
    
    marker_offset = np.load("marker_offset.npy", allow_pickle=True).item()
    marker_id = list(marker_offset.keys())
    A = []
    B = []
    
    for mid in marker_id:
        if mid not in marker_3d or marker_3d[mid] is None:
            continue
        
        A.append(marker_offset[mid])
        B.append(marker_3d[mid])
    
    A = np.concatenate(A)
    B = np.concatenate(B)
    obj_T = SOLVE_XA_B(A, B)
    return obj_T

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

index = "1"
index_path = os.path.join("lookup", "pringles", index)

pick_traj = np.load(f"{index_path}/refined_pick_action.npy")
place_traj = np.load(f"{index_path}/refined_place_action.npy")

pick_hand_traj_tmp = np.load(f"{index_path}/refined_pick_hand.npy")
pick_hand_traj = np.load(f"{index_path}/refined_pick_hand.npy")

pick_hand_traj[:, :4] = pick_hand_traj_tmp[:, 12:] # thumb
pick_hand_traj[:, 4:] = pick_hand_traj_tmp[:, :12] # index

place_hand_traj_tmp = np.load(f"{index_path}/refined_place_hand.npy")
place_hand_traj = np.load(f"{index_path}/refined_place_hand.npy")

place_hand_traj[:, :4] = place_hand_traj_tmp[:, 12:] # thumb
place_hand_traj[:, 4:] = place_hand_traj_tmp[:, :12] # index
place_hand_traj[-70:, :] = np.zeros((70,16))  # open hand at the end

c2r = load_current_C2R()

filename = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(shared_dir, "inference", "grasp_eval", filename), exist_ok=True)

rcc.start("image", False, \
    f"shared_data/inference/grasp_eval/{filename}/raw")
rcc.stop()
rcc.end()

save_current_camparam(os.path.join(shared_dir, "inference", "grasp_eval", filename))

raw_img_dict = ImageDict.from_path(os.path.join(shared_dir, "inference", "grasp_eval", filename))
raw_img_dict.undistort(os.path.join(shared_dir, "inference", "grasp_eval", filename))


obj_T = get_object_6d("pringles", filename)
print("Object 6D pose:\n", obj_T)

obj_path = os.path.join(rsc_path, "object", "pringles", "pringles.obj")
obj_mesh = trimesh.load(obj_path)
obj_mesh.apply_transform(obj_T)

img_dict = ImageDict.from_path(os.path.join(shared_dir, "inference", "grasp_eval", filename))
img_dict.project_mesh(obj_mesh, color=(0,255,0))
img_dict.save(os.path.join(shared_dir, "inference", "grasp_eval", filename, "projected"))

obj_T = np.linalg.inv(c2r) @ obj_T
obj_T = normalize_cylinder(obj_T)

pick_traj = obj_T @ pick_traj
place_traj = obj_T @ place_traj

# 올바른 형식: 각 trajectory는 'arm'과 'hand' 키를 가진 딕셔너리
predefined_traj = {
    "pick": {
        "arm": pick_traj,      # (N, 4, 4)
        "hand": pick_hand_traj  # (N, 16)
    },
    "place": {
        "arm": place_traj,      # (N, 4, 4)
        "hand": place_hand_traj # (N, 16)
    }
}

rgc = RobotGUIController(get_arm("xarm"), get_hand("allegro"), predefined_traj)
rgc.run()