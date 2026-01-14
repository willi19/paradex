import os
import numpy as np
import trimesh
from datetime import datetime

from paradex.robot.robot_wrapper_deprecated import RobotWrapper
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.utils import get_robot_urdf_path
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import detect_aruco
from paradex.utils.path import shared_dir
from paradex.calibration.utils import save_current_camparam, load_current_C2R
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.io.robot_controller.gui_controller_prev import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.path import rsc_path


def get_object_6d(obj_name, filename):
    img_dict = ImageDict.from_path(os.path.join(shared_dir, "inference", "grasp_eval", filename))
    marker_2d, marker_3d = img_dict.triangulate_markers()
    
    marker_offset = np.load(os.path.join(shared_dir, "object", "marker_offset", obj_name, "0", "marker_offset.npy"), allow_pickle=True).item()
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

c2r = load_current_C2R()
    
filename = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(shared_dir, "inference", "grasp_eval", filename), exist_ok=True)

rcc = remote_camera_controller("object6d")
rcc.start("image", False, \
    f"shared_data/inference/grasp_eval/{filename}/raw")
rcc.stop()
rcc.end()

save_current_camparam(os.path.join(shared_dir, "inference", "grasp_eval", filename))

raw_img_dict = ImageDict.from_path(os.path.join(shared_dir, "inference", "grasp_eval", filename))
raw_img_dict.undistort(os.path.join(shared_dir, "inference", "grasp_eval", filename))


obj_T = get_object_6d("pringles", filename)
obj_path = os.path.join(rsc_path, "object", "pringles", "pringles.obj")
obj_mesh = trimesh.load(obj_path)
obj_T = np.linalg.inv(c2r) @ obj_T
print("Object 6D pose:\n", obj_T)

index = 6

vis = ViserViewer()

wrist_6d = np.load(f"dexgraspnet/results/pringles/{index}/wrist_6d.npy")

wrist_6d = obj_T @ wrist_6d
qpos_tmp = np.load(f"dexgraspnet/results/pringles/{index}/qpos.npy")

qpos = np.zeros(16)
qpos[0:4] = qpos_tmp[12:16]
qpos[4:] = qpos_tmp[0:12]

robot = RobotWrapper(get_robot_urdf_path(arm_name="xarm", hand_name="allegro"))
q, succ = robot.solve_ik(wrist_6d, "palm_link")

if not succ:
    raise ValueError("IK solution not found")
qtot = np.concatenate([q[:6], qpos])

vis.add_robot("robot", get_robot_urdf_path(arm_name="xarm", hand_name="allegro"))
vis.add_object("pringles", obj_mesh, obj_T)
vis.add_traj("robot", {"robot": np.array([qtot])})
vis.start_viewer()

for i in range(10):
    qpos = np.load(f"dexgraspnet/results/pringles/{index}/qpos.npy")
    print(qpos)