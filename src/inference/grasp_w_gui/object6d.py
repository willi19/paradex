
from datetime import datetime
import os
import numpy as np
import trimesh
import transforms3d

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import detect_aruco
from paradex.utils.path import shared_dir
from paradex.calibration.utils import save_current_camparam, load_current_C2R
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.io.robot_controller.gui_controller import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.path import rsc_path
from paradex.robot.robot_wrapper_deprecated import RobotWrapper
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

for index in [1]:
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
    
    data = np.load(f"bodex/scale010_grasp.npy",allow_pickle=True).item()
    qpos_tmp = data['robot_pose'][0, index]
    
    trans = qpos_tmp[0][:3]
    quat = qpos_tmp[0][3:7]
    rotmat = transforms3d.quaternions.quat2mat(quat) 
    wrist_6d = np.eye(4)
    wrist_6d[:3, :3] = rotmat
    wrist_6d[:3, 3] = trans
            
    qpos = np.zeros((3, 16))
    qpos_tmp = qpos_tmp[:, 7:]
    qpos[:, :4] = qpos_tmp[:, 12:]
    qpos[:, 4:] = qpos_tmp[:, :12]
    
    # wrist_6d = np.load(f"dexgraspnet/results/pringles/{index}/wrist_6d.npy")
    wrist_6d = obj_T @ wrist_6d
    
    robot = RobotWrapper(get_robot_urdf_path(arm_name="xarm", hand_name="allegro"))
    q, succ = robot.solve_ik(wrist_6d, "palm_link")
    pick_action = robot.compute_forward_kinematics(q, ["link6"])["link6"]

    squeezed_qpos = qpos[1, :] * 8 - qpos[0, :] * 7
    rgc = RobotGUIController(get_arm("xarm"), get_hand("allegro"), {"grasp":pick_action}, {"start": np.zeros(16), "pregrasp": qpos[0], "grasp": qpos[1], 'squeezed': squeezed_qpos})
    rgc.run()