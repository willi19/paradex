import numpy as np
import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from paradex.image.aruco import detect_aruco, triangulate_marker, draw_aruco
from paradex.utils.file_io import shared_dir, find_latest_directory, rsc_path, load_c2r, get_robot_urdf_path
from paradex.geometry.math import rigid_transform_3D
from paradex.robot import RobotWrapper
from paradex.utils.file_io import eef_calib_path, load_camparam
from paradex.image.projection import get_cammtx
from paradex.image.undistort import undistort_img
from paradex.geometry.Tsai_Lenz import solve, solve_axb_pytorch
from paradex.geometry.conversion import project
from paradex.geometry.coordinate import DEVICE2WRIST
from paradex.visualization_.renderer import BatchRenderer
from paradex.visualization_.robot_module import Robot_Module
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.image.overlay import overlay_mask

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(eef_calib_path)
    
eef_calib_path = os.path.join(shared_dir, "eef", args.name)
index_list = os.listdir(os.path.join(eef_calib_path))

robot = RobotWrapper(
    os.path.join(rsc_path, "robot", "allegro", "allegro.urdf")
)
arm = RobotWrapper(
    os.path.join(rsc_path, "robot", "xarm.urdf")
)
finger_index = {f"{finger_name}_proximal":robot.get_link_index(f"{finger_name}_proximal") for finger_name in ["thumb", "index", "middle", "ring"]}

# finger_index["index_proximal"] = 8
# finger_index["middle_proximal"] = 8
# finger_index["ring_proximal"] = 28

finger_id_list = [11,13,14]
finger_marker = {11:"ring_proximal", 14:"middle_proximal", 13:"index_proximal", 10:"thumb_proximal"}

c2r = load_c2r(os.path.join(eef_calib_path, "0"))
    
link_index = arm.get_link_index("link6")
intrinsic, extrinsic = load_camparam(os.path.join(eef_calib_path, "0"))
cammtx = get_cammtx(intrinsic, extrinsic)

robot_cor = []
cam_cor = []
hand_action = []
qpos = []

for idx in index_list:
    # robot_cor.append(np.load(os.path.join(eef_calib_path, idx, "robot.npy")))
    hand_action.append(np.load(os.path.join(eef_calib_path, idx, "hand.npy")))
    qpos.append(np.load(os.path.join(eef_calib_path, idx, "qpos.npy")))
    
    arm.compute_forward_kinematics(qpos[-1])
    robot_cor.append(arm.get_link_pose(link_index))
    
    if not os.path.exists(os.path.join(eef_calib_path, idx, "marker_3d.npy")):        
        img_dir = os.path.join(eef_calib_path, idx, "image")
        
        img_dict = {}
        for img_name in os.listdir(img_dir):
            img_dict[img_name.split(".")[0]] = cv2.imread(os.path.join(img_dir, img_name))
        
        cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)
        for serial_num, img in img_dict.items():
            if serial_num not in cammtx:
                continue
            
            undist_img = undistort_img(img, intrinsic[serial_num])
            
            os.makedirs(os.path.join(eef_calib_path, idx, "undistort"), exist_ok=True)
            cv2.imwrite(os.path.join(eef_calib_path, idx, "undistort", f"{serial_num}.png"), undist_img)
            
            undist_kypt, ids = detect_aruco(undist_img)
            
            if ids is None:
                continue
            draw_aruco(undist_img, undist_kypt, ids, (0, 0, 255))
            
            for mid in finger_id_list:
                if mid not in ids or cor_3d[mid] is None:
                    continue
                pt_2d = project(cammtx[serial_num], cor_3d[mid])
                draw_aruco(undist_img, [pt_2d], None, (255, 0, 0))
            
            os.makedirs(os.path.join(eef_calib_path, idx, "debug"), exist_ok=True)
            cv2.imwrite(os.path.join(eef_calib_path, idx, "debug", f"{serial_num}.png"), undist_img)

        marker_3d = {}
        for mid in finger_id_list:
            if mid not in cor_3d or cor_3d[mid] is None:
                continue
            marker_3d[mid] = cor_3d[mid]
        np.save(os.path.join(eef_calib_path, idx, "marker_3d.npy"), marker_3d)
    
    else:
        marker_3d = np.load(os.path.join(eef_calib_path, idx, "marker_3d.npy"),allow_pickle=True).item()
    cam_cor.append(marker_3d)

A_list = []
B_list = []
finger_name = []
index_name = []

ans = np.linalg.inv(DEVICE2WRIST["xarm"]) @ DEVICE2WRIST["allegro"]
for i in range(len(index_list)-1):
    T_r1 = robot_cor[i]
    T_r2 = robot_cor[i+1]
    
    marker_3d1 = cam_cor[i]
    marker_3d2 = cam_cor[i+1]
    
    hand_action1 = hand_action[i]
    hand_action2 = hand_action[i+1]

    for finger_id in finger_id_list:
        if finger_id not in marker_3d1 or finger_id not in marker_3d2:
            continue
        
        if marker_3d1[finger_id] is None or marker_3d2[finger_id] is None:
            continue
        
        A = rigid_transform_3D(marker_3d2[finger_id], marker_3d1[finger_id])
        
        robot.compute_forward_kinematics(hand_action1)
        T_h1 = robot.get_link_pose(finger_index[finger_marker[finger_id]])

        robot.compute_forward_kinematics(hand_action2)
        T_h2 = robot.get_link_pose(finger_index[finger_marker[finger_id]])
        
        A = np.linalg.inv(T_r1) @ np.linalg.inv(c2r) @ A @ c2r @ T_r2
        B = T_h1 @ np.linalg.inv(T_h2)

        A_list.append(A)
        B_list.append(B)
        finger_name.append(finger_marker[finger_id])
        index_name.append(i)
        
X = np.eye(4)
theta, b_x = solve(A_list, B_list)
X[0:3, 0:3] = theta
X[0:3, -1] = b_x.flatten()
X, loss = solve_axb_pytorch(A_list, B_list,X.copy(),learning_rate=0.001)
print(X)

for i in range(len(index_list)-1):
    # print(A_list[i] @ X - X @ B_list[i], "error")
    # print(A_list[i] @ ans - ans @ B_list[i], index_name[i], finger_name[i])
    print(A_list[i] @ X - X @ B_list[i], index_name[i], finger_name[i])

extrinsic_list = []
intrinsic_list = []

serial_list = os.listdir(os.path.join(eef_calib_path, idx, "image"))
serial_list.sort()

for serial_name in serial_list:
    sn = serial_name.split(".")[0]
    extmat = extrinsic[sn]
    extrinsic_list.append(extmat @ c2r)        
    intrinsic_list.append(intrinsic[sn]['intrinsics_undistort'])

hand_action = np.array(hand_action)
qpos = np.array(qpos)

qpos = np.concatenate([qpos, hand_action], axis=1)
rm = Robot_Module(get_robot_urdf_path("xarm", "allegro"), state=qpos)
print(rm.link_list)
renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')

for fid in index_list:
    img_dir = os.path.join(eef_calib_path, fid, "undistort")
    robot_mesh = rm.get_mesh(int(fid))
    img_dict = {img_name:cv2.imread(os.path.join(img_dir, img_name)) for img_name in serial_list}
    os.makedirs(os.path.join(eef_calib_path, fid, "overlay"), exist_ok=True)
    
    for mesh in robot_mesh:
        frame, mask = project_mesh_nvdiff(mesh, renderer)
        mask = mask.detach().cpu().numpy()[:,:,:,0]
        
        for i, img_name in enumerate(serial_list):
            img_dict[img_name] = overlay_mask(img_dict[img_name], mask[i], 0.3, (0, 255, 0))
    
    for img_name in serial_list:
        cv2.imwrite(os.path.join(eef_calib_path, fid, "overlay", img_name), img_dict[img_name])    
        