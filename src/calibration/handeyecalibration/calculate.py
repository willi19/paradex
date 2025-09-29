import os
import cv2
import argparse
import numpy as np
import json
from copy import deepcopy

from paradex.image.aruco import detect_charuco, merge_charuco_detection
from paradex.image.undistort import undistort_img
from paradex.geometry.Tsai_Lenz import solve, solve_axb_pytorch
from paradex.geometry.conversion import project, to_homo
from paradex.utils.file_io import handeye_calib_path, find_latest_directory, load_camparam, config_dir, get_robot_urdf_path
from paradex.image.projection import get_cammtx
from paradex.geometry.math import rigid_transform_3D
from paradex.geometry.triangulate import ransac_triangulation
from paradex.image.aruco import draw_charuco
from paradex.visualization_.renderer import BatchRenderer
from paradex.visualization_.robot_module import Robot_Module
from paradex.image.overlay import overlay_mask
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.visualization.shape import draw_pose_axes
from paradex.robot.robot_wrapper import RobotWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(handeye_calib_path)

name = args.name
he_calib_path = os.path.join(handeye_calib_path, name)

intrinsic, extrinsic = load_camparam(os.path.join(he_calib_path, "0"))
cammtx = get_cammtx(intrinsic, extrinsic)
board_info = json.load(open(os.path.join(config_dir, "environment", "charuco_info.json"), "r"))

index_list = os.listdir(he_calib_path)

robot_cor = []
robot_cor_qpos = []

cam_cor = []
qpos = []

robot = RobotWrapper(get_robot_urdf_path("xarm"))

for idx in index_list:
    qpos_idx = np.load(os.path.join(he_calib_path, idx, "qpos.npy"))
    robot.compute_forward_kinematics(qpos_idx)
    rp = robot.get_link_pose(robot.get_link_index("link6"))
    robot_cor_qpos.append(rp)
    
    robot_cor.append(np.load(os.path.join(he_calib_path, idx, "robot.npy")))
    qpos.append(qpos_idx)    
    # if os.path.exists(os.path.join(he_calib_path, idx, "marker_3d.npy")):
    #     marker_3d = np.load(os.path.join(he_calib_path, idx, "marker_3d.npy"), allow_pickle=True).item()
    #     cam_cor.append(marker_3d)
    #     continue
    if not os.path.exists(os.path.join(he_calib_path, idx, "marker_3d.npy")):
        img_dir = os.path.join(he_calib_path, idx, "image")
        os.makedirs(os.path.join(he_calib_path, idx, "undistort"), exist_ok=True)
        
        img_dict = {}
        debug_img_dict = {}
        kypt_2d = {}
        
        for img_name in os.listdir(img_dir):
            serial_num = img_name.split(".")[0]
            
            img = cv2.imread(os.path.join(img_dir, img_name))
            undist_img = undistort_img(img, intrinsic[serial_num])
            cv2.imwrite(os.path.join(he_calib_path, idx, "undistort", img_name), undist_img)
            img_dict[serial_num] = undist_img.copy()
            debug_img_dict[serial_num] = undist_img.copy()
            
            detect_result = detect_charuco(undist_img, board_info)
            merged_detect_result = merge_charuco_detection(detect_result, board_info)
            
            # print(merged_detect_result['checkerIDs'].shape)
            if merged_detect_result['checkerIDs'].shape[0] == 0:
                continue
            
            ids = merged_detect_result['checkerIDs'][:,0]
            corners = merged_detect_result['checkerCorner'][:,0,:]
            
            draw_charuco(debug_img_dict[serial_num], corners, (255,0,0), 2, -1, merged_detect_result['checkerIDs'])
            for id, cor in zip(ids, corners):
                if id not in kypt_2d:
                    kypt_2d[id] = {"2d":[], "projection":[]}
                kypt_2d[id]["2d"].append(cor)
                kypt_2d[id]["projection"].append(cammtx[serial_num])

            
        marker_3d = {}    
        for id in kypt_2d.keys():
            proj_mat = np.array(kypt_2d[id]["projection"])
            cor_2d = np.array(kypt_2d[id]["2d"])
            
            pt3d = ransac_triangulation(cor_2d, proj_mat)
            if pt3d is None:
                continue
            
            
            marker_3d[id] = pt3d
        np.save(os.path.join(he_calib_path, idx, "marker_3d.npy"), marker_3d)
        cam_cor.append(marker_3d)

        os.makedirs(os.path.join(he_calib_path, idx, "debug"), exist_ok=True)
        pt_3d = np.array(list(marker_3d.values()))
        id_list = list(marker_3d.keys())
        
        for img_name in os.listdir(img_dir):
            serial_num = img_name.split(".")[0]
            cor_2d = np.array(project(cammtx[serial_num], pt_3d))
            draw_charuco(debug_img_dict[serial_num], cor_2d, (0,255,0), 2, -1, id_list)
            cv2.imwrite(os.path.join(he_calib_path, idx, "debug", f"{serial_num}.png"), debug_img_dict[serial_num])
    
    else:
        marker_3d = np.load(os.path.join(he_calib_path, idx, "marker_3d.npy"), allow_pickle=True).item()
        cam_cor.append(marker_3d)
        
A_list = []
B_list = []

for i in range(len(index_list)-1):
    B_list.append(robot_cor_qpos[i] @ np.linalg.inv(robot_cor_qpos[i+1]))
    
    marker1 = []
    marker2 = []
    for mid in cam_cor[i]:
        if mid in cam_cor[i+1]:
            marker1.append(cam_cor[i][mid])
            marker2.append(cam_cor[i+1][mid])
    
    marker1 = np.vstack(marker1)
    marker2 = np.vstack(marker2)
    A_list.append(rigid_transform_3D(marker2, marker1))
    # print(np.max(np.linalg.norm(((A_list[-1][:3,:3] @ marker2.T + A_list[-1][:3,3:]).T - marker1), axis=1)))

X = np.eye(4)
theta, b_x = solve(A_list, B_list)
X[0:3, 0:3] = theta
X[0:3, -1] = b_x.flatten()

print(X)
# for i in range(len(index_list)-1):
#     print(np.linalg.norm((A_list[i] @ X - X @ B_list[i])[:3,3]), "error", i)
    
X, loss = solve_axb_pytorch(A_list, B_list,X.copy(),learning_rate=0.001)
for i in range(len(index_list)-1):
    print(np.linalg.norm((A_list[i] @ X - X @ B_list[i])[:3,3]), "error", i)
    
np.save(os.path.join(he_calib_path, "0", "C2R.npy"), X)

marker_pos = {}

for idx in range(len(index_list)):
    for mid in cam_cor[idx]:
        if mid not in marker_pos:
            marker_pos[mid] = []
        
        marker_cam_pose = np.ones((4))
        marker_cam_pose[:3] = cam_cor[idx][mid]
        
        # marker_cam_pose = to_homo(cam_cor[idx][mid])
        marker_pos[mid].append((np.linalg.inv(robot_cor[idx]) @ np.linalg.inv(X) @ marker_cam_pose.T).T)
        
for mid in marker_pos:
    print(np.std(np.array(marker_pos[mid]), axis=0), "marker offset error")
    marker_pos[mid] = np.mean(marker_pos[mid], axis=0)
    
np.save(os.path.join(he_calib_path, "0", "marker_pos.npy"), marker_pos)

# extrinsic_list = []
# intrinsic_list = []

# serial_list = os.listdir(os.path.join(he_calib_path, "0", "image"))
# serial_list.sort()

# for serial_name in serial_list:
#     sn = serial_name.split(".")[0]
#     extmat = extrinsic[sn]
#     extrinsic_list.append(extmat)        
#     intrinsic_list.append(intrinsic[sn]['intrinsics_undistort'])
    
# qpos = np.array(qpos)
# rm = Robot_Module(get_robot_urdf_path("xarm"), state=qpos)

# renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')

# for fid in index_list:
#     img_dir = os.path.join(he_calib_path, fid, "debug")
#     img_dict = {img_name:cv2.imread(os.path.join(img_dir, img_name)) for img_name in serial_list}
    
#     robot_mesh_list = rm.get_mesh(int(fid))
#     robot_mesh = robot_mesh_list[0]
#     for i in range(1, len(robot_mesh_list)):
#         robot_mesh += robot_mesh_list[i]
#     robot_mesh.transform(X)
#     frame, mask = project_mesh_nvdiff(robot_mesh, renderer)
#     mask = mask.detach().cpu().numpy()[:,:,:,0]
#     mask = mask.astype(np.bool_)
#     for i, img_name in enumerate(serial_list):
#         if np.sum(mask[i]) > 0:
#             overlay_mask(img_dict[img_name], mask[i], 0.3, np.array((255, 0, 0)))
        
        
#     # os.makedirs(os.path.join(he_calib_path, fid, "overlay"), exist_ok=True)
#     cam_robot_pos = X @ robot_cor[int(fid)]
#     robot.compute_forward_kinematics(qpos[int(fid)])
#     rp = robot.get_link_pose(robot.get_link_index("link6"))
    
#     print(np.linalg.norm((rp-robot_cor[int(fid)])[:3,3]))
    
#     for img_name in serial_list:
#         draw_pose_axes(img_dict[img_name], cam_robot_pos, cammtx[img_name.split(".")[0]], text="xarm", color=(0,0,255))
    
#     cam_robot_pos = X @ rp
#     for img_name in serial_list:
#         draw_pose_axes(img_dict[img_name], cam_robot_pos, cammtx[img_name.split(".")[0]], text="qpos", color=(0,255,0))
    
#     os.makedirs(os.path.join(he_calib_path, fid, "overlay"), exist_ok=True)
#     for img_name in serial_list:
#         cv2.imwrite(os.path.join(he_calib_path, fid, "overlay", f"{img_name}"), img_dict[img_name])    