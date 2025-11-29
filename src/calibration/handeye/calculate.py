import os
import cv2
import argparse
import numpy as np
import json
from copy import deepcopy

# from paradex.image.aruco import detect_charuco, merge_charuco_detection
# from paradex.image.undistort import undistort_img
# from paradex.geometry.Tsai_Lenz import solve, solve_axb_pytorch
# from paradex.geometry.conversion import project, to_homo
from paradex.utils.file_io import find_latest_directory
# from paradex.image.projection import get_cammtx
# from paradex.geometry.math import rigid_transform_3D
# from paradex.geometry.triangulate import ransac_triangulation
# from paradex.image.aruco import draw_charuco
# from paradex.visualization_.renderer import BatchRenderer
# from paradex.visualization_.robot_module import Robot_Module
# from paradex.image.overlay import overlay_mask
# from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
# from paradex.visualization.shape import draw_pose_axes

from paradex.calibration.utils import handeye_calib_path, load_camparam
from paradex.calibration.Tsai_Lenz import solve_ax_xb
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import merge_charuco_detection, find_common_indices
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.visualization.robot import RobotModule

def detect_charuco(name):
    img_dict = None
    root_dir = os.path.join(handeye_calib_path, name)
    index_list = sorted(os.listdir(root_dir))
    
    for index in index_list:
        if os.path.exists(os.path.join(root_dir, index, "charuco_3d_ids.npy")) and \
           os.path.exists(os.path.join(root_dir, index, "charuco_3d_corners.npy")):
            continue

        if os.path.exists(os.path.join(root_dir, index, "undistort", "images")) and \
            len(os.listdir(os.path.join(root_dir, index, "undistort", "images"))) == \
            len(os.listdir(os.path.join(root_dir, index, "image"))):
            continue
        
        os.makedirs(os.path.join(root_dir, index, "undistort", "images"), exist_ok=True)
        if img_dict is None:
            img_dict = ImageDict.from_path(os.path.join(root_dir, index))
        else:
            img_dict.update_path(os.path.join(root_dir, index))
            
        undistort_img_dict = img_dict.undistort(save_path=os.path.join(root_dir, index, "undistort"))
        
        charuco_3d = undistort_img_dict.triangulate_charuco()
        charuco_3d = merge_charuco_detection(charuco_3d)
        
        np.save(os.path.join(root_dir, index, "charuco_3d_ids.npy"), charuco_3d['checkerIDs'])
        np.save(os.path.join(root_dir, index, "charuco_3d_corners.npy"), charuco_3d['checkerCorner'])

def compute_fk(name, arm):
    root_dir = os.path.join(handeye_calib_path, name)
    index_list = sorted(os.listdir(root_dir))

    robot_wrapper = RobotWrapper(get_robot_urdf_path(arm_name=arm))

    for index in index_list:
        if os.path.exists(os.path.join(root_dir, index, "eef_fk.npy")):
            continue
        
        qpos = np.load(os.path.join(root_dir, index, "qpos.npy"))
        eef = robot_wrapper.compute_forward_kinematics(qpos, link_list=["link6"])['link6']
        np.save(os.path.join(root_dir, index, "eef_fk.npy"), eef)

def compute_motion(name):
    motion_wrt_cam = []
    motion_wrt_robot = []
    
    root_dir = os.path.join(handeye_calib_path, name)
    index_list = os.listdir(root_dir)
    index_list.sort()
    
    eef_list = [np.load(os.path.join(root_dir, index, "eef_fk.npy")) for index in index_list]
    
    charuco_id_list = [np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy")) for index in index_list]
    charuco_cor_list = [np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy")) for index in index_list]
    
    for i in range(1, len(index_list)):
        eef = eef_list[i]
        eef_prev = eef_list[i-1]

        motion_wrt_robot.append(eef @ np.linalg.inv(eef_prev))
        
        ids = charuco_id_list[i]
        ids_prev = charuco_id_list[i-1]
        
        common_idx, common_idx_prev = find_common_indices(ids, ids_prev)

        cam_cor = charuco_cor_list[i][common_idx]
        cam_cor_prev = charuco_cor_list[i-1][common_idx_prev]

        motion_wrt_cam.append(SOLVE_XA_B(cam_cor_prev, cam_cor))
    
    return motion_wrt_cam, motion_wrt_robot

def debug(name, arm):
    root_dir = os.path.join(handeye_calib_path, name)

    index_list = os.listdir(root_dir)
    index_list.sort()

    C2R = np.load(os.path.join(root_dir, index_list[0], "C2R.npy")) # cam_wrt_robot
    marker_pos = {}
    
    rm = RobotModule(get_robot_urdf_path(arm_name=arm))
    
    for index in index_list:
        eef = np.load(os.path.join(root_dir, index, "eef_fk.npy"))
        eef_from_robot = np.load(os.path.join(root_dir, index, "eef.npy"))
        # FK error
        eef_error = np.linalg.inv(eef) @ eef_from_robot
        trans_error = np.linalg.norm(eef_error[:3, 3]) * 1000
        
        # Rotation error (degrees)
        R_error = eef_error[:3, :3]
        angle_error = np.arccos((np.trace(R_error) - 1) / 2) * 180 / np.pi
        
        print(f"{index}: trans={trans_error:.2f}mm, rot={angle_error:.2f}deg")
        
        charuco_3d_cor = np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        charuco_id_cor = np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy"))
        
        for mid, cor in zip(charuco_id_cor, charuco_3d_cor):
            if mid not in marker_pos:
                marker_pos[mid] = []
            marker_pos[mid].append((np.linalg.inv(eef) @ np.linalg.inv(C2R) @ cor.T).T)
    
    marker_wrt_eef = []
    for mid in marker_pos:
        marker_wrt_eef = np.array(marker_pos[mid])
        mean_pos = np.mean(marker_wrt_eef, axis=0)
        std_pos = np.std(marker_wrt_eef, axis=0)
        print(f"Marker {mid}: std={std_pos}, mean={mean_pos}")

        marker_wrt_eef.append(mean_pos)
    marker_wrt_eef = np.array(marker_wrt_eef)
    
    img_dict = None
    for index in index_list:
        if os.path.exists(os.path.join(root_dir, index, "debug", 'images')) and \
           len(os.listdir(os.path.join(root_dir, index, "debug", 'images'))) == \
           len(os.listdir(os.path.join(root_dir, index, "image"))):
            continue
        
        if img_dict is None:
            img_dict = ImageDict.from_path(os.path.join(root_dir, index, "undistort"))
        else:
            img_dict.update_path(os.path.join(root_dir, index, "undistort"))
        
        qpos = np.load(os.path.join(root_dir, index, "qpos.npy"))
        eef = np.load(os.path.join(root_dir, index, "eef_fk.npy"))
        
        rm.update_cfg(qpos)
        robot_mesh = rm.get_robot_mesh()
        robot_mesh.transform(np.linalg.inv(C2R))
        
        overlay_img_dict = img_dict.project_mesh(robot_mesh)
        
        marker_wrt_eef_h = np.ones((marker_wrt_eef.shape[0], 4))
        marker_wrt_eef_h[:,:3] = marker_wrt_eef 
        
        marker_wrt_cam = C2R @ eef @ marker_wrt_eef_h.T
        proj_marker = overlay_img_dict.project_pointcloud(marker_wrt_cam, color=(0,255,0), point_size=5)
        overlay_img_dict.draw_keypoint(proj_marker, (255, 0, 0))
        
        marker_3d  = np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        proj_marker_3d = overlay_img_dict.project_pointcloud(marker_3d.T, color=(0,0,255), point_size=5)
        overlay_img_dict.draw_keypoint(proj_marker_3d, (0, 0, 255))
        
        overlay_img_dict.save(os.path.join(root_dir, index, "debug"))
                

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")
parser.add_argument("--arm", type=str, default="xarm", help="Name of the robot arm.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(handeye_calib_path)

name = args.name
root_path = os.path.join(handeye_calib_path, name)
index_list = os.listdir(root_path)
intrinsic, extrinsic = load_camparam(os.path.join(root_path, "0"))

detect_charuco(name)
compute_fk(name, args.arm)
motion_wrt_cam, motion_wrt_robot = compute_motion(name)

robot_wrt_cam = solve_ax_xb(motion_wrt_cam, motion_wrt_robot) 
cam_wrt_robot = np.linalg.inv(robot_wrt_cam)

for i in range(len(index_list)-1):
    diff = (motion_wrt_cam[i] @ robot_wrt_cam) @ np.linalg.inv(robot_wrt_cam @ motion_wrt_robot[i])
    trans_error = np.linalg.norm(diff[:3, 3]) * 1000

    # Rotation error (degrees)
    R_error = diff[:3, :3]
    angle_error = np.arccos((np.trace(R_error) - 1) / 2) * 180 / np.pi
    
    print(np.linalg.norm((motion_wrt_cam[i] @ robot_wrt_cam - robot_wrt_cam @ motion_wrt_robot[i])[:3,3]), "error", i)
    
np.save(os.path.join(root_path, index_list[0], "C2R.npy"), robot_wrt_cam)

debug(name, args.arm)