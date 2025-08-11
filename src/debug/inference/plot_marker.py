import os
import cv2
import argparse
import numpy as np

from paradex.image.aruco import detect_aruco, triangulate_marker, draw_aruco
from paradex.image.undistort import undistort_img
from paradex.geometry.Tsai_Lenz import solve
from paradex.geometry.conversion import project, to_homo
from paradex.utils.file_io import shared_dir, load_camparam
from paradex.image.projection import get_cammtx
from paradex.geometry.math import rigid_transform_3D

marker_id = [261, 262, 263, 264, 265, 266]

root_path = os.path.join(shared_dir, "debug_", "inference")
import os
import numpy as np
import argparse
import trimesh
from multiprocessing import Pool

from paradex.utils.file_io import rsc_path, shared_dir, load_camparam, get_robot_urdf_path
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.image.merge import merge_image
from paradex.image.overlay import overlay_mask
from paradex.video.process_video import process_video_list

from paradex.video.convert_codec import change_to_h264
from paradex.utils.upload_file import copy_file

from paradex.visualization_.renderer import BatchRenderer
from paradex.visualization_.robot_module import Robot_Module
from paradex.robot.mimic_joint import parse_inspire
import copy

def load_info(video_dir):
    root_path = os.path.dirname(video_dir) # obj_name/grasp_type/index/video
    serial_list = [vid_name.split('.')[0] for vid_name in os.listdir(os.path.join(root_path, "videos"))]
    
    intrinsic, extrinsic = load_camparam(root_path)
    cammtx = get_cammtx(intrinsic, extrinsic)
    
    cor_3d = np.load(os.path.join(root_path, "cor_3d.npy"),allow_pickle=True).item()
    cor_2d = {serial:np.load(os.path.join(root_path, "marker2D", f"{serial}.npy"), allow_pickle=True).item() for serial in serial_list}
    
    c2r = np.load(os.path.join(root_path, "C2R.npy"))
    
    marker_offset = np.load(os.path.join(root_path, "marker_pos.npy"),allow_pickle=True).item()
    robot_pose = np.load(os.path.join(root_path, "xarm", "position.npy"))
    
    extrinsic_list = []
    intrinsic_list = []
    cammtx_list = []
    
    for serial_name in serial_list:
        extmat = extrinsic[serial_name]
        extrinsic_list.append(extmat @ c2r)
        
        intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
        cammtx_list.append(intrinsic_list[-1] @ extrinsic_list[-1])
    
    return cor_3d, cor_2d, serial_list, cammtx, c2r, robot_pose, marker_offset

def process_frame(img_dict, video_path, fid, data):
    (cor_3d, cor_2d, serial_list, cammtx, c2r, robot_pose, marker_offset) = data
    robot_3d = {}
    for id_, pose in marker_offset.items():
        pose = c2r @ robot_pose[fid+1] @ pose.T
        pose = pose[:3, :] / pose[3:, :]
        
        robot_3d[id_] = pose.T
    
    for i, serial_num in enumerate(serial_list):
        if fid+1 not in cor_2d[serial_num]:
            continue
        
        cor2d_img = list(cor_2d[serial_num][fid+1].values())
        img_dict[serial_num] = draw_aruco(img_dict[serial_num], cor2d_img, color=(0, 0, 255))
        
        if fid+1 in cor_3d:
            for id, cor in cor_3d[fid+1].items():
                if id not in cor_2d[serial_num][fid+1]:
                    continue
                img_dict[serial_num] = project_point(cor, cammtx[serial_num], img_dict[serial_num])
        
        for id, cor in robot_3d.items():
            if id not in cor_2d[serial_num][fid+1]:
                continue
            img_dict[serial_num] = project_point(cor, cammtx[serial_num], img_dict[serial_num], color=(0, 255, 0))


    # for id, cor in cor_3d[fid+1].items():
    #     if cor is None:
    #         continue
    #     cor_h = np.concatenate([cor, np.ones((cor.shape[0], 1))], axis=1)
    #     cor = (np.linalg.inv(c2r) @ cor_h.T).T[:,:3]
    #     for i, serial_num in enumerate(serial_list):
    #         img_dict[serial_num] = project_point(cor, cammtx_list[i], img_dict[serial_num])
    
    frame = merge_image(img_dict)
    return frame

def process_frame_list(img_dict, video_dir, fid):
    frame = merge_image(img_dict)
    return frame

for index in os.listdir(root_path):
    index_dir = os.path.join(root_path, index)
    intrinsic, extrinsic = load_camparam(index_dir)
    cammtx = get_cammtx(intrinsic, extrinsic)

    
    robot_cor = []
    cam_cor = []

    
    index_dir = os.path.join(os.path.join(root_path, str(index)))
    os.makedirs(os.path.join(index_dir, "overlay"), exist_ok=True)
    os.makedirs(os.path.join(index_dir, "videos"), exist_ok=True)
    if os.path.exists(os.path.join(index_dir, "merge_overlay.mp4")):
        continue
    
    process_video_list(os.path.join(index_dir, "videos"), 
            os.path.join(index_dir, "overlay"), 
            load_info(os.path.join(index_dir, "videos")), 
            process_frame)
    change_to_h264(os.path.join(index_dir, "overlay_tmp.avi"), os.path.join(index_dir, "merge_overlay.mp4"))