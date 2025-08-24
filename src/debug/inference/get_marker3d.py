import os
import cv2
import argparse
import numpy as np
import tqdm

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

from paradex.robot.mimic_joint import parse_inspire
from paradex.geometry.triangulate import ransac_triangulation

marker_id = [261, 262, 263, 264, 265, 266]

for index in os.listdir(root_path):
    index_dir = os.path.join(root_path, index)
    intrinsic, extrinsic = load_camparam(index_dir)
    cammtx = get_cammtx(intrinsic, extrinsic)
    c2r = np.load(os.path.join(index_dir, "C2R.npy"))
    offset = np.load(os.path.join(index_dir, "marker_pos.npy"), allow_pickle=True).item()   
    cor_3d_dict = {}
    
    for np_file in os.listdir(os.path.join(index_dir, "marker2D")):
        
        serial_num = np_file.split(".")[0]
        if serial_num not in cammtx:
            continue
        
        marker_2d = np.load(os.path.join(index_dir, "marker2D", np_file), allow_pickle=True).item()
        for fid, marker in marker_2d.items():
            if marker is None:
                continue
            
            if fid not in cor_3d_dict:
                cor_3d_dict[fid] = {}
                
                
                
            for mid, cor in marker.items():
                if mid not in marker_id:
                    continue
                if mid not in cor_3d_dict[fid]:
                    cor_3d_dict[fid][mid] = {"2d":[], "cammtx":[]}
                    
                cor_3d_dict[fid][mid]["2d"].append(cor.squeeze())
                cor_3d_dict[fid][mid]["cammtx"].append(cammtx[serial_num])
                
    cor_3d = {}
    for fid, data in tqdm.tqdm(cor_3d_dict.items()):
        cor_3d[fid] = {}
        for mid, cor in data.items():
            if len(cor["2d"]) < 3:
                continue
            
            pts_2d = np.array(cor["2d"])
            cammtx = np.array(cor["cammtx"])
            
            
            pts_3d = ransac_triangulation(pts_2d, cammtx)
            if pts_3d is None:
                continue
            cor_3d[fid][mid] = pts_3d
        
    np.save(os.path.join(index_dir, "cor_3d.npy"), cor_3d)