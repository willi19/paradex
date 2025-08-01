import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import shutil
import tqdm

from paradex.io.contact.process import process_contact
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path, shared_dir, load_camparam
from paradex.video.raw_video import fill_framedrop, get_synced_data
from paradex.geometry.coordinate import DEVICE2WRIST
from paradex.image.projection import get_cammtx
from paradex.geometry.triangulate import ransac_triangulation
from paradex.geometry.math import rigid_transform_3D

# td = 0.09 # latency difference between camera and sensor
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_name', nargs="+", type=str, default=None)
    parser.add_argument('--grasp_type', nargs="+", type=str, default=None)
    args = parser.parse_args()

    process_list = []
    
    if args.obj_name == None:
        name_list = os.listdir(os.path.join(shared_dir, 'capture_', "lookup"))
        name_list.sort()

    else:
        name_list = args.obj_name
        
    for name in name_list:
        grasp_list = os.listdir(os.path.join(shared_dir, "capture_", "lookup", name))
        if args.grasp_type is not None:
            for grasp_name in args.grasp_type:
                if grasp_name in grasp_list:
                    process_list.append((name, grasp_name))
        if args.grasp_type is None:
            for grasp_name in grasp_list:
                process_list.append((name, grasp_name))
    
    for name, grasp_type in process_list:
        root_dir = os.path.join(shared_dir, "capture_", "lookup", name, grasp_type)
        index_list = os.listdir(root_dir)
        
        marker_offset = np.load(os.path.join(shared_dir, "marker_offset", name, "0", "marker_offset.npy"), allow_pickle=True).item()
        marker_id = list(marker_offset.keys())

        for index in index_list:
            index_dir = os.path.join(os.path.join(root_dir, str(index)))
            marker_list = os.listdir(os.path.join(index_dir, "marker2D"))
            
            intrinsic, extrinsic = load_camparam(index_dir)
            cammat = get_cammtx(intrinsic, extrinsic)
            
            id_cor = {}
            
            for file_name in tqdm.tqdm(marker_list):
                marker_hist = np.load(os.path.join(index_dir, "marker2D", file_name),allow_pickle=True).item()
                serial_num = file_name.split(".")[0]
                for t, marker in marker_hist.items():
                    if t not in id_cor:
                        id_cor[t] = {}
                    
                    for id, cor in marker.items():
                        if id not in marker_id:
                            continue
                        
                        if id not in id_cor[t]:
                            id_cor[t][id] = {"2d":[], "cammtx":[]}
                        
                        id_cor[t][id]["2d"].append(cor)
                        id_cor[t][id]["cammtx"].append(cammat[serial_num])
                    
            obj_T = []
            cor_3d_hist = {}
            for t, marker_dict in id_cor.items():
                cor_3d = {id:ransac_triangulation(np.array(marker_dict[id]["2d"]), np.array(marker_dict[id]["cammtx"])) for id in marker_dict.keys()}
                cor_3d_hist[t] = cor_3d
                A = []
                B = []
                for id in cor_3d.keys():
                    if cor_3d[id] is None:
                        continue
                    A.append(marker_offset[id])
                    B.append(cor_3d[id])
                if len(A) == 0:
                    obj_T.append(np.zeros((4,4)))
                    continue
                
                A = np.concatenate(A)
                B = np.concatenate(B)
                obj_T.append(rigid_transform_3D(A, B))
            obj_T = np.array(obj_T)
            np.save(os.path.join(index_dir, "obj_T.npy"), np.array(obj_T))
            np.save(os.path.join(index_dir, "cor_3d"), cor_3d_hist)