import argparse
import os
import json
import yaml
from itertools import chain
from multiprocessing import Pool
from glob import glob
import pycolmap
import cv2
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import contextlib

from paradex.utils.file_io import find_latest_directory
from paradex.utils.path import shared_dir
from paradex.calibration.colmap import *
from paradex.image.aruco import draw_charuco
from paradex.image.undistort import undistort_points
from paradex.image.projection import get_cammtx
from paradex.geometry.triangulate import ransac_triangulation


config_dir = "config"
cam_param_dir = os.path.join(shared_dir, "cam_param")

def process_match(args):
    """ Function to process a single match pair """
    (cam_key1, cam_key2), matches, corners1, corners2, cam_id1, cam_id2 = args
    idx1 = matches[:, 0]
    idx2 = matches[:, 1]


    twoviewgeom = get_two_view_geometries(
        cam_key1, cam_key2, corners1[idx1], corners2[idx2], matches, (cam_id1, cam_id2)
    )
    return twoviewgeom

def parallel_processing(db, serial_index, tot_kypt_matches, tot_kypt_dict, cam_keys):
    """ Parallelize the processing of keypoint matches """
    args = [((cam_keys[serial_index[serial_1]], cam_keys[serial_index[serial_2]]), np.vstack(matches, dtype=np.int32), tot_kypt_dict[serial_1], tot_kypt_dict[serial_2], serial_index[serial_1], serial_index[serial_2]) for (serial_1, serial_2), matches in tot_kypt_matches.items()]
    num_processes = mp.cpu_count()  # Use all available cores
    with mp.Pool(processes=num_processes) as pool:
        results = list(pool.map(process_match, args))
    # Add two-view geometry to database
    for twoviewgeom in results:
        if twoviewgeom is not None:
            db.add_two_view_geometry(*twoviewgeom)
    
def load_keypoint(root_dir):
    index_list = os.listdir(root_dir)
    index_list.sort()

    index_list = [int(index) for index in index_list]

    if len(index_list) == 0:
        print("No valid directories found.")
        return 
    
    keypoint_dict = {index: {} for index in index_list}
    for index in index_list:
        frame_dir = os.path.join(root_dir, str(index))
        if not os.path.exists(frame_dir):
            continue
        kypt_file = os.listdir(frame_dir)
        for f in kypt_file:
            if "cor" not in f:
                continue
            serial_num = f.split("_")[0]
            
            corners = np.load(os.path.join(frame_dir, f))
            if corners.shape[0] == 0:
                continue

            keypoint_dict[index][serial_num] = {}
            keypoint_dict[index][serial_num]["corners"] = corners[:,0, :]
            keypoint_dict[index][serial_num]["ids"] = np.load(os.path.join(frame_dir, f"{serial_num}_id.npy"))
    return keypoint_dict

def add_camera(db, intrinsic, serial_list):
    for serial_num in serial_list: # add 50 cameras
        intrinsic = intrinsics_dict[serial_num]
        width = intrinsic["width"]
        height = intrinsic["height"]
        fx = intrinsic["original_intrinsics"][0][0]
        fy = intrinsic["original_intrinsics"][1][1]
        k1 = intrinsic["dist_params"][0][0]
        k2 = intrinsic["dist_params"][0][1]
        p1 = intrinsic["dist_params"][0][2]
        p2 = intrinsic["dist_params"][0][3]
        cx = intrinsic["original_intrinsics"][0][2]
        cy = intrinsic["original_intrinsics"][1][2]

        camera_id = db.add_camera(4, width, height, np.array([fx,fy, cx, cy, k1, k2, p1, p2]), 0)
        image_id = db.add_image(f"{serial_num}.jpg", camera_id)
    return db

def get_total_keypoint(keypoint_dict, serial_list):
    tot_kypt_dict = {serial_num:[] for serial_num in serial_list}
    tot_kypt_matches = {}
    kypt_offset = {serial_num:0 for serial_num in serial_list}
    
    for index, kypt_data in keypoint_dict.items():
        kypt_serial_list = list(kypt_data.keys()
                                )
        for i in range(len(kypt_serial_list)):
            for j in range(i+1, len(kypt_serial_list)):
                serial_1 = kypt_serial_list[i]
                serial_2 = kypt_serial_list[j]
                
                common_ids, idx1, idx2 = np.intersect1d(kypt_data[serial_1]["ids"], kypt_data[serial_2]["ids"], return_indices=True)
                idx1 += kypt_offset[serial_1]
                idx2 += kypt_offset[serial_2]
                
                if len(common_ids) > 0:
                    matches = np.column_stack((idx1, idx2))
                    if (serial_1, serial_2) not in tot_kypt_matches.keys():
                        tot_kypt_matches[(serial_1, serial_2)] = []
                    tot_kypt_matches[(serial_1, serial_2)].append(matches)

        for serial_num in kypt_serial_list:
            tot_kypt_dict[serial_num].append(kypt_data[serial_num]["corners"])
            kypt_offset[serial_num] += len(kypt_data[serial_num]["ids"])
    
    return tot_kypt_dict, tot_kypt_matches

def generate_db(database_path, intrinsics_dict, serial_list, keypoint_dict):
    if os.path.exists(database_path):
        os.remove(database_path)
        
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    
    db = add_camera(db, intrinsics_dict, serial_list)
    serial_index = {serial_num:i+1 for i, serial_num in enumerate(serial_list)}    

    cam_keys = db.get_camera()
    
    tot_kypt_dict, tot_kypt_matches = get_total_keypoint(keypoint_dict, serial_list)
    
    for serial_num, kypt_list in tot_kypt_dict.items():
        kypt_list = np.vstack(kypt_list)
        tot_kypt_dict[serial_num] = kypt_list
        camera_id = serial_index[serial_num]
        db.add_keypoints(camera_id, kypt_list)

    for (serial_1, serial_2), matches in tot_kypt_matches.items():        
        image_id_1 = serial_index[serial_1]
        image_id_2 = serial_index[serial_2]
        matches = np.vstack(matches)
        db.add_matches(image_id_1, image_id_2, matches)
    
    parallel_processing(db, serial_index, tot_kypt_matches, tot_kypt_dict, cam_keys)

    db.commit()
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--name", type=str, help="Name of the directory to detect keypoint.")
    
    args = parser.parse_args()
    
    if args.name is None:
        name = find_latest_directory(extrinsic_dir)
    else:
        name = args.name

    root_dir = os.path.join(extrinsic_dir, name)
    index_list = os.listdir(root_dir)
    keypoint_dict_distort = load_keypoint(root_dir)
    intrinsics_dict = load_intrinsic()
    
    serial_list = []
    for kypt_dict in keypoint_dict_distort.values():
        for serial_num in kypt_dict.keys():
            if serial_num not in serial_list:
                serial_list.append(serial_num)
    num_cameras = len(serial_list)
    

    out_pose_dir = os.path.join(root_dir, "0", "colmap")
    os.makedirs(out_pose_dir, exist_ok=True)

    database_path = out_pose_dir + "/database.db"
    
    generate_db(database_path, intrinsics_dict, serial_list, keypoint_dict_distort)
    
    with open(f"{config_dir}/colmap_options.yaml", "r") as f:
        options = yaml.safe_load(f)
        
    Options = pycolmap.IncrementalPipelineOptions(options['MapperOptions'])
    
    maps = pycolmap.incremental_mapping(database_path, ".", out_pose_dir, options=Options)
    best_idx = max(maps, key=lambda i: maps[i].num_images())
    maps[best_idx].write(out_pose_dir)
    maps[best_idx].write_text(out_pose_dir)

    print("name : ", name)
    intrinsics, extrinsics = load_colmap_camparam(root_dir)
    cammtx = get_cammtx(intrinsics, extrinsics)
    
    for serial in serial_list:
        if str(serial) not in list(intrinsics.keys()):
            print(serial, "not found")
    print("============")

    length = []
    proj_err = {serial_num:[] for serial_num in serial_list}

    keypoint_dict = {}
    for index, kypt_2d_dict in keypoint_dict_distort.items():
        keypoint_dict[index] = {}
        for serial_num, kypt_2d in kypt_2d_dict.items():
            kypt_2d_undist = {'ids':[], 'corners':[]}
            for id, cor in zip(kypt_2d['ids'], kypt_2d['corners']):
                cor_undist = undistort_points(cor, intrinsics[serial_num])
                
                kypt_2d_undist['ids'].append(id)
                kypt_2d_undist['corners'].append(cor_undist)
            keypoint_dict[index][serial_num] = kypt_2d_undist
    
    for index, kypt_2d_dict in keypoint_dict.items():
        kypt_id_dict = {}
        kypt_3d = {}
        
        for serial_num, kypt_2d in kypt_2d_dict.items():
            for id, cor in zip(kypt_2d['ids'], kypt_2d['corners']):
                if id[0] not in kypt_id_dict.keys():
                    kypt_id_dict[id[0]] = {"2d": [], "projection":[]}
                
                kypt_id_dict[id[0]]["2d"].append(cor)
                kypt_id_dict[id[0]]["projection"].append(cammtx[serial_num].copy())

        for id in kypt_id_dict.keys():
            proj_mat = np.array(kypt_id_dict[id]["projection"])
            kypt_2d = np.array(kypt_id_dict[id]["2d"])
            
            pt3d = ransac_triangulation(kypt_2d, proj_mat)
            if pt3d is None:
                continue
            
            
            kypt_3d[id] = pt3d

        idx_list = list(kypt_3d.keys())
        idx_list.sort()
        for i in idx_list:
            if i-1 in list(kypt_3d.keys()) and i % 10 != 0:
                length.append(np.linalg.norm(kypt_3d[i] - kypt_3d[i-1]))
            if i+10 in list(kypt_3d.keys()):
                length.append(np.linalg.norm(kypt_3d[i] - kypt_3d[i+10]))
        
        for serial_num, kypt_2d in kypt_2d_dict.items():
            for id, cor in zip(kypt_2d['ids'], kypt_2d['corners']):
                if id[0] not in kypt_3d.keys():
                    continue
                pt3d = kypt_3d[id[0]][0]
                pt3d_h = np.hstack((pt3d, np.ones((1))))
                proj = cammtx[serial_num] @ pt3d_h
                proj = proj[:2] / proj[2]

                err = np.linalg.norm(proj - cor)
                if err > 10:
                    print(f"index {index}, serial {serial_num}, id {id[0]}, err {err}")
                proj_err[serial_num].append(err)
                    
    print(np.std(length))
    print(np.mean(length))
    
    for serial_num, err in proj_err.items():
        print(serial_num, np.mean(err), np.max(err))

    with open(os.path.join(root_dir, '0', 'colmap', 'result.txt'), 'w') as f:
        for serial_num, proj in proj_err.items():
            f.write(f"{serial_num} : mean {np.mean(proj)}, max{np.max(proj)} \n")

    new_extrinsics = {}
    for serial_num, extrinsic in extrinsics.items():
        new_extrinsic = np.array(extrinsic)
        new_extrinsic[:3, 3] *= (0.025 / np.mean(length))
        new_extrinsics[serial_num] = new_extrinsic.tolist()

    os.makedirs(os.path.join(cam_param_dir, name), exist_ok=True)
        
    with open(os.path.join(cam_param_dir, name, "extrinsics.json"), "w") as f:
        json.dump(new_extrinsics, f, indent=4)

    for serial_num, intrinsic in intrinsics.items():
        new_intrinsic = np.array(intrinsic['intrinsics_undistort'])
        intrinsics[serial_num]['intrinsics_undistort'] = new_intrinsic.tolist()
        intrinsics[serial_num]['original_intrinsics'] = np.array(intrinsic['original_intrinsics']).tolist()
        intrinsics[serial_num]['dist_params'] = np.array(intrinsic['dist_params']).tolist()

    with open(os.path.join(cam_param_dir, name, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=4)
    