import argparse
import os
import pycolmap
import multiprocessing as mp
import tqdm
import numpy as np
from datetime import datetime
import shutil
import json

from paradex.utils.file_io import find_latest_directory
from paradex.calibration.colmap import *
from paradex.image.aruco import detect_charuco, merge_charuco_detection, get_adjecent_ids, find_common_indices
from paradex.calibration.utils import cam_param_dir, extrinsic_dir, load_current_intrinsic
from paradex.image.image_dict import ImageDict

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
    index_list = sorted(os.listdir(root_dir))
    if len(index_list) == 0:
        print("No valid directories found.")
        return 
    
    keypoint_dict = {index: {} for index in index_list}
    for index in index_list:
        
        frame_dir = os.path.join(root_dir, index, 'markers_2d')
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
            keypoint_dict[index][serial_num]["corners"] = corners
            keypoint_dict[index][serial_num]["ids"] = np.load(os.path.join(frame_dir, f"{serial_num}_id.npy"))
    return keypoint_dict

def add_camera(db, intrinsics_dict, serial_list):
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
        kypt_serial_list = sorted(list(kypt_data.keys()))
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

def run_calibration(name):
    root_dir = os.path.join(extrinsic_dir, name)
    index_list = sorted(os.listdir(root_dir))
    
    keypoint_dict_distort = load_keypoint(root_dir)
    intrinsics_dict = load_current_intrinsic()
    
    serial_list = []
    for kypt_dict in keypoint_dict_distort.values():
        for serial_num in kypt_dict.keys():
            if serial_num not in serial_list:
                serial_list.append(serial_num)
    num_cameras = len(serial_list)
    

    out_pose_dir = os.path.join(root_dir, index_list[0], "colmap")
    os.makedirs(out_pose_dir, exist_ok=True)

    database_path = out_pose_dir + "/database.db"
    
    generate_db(database_path, intrinsics_dict, serial_list, keypoint_dict_distort)
    
    options = {
        'ba_refine_principal_point': True,
        'ba_refine_extra_params': True,
        'ba_refine_focal_length': True,
        'max_extra_param': 4,
    }
    Options = pycolmap.IncrementalPipelineOptions(options)
    
    maps = pycolmap.incremental_mapping(database_path, ".", out_pose_dir, options=Options)

    best_idx = max(maps, key=lambda i: maps[i].num_images())
    maps[best_idx].write(out_pose_dir)
    maps[best_idx].write_text(out_pose_dir)
    return 

def undistort(name):
    index_list = sorted(os.listdir(os.path.join(extrinsic_dir, name)))
    out_pose_dir = os.path.join(extrinsic_dir, name, index_list[0], "colmap")
    intrinsics, extrinsics = load_colmap_camparam(out_pose_dir)
    
    img_dict = None
    for index in tqdm.tqdm(index_list, desc="Undistorting images"):
        # if os.path.exists(os.path.join(extrinsic_dir, name, index, "undistort", "images")) \
        #     and len(os.listdir(os.path.join(extrinsic_dir, name, index, "undistort", "images"))) == \
        #         len(os.listdir(os.path.join(extrinsic_dir, name, index, "images"))):
            
        #     continue
        
        if img_dict is None:
            img_dict = ImageDict.from_path(os.path.join(extrinsic_dir, name, index))
            img_dict.set_camparam(intrinsics, extrinsics)
        else:
            img_dict.update_path(os.path.join(extrinsic_dir, name, index), reload_images=True)
        # img_dict.set_camparam(intrinsics, extrinsics)
        img_dict_undistort = img_dict.undistort(save_path=os.path.join(extrinsic_dir, name, index, "undistort"))
    
    return

def save_kypt_3d(name):
    index_list = sorted(os.listdir(os.path.join(extrinsic_dir, name)))
    out_pose_dir = os.path.join(extrinsic_dir, name, index_list[0], "colmap")
    intrinsics, extrinsics = load_colmap_camparam(out_pose_dir)

    for index in tqdm.tqdm(index_list, desc="Saving 3D keypoints"):
        # if os.path.exists(os.path.join(extrinsic_dir, name, index, "kypt_3d_id.npy")):
        #     continue
        
        img_dict_undistort = ImageDict.from_path(os.path.join(extrinsic_dir, name, index, "undistort"))
        img_dict_undistort.set_camparam(intrinsics, extrinsics)
        charuco_3d = img_dict_undistort.triangulate_charuco()
        merged_charuco_3d = merge_charuco_detection(charuco_3d)
        np.save(os.path.join(extrinsic_dir, name, index, "kypt_3d_id.npy"), merged_charuco_3d["checkerIDs"])
        np.save(os.path.join(extrinsic_dir, name, index, "kypt_3d_cor.npy"), merged_charuco_3d["checkerCorner"])
    
def get_length(name):
    index_list = sorted(os.listdir(os.path.join(extrinsic_dir, name)))
    length_list = []
    
    adj_ids = get_adjecent_ids()

    for index in tqdm.tqdm(index_list, desc="Getting 3D keypoints length"):
        kypt_3d_ids = np.load(os.path.join(extrinsic_dir, name, index, "kypt_3d_id.npy"))
        kypt_3d_cor = np.load(os.path.join(extrinsic_dir, name, index, "kypt_3d_cor.npy"))
        kypt_3d = {}
        
        for id, cor in zip(kypt_3d_ids, kypt_3d_cor):
            kypt_3d[id] = cor
        for mid in kypt_3d.keys():
            for adj_id in adj_ids[mid]:
                if adj_id in kypt_3d:
                    length_list.append(np.linalg.norm(kypt_3d[mid] - kypt_3d[adj_id]))
    
    print("Length mean:", np.mean(length_list))
    print("Length std:", np.std(length_list))
    print("Max length:", np.max(length_list))
    print("Min length:", np.min(length_list))
    return np.mean(length_list)
     
def debug(name, refine=True):
    index_list = sorted(os.listdir(os.path.join(extrinsic_dir, name)))
    out_pose_dir = os.path.join(extrinsic_dir, name, index_list[0], "colmap")
    err_dict = {}
    intrinsics, extrinsics = load_colmap_camparam(out_pose_dir)
    
    if refine:
        new_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(extrinsic_dir, new_name), exist_ok=True)
    
    for index in tqdm.tqdm(index_list, desc="Debugging 2D keypoints"):
        # if os.path.exists(os.path.join(extrinsic_dir, name, index, "debug", "images")) and \
        #     len(os.listdir(os.path.join(extrinsic_dir, name, index, "debug", "images"))) == \
        #         len(os.listdir(os.path.join(extrinsic_dir, name, index, "images"))):
        #     continue
        
        img_dict_undistort = ImageDict.from_path(os.path.join(extrinsic_dir, name, index, "undistort"))
        img_dict_undistort.set_camparam(intrinsics, extrinsics)
        charuco_2d = img_dict_undistort.apply(detect_charuco, False)
        charuco_2d = {serial_num: merge_charuco_detection(detection) for serial_num, detection in charuco_2d.items()}
        
        kypt_3d_id = np.load(os.path.join(extrinsic_dir, name, index, "kypt_3d_id.npy"))
        kypt_3d_cor = np.load(os.path.join(extrinsic_dir, name, index, "kypt_3d_cor.npy"))
        
        proj_kypt_3d = img_dict_undistort.project_pointcloud(kypt_3d_cor)
        
        if refine:
            shutil.copytree(os.path.join(extrinsic_dir, name, index, "images"), 
                                os.path.join(extrinsic_dir, new_name, index, "images"))
        
        kypt_2d = {}
        
        idx_error_dict = {}
        for serial_num, det_2d in charuco_2d.items():
            idx_error_dict[serial_num] = []
            if serial_num not in err_dict:
                err_dict[serial_num] = []
                
            charuco_orig_2d = np.load(os.path.join(extrinsic_dir, name, index, "markers_2d", f"{serial_num}_corner.npy"))
            charuco_orig_id = np.load(os.path.join(extrinsic_dir, name, index, "markers_2d", f"{serial_num}_id.npy"))
            
            proj_2d = proj_kypt_3d[serial_num]
            
            kypt_2d[serial_num] = det_2d["checkerCorner"]
            kypt_id = det_2d["checkerIDs"]
            
            refined_kypt_id = []
            refined_kypt_2d = []
            
            for mid, cor_2d in zip(kypt_id, kypt_2d[serial_num]):
                if not refine and mid not in charuco_orig_id:
                    continue
                
                idx = np.where(kypt_3d_id == mid)[0]
                if len(idx) == 0:
                    continue
                
                proj_cor_2d = proj_2d[idx[0]]
                err = np.linalg.norm(cor_2d - proj_cor_2d)
                idx_error_dict[serial_num].append(err)
                if err < 2.0 and refine:
                    new_id = np.where(charuco_orig_id == mid)[0]
                    if len(new_id) == 0:
                        continue
                    
                    refined_kypt_id.append(mid)
                    refined_kypt_2d.append(charuco_orig_2d[new_id[0]])
                    
            
            if refine:
                refined_kypt_id = np.array(refined_kypt_id)
                refined_kypt_2d = np.array(refined_kypt_2d)
                os.makedirs(os.path.join(extrinsic_dir, new_name, index, "markers_2d"), exist_ok=True)
                np.save(os.path.join(extrinsic_dir, new_name, index, "markers_2d", f"{serial_num}_id.npy"), refined_kypt_id)
                np.save(os.path.join(extrinsic_dir, new_name, index, "markers_2d", f"{serial_num}_corner.npy"), refined_kypt_2d)
                
                
        plot_2d = img_dict_undistort.draw_keypoint(kypt_2d, (255, 0, 0))
        plot_proj = plot_2d.draw_keypoint(proj_kypt_3d, (0, 255, 0))
        
        plot_proj.save(os.path.join(extrinsic_dir, name, index, "debug"))
        
        for serial_num, err_list in idx_error_dict.items():
            if len(err_list) == 0:
                continue
            err_dict[serial_num].extend(err_list)
            if np.max(err_list) > 2.0:
                os.makedirs(os.path.join(extrinsic_dir, name, index_list[0], "outlier"), exist_ok=True)
                cv2.imwrite(os.path.join(extrinsic_dir, name, index_list[0], "outlier", f"{index}_{serial_num}.png"), plot_proj.images[serial_num])
                print(f"Index {index}, Serial {serial_num}, Max reproj err: {np.max(err_list)}")
    return err_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--name", type=str, help="Name of the directory to detect keypoint.")
    
    args = parser.parse_args()
    
    if args.name is None:
        name = find_latest_directory(extrinsic_dir)
    else:
        name = args.name

    run_calibration(name)
    index_list = sorted(os.listdir(os.path.join(extrinsic_dir, name)))
    out_pose_dir = os.path.join(extrinsic_dir, name, index_list[0], "colmap")
    intrinsics, extrinsics = load_colmap_camparam(out_pose_dir)

    undistort(name)
    save_kypt_3d(name)
    get_length(name)
    err_dict = debug(name)
    with open(os.path.join(extrinsic_dir, name, index_list[0], " reproj_error.txt"), 'w') as f:
        for serial_num, proj in err_dict.items():
            f.write(f"{serial_num} : mean {np.mean(proj)}, max{np.max(proj)} \n")
    
    new_name = find_latest_directory(extrinsic_dir)
    index_list = sorted(os.listdir(os.path.join(extrinsic_dir, new_name)))  
    run_calibration(new_name)
    undistort(new_name)
    save_kypt_3d(new_name)
    length = get_length(new_name)
    err_dict = debug(new_name, refine=False)
    
    with open(os.path.join(extrinsic_dir, new_name, index_list[0], " reproj_error.txt"), 'w') as f:
        for serial_num, proj in err_dict.items():
            f.write(f"{serial_num} : mean {np.mean(proj)}, max{np.max(proj)} \n")
    
    intrinsics, extrinsics = load_colmap_camparam(os.path.join(extrinsic_dir, new_name, index_list[0], "colmap"))
    new_extrinsics = {}
    for serial_num, extrinsic in extrinsics.items():
        new_extrinsic = np.array(extrinsic)
        new_extrinsic[:3, 3] *= (0.06/ np.mean(length))
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
    