import argparse
import os
import json
import yaml
from itertools import chain
from multiprocessing import Pool
from glob import glob
import pycolmap
import cv2
from cv2 import aruco
import multiprocessing as mp
from paradex.calibration.database import *
from paradex.calibration.colmap import get_two_view_geometries
from paradex.utils.io import find_latest_directory, home_dir, download_dir
import tqdm

download_dir = os.path.join(download_dir,"calibration")
config_dir = "config"

def process_match(args):
    """ Function to process a single match pair """
    (cam_key1, cam_key2), matches, corners1, corners2, cam_id1, cam_id2 = args
    idx1 = matches[:, 0]
    idx2 = matches[:, 1]


    twoviewgeom = get_two_view_geometries(
        cam_key1, cam_key2, corners1[idx1], corners2[idx2], matches, (cam_id1, cam_id2)
    )
    return (serial_1, serial_2, twoviewgeom)

def parallel_processing(tot_kypt_matches, tot_kypt_dict, cam_keys):
    """ Parallelize the processing of keypoint matches """
    args = [((cam_keys[camera_id_dict[serial_1]], cam_keys[camera_id_dict[serial_2]]), np.vstack(matches, dtype=np.int32), tot_kypt_dict[serial_1], tot_kypt_dict[serial_2], camera_id_dict[serial_1], camera_id_dict[serial_2]) for (serial_1, serial_2), matches in tot_kypt_matches.items()]
    num_processes = mp.cpu_count()  # Use all available cores
    with mp.Pool(processes=num_processes) as pool:
        results = list(pool.map(process_match, args))
    # Add two-view geometry to database
    for serial_1, serial_2, twoviewgeom in results:
        if twoviewgeom is not None:
            db.add_two_view_geometry(*twoviewgeom)
    
    

if __name__ == "__main__":
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--name", type=str, help="Name of the directory to detect keypoint.")
    parser.add_argument("--latest", action="store_true", help="Split the latest video files.")
    
    args = parser.parse_args()
    
    if not args.latest and not args.name:
        print("Please specify either --latest or --name.")
        exit()
    
    if args.latest:
        name = find_latest_directory(download_dir)
    else:
        name = args.name

    root_dir = os.path.join(download_dir, name)
    index_list = os.listdir(root_dir)
    index_list.sort()
    if len(index_list) == 0:
        print("No valid directories found.")
        exit()

    out_pose_dir = os.path.join(root_dir, index_list[0], "colmap")
    os.makedirs(out_pose_dir, exist_ok=True)

    num_cam = 24
    with open("src/calibration/options.yaml", "r") as f:
        options = yaml.safe_load(f)

    database_path = out_pose_dir + "/database.db"

    keypoint_path_list = []
    for index in index_list:
        frame_dir = os.path.join(root_dir, index, "keypoints")
        keypoint_path_list += [os.path.join(frame_dir, d) for d in os.listdir(frame_dir) if int(d) % 10 == 0]

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    with open(f"{config_dir}/initial_intrinsic.json", "r") as f:
        camera_lens = json.load(f) # {camera_serial : lens_type}

    with open(f"{config_dir}/camera_index.json", "r") as f:
        camera_index = json.load(f)  # {camera_serial : camera_index}

    # with open(f"{config_dir}/lens_info.json", "r") as f:
    #     lens_info = json.load(f)

    camera_index_inv = dict()
    for k,v in camera_index.items():
        camera_index_inv[v] = k

    # Initial camera intrinsics
    width, height = 2048, 1536
    cx, cy = width // 2, height // 2
    num_cameras = 24

    kypt_offset = {}
    image_id_dict = {}
    camera_id_dict = {}
    for i in range(1, num_cameras+1): # add 50 cameras
        cur_serial = camera_index_inv[i]

        cur_lens_info = camera_lens[cur_serial]

        fx = cur_lens_info["original_intrinsics"][0]
        fy = cur_lens_info["original_intrinsics"][4]
        k1 = cur_lens_info["dist_param"][0]
        k2 = cur_lens_info["dist_param"][1]
        p1 = cur_lens_info["dist_param"][2]
        p2 = cur_lens_info["dist_param"][3]
        cx = cur_lens_info["original_intrinsics"][2]
        cy = cur_lens_info["original_intrinsics"][5]

        # OPENCV, (fx, fy, cx, cy, k1, k2, p1, p2) considered
        camera_id = db.add_camera(4, width, height, np.array([fx,fy, cx, cy, k1, k2, p1, p2]), 0)
        image_id = db.add_image(f"{cur_serial}.png", camera_id)
        kypt_offset[cur_serial] = 0
        image_id_dict[cur_serial] = image_id
        camera_id_dict[cur_serial] = camera_id

    img_keys = db.get_images()
    cam_keys = db.get_camera()

    tot_kypt_dict = {}
    tot_kypt_matches = {}

    for kypt_path in tqdm.tqdm(keypoint_path_list):
        # print(tot_kypt_dict.keys(), tot_kypt_matches.keys(), kypt_path)
        kypt_file_list = os.listdir(os.path.join(root_dir, kypt_path))
        kypt_dict = {}

        for kypt_file in kypt_file_list:
            if "ids" in kypt_file:
                continue
            serial_num = kypt_file.split("_")[0]

            kypt_dict[serial_num] = {}
            kypt_dict[serial_num]["corners"] = np.load(os.path.join(root_dir, kypt_path, kypt_file))
            kypt_dict[serial_num]["ids"] = np.load(os.path.join(root_dir, kypt_path, f"{serial_num}_ids.npy"))

        for serial_num, kypt in kypt_dict.items():
            image_id = image_id_dict[serial_num]
            camera_id = camera_id_dict[serial_num]
            if kypt["corners"].shape[0] == 0:
                continue
            if serial_num not in tot_kypt_dict:
                tot_kypt_dict[serial_num] = []
            tot_kypt_dict[serial_num].append(kypt["corners"])
        
        serial_list = list(kypt_dict.keys())
        serial_list.sort()
        
        for i in range(len(serial_list)):  
            for j in range(i+1, len(serial_list)):
                serial_1 = serial_list[i]
                serial_2 = serial_list[j]

                common_ids, idx1, idx2 = np.intersect1d(kypt_dict[serial_1]["ids"], kypt_dict[serial_2]["ids"], return_indices=True)
                idx1 += kypt_offset[serial_1]
                idx2 += kypt_offset[serial_2]
                
                if len(common_ids) > 0:
                    matches = np.column_stack((idx1, idx2))
                    # db.add_matches(image_id_dict[serial_1], image_id_dict[serial_2], matches)\
                    
                    if (serial_1, serial_2) not in tot_kypt_matches.keys():
                        tot_kypt_matches[(serial_1, serial_2)] = []
                    tot_kypt_matches[(serial_1, serial_2)].append(matches)

        for serial_num in serial_list:
            kypt_offset[serial_num] += len(kypt_dict[serial_num]["ids"])
    
    for serial_num, kypt_list in tot_kypt_dict.items():
        kypt_list = np.vstack(kypt_list)
        tot_kypt_dict[serial_num] = kypt_list
        camera_id = camera_id_dict[serial_num]
        db.add_keypoints(camera_id, kypt_list)

    for (serial_1, serial_2), matches in tot_kypt_matches.items():        
        image_id_1 = image_id_dict[serial_1]
        image_id_2 = image_id_dict[serial_2]
        matches = np.vstack(matches)
        idx1 = matches[:, 0]
        idx2 = matches[:, 1]
        
        # Add matches to database
        db.add_matches(image_id_1, image_id_2, matches)
    
    parallel_processing(tot_kypt_matches, tot_kypt_dict, cam_keys)

     
    

    db.commit()
    db.close()
    print("--- %s seconds ---" % (time.time() - start_time))
    mapperOptions = pycolmap.IncrementalPipelineOptions(options['MapperOptions'])
    maps = pycolmap.incremental_mapping(database_path, ".", out_pose_dir, options = mapperOptions)
    maps[0].write_text(out_pose_dir)
