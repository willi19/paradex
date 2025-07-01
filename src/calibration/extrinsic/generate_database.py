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
from paradex.colmap.database import *
from paradex.colmap.colmap import get_two_view_geometries
from paradex.utils.file_io import find_latest_directory, home_path, download_dir, shared_dir, load_intrinsic
import tqdm
import matplotlib.pyplot as plt

def draw_keypoint(image, corners, color=(0, 255, 255), radius=4, thickness=2, ids=None):
    for i in range(len(corners)):
        corner = tuple(int(x) for x in corners[i])
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, corner, radius, color, thickness)
        if ids is not None:
            cv2.putText(image, str(int(ids[i])), (corner[0] + 5, corner[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)


download_dir = os.path.join(shared_dir, "extrinsic")
config_dir = "config"

def process_match(args):
    """ Function to process a single match pair """
    (cam_key1, cam_key2), matches, corners1, corners2, cam_id1, cam_id2 = args
    idx1 = matches[:, 0]
    idx2 = matches[:, 1]


    twoviewgeom = get_two_view_geometries(
        cam_key1, cam_key2, corners1[idx1], corners2[idx2], matches, (cam_id1, cam_id2)
    )
    return twoviewgeom

def parallel_processing(tot_kypt_matches, tot_kypt_dict, cam_keys):
    """ Parallelize the processing of keypoint matches """
    args = [((cam_keys[camera_id_dict[serial_1]], cam_keys[camera_id_dict[serial_2]]), np.vstack(matches, dtype=np.int32), tot_kypt_dict[serial_1], tot_kypt_dict[serial_2], camera_id_dict[serial_1], camera_id_dict[serial_2]) for (serial_1, serial_2), matches in tot_kypt_matches.items()]
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


if __name__ == "__main__":
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--name", type=str, help="Name of the directory to detect keypoint.")
    parser.add_argument("--latest", action="store_true", help="Split the latest video files.")
    
    with open(f"{config_dir}/camera/camera_index.json", "r") as f:
        camera_index = json.load(f)  # {camera_serial : camera_index}

    camera_index_inv = dict()
    for k,v in camera_index.items():
        camera_index_inv[v] = k

    args = parser.parse_args()
    
    if not args.latest and not args.name:
        print("Please specify either --latest or --name.")
        exit()
    
    if args.latest:
        name = find_latest_directory(download_dir)
    else:
        name = args.name

    intrinsics_dict = load_intrinsic()
    num_cameras = len(camera_index)

    root_dir = os.path.join(download_dir, name)
    keypoint_dict = load_keypoint(os.path.join(root_dir))

    N = max(keypoint_dict.keys()) + 1
    colormap = plt.cm.get_cmap("hsv", N)
    index2color = {idx: tuple((np.array(colormap(i)[:3]) * 255).astype(int)) for i, idx in enumerate(keypoint_dict)}

    images_dict = {serial_name: np.zeros((intrinsic['height'], intrinsic['width'], 3), dtype=np.uint8) for serial_name, intrinsic in intrinsics_dict.items()}
    minmax_id = {}

    for index, kypt in keypoint_dict.items():
        minmax_id[index] = {"max": -1, "min": 1000000}

        for serial_num, kypt_data in kypt.items():

            cor = kypt_data["corners"]
            ids = kypt_data["ids"]

            if cor.shape[0] == 0:
                continue
            
            draw_keypoint(images_dict[serial_num], cor, color = index2color[int(index)])
            
            minmax_id[index]["max"] = max(minmax_id[index]["max"], np.max(ids))
            minmax_id[index]["min"] = min(minmax_id[index]["min"], np.min(ids))

    os.makedirs(os.path.join(root_dir, "0", "images"), exist_ok=True)
    for serial_num, image in images_dict.items():
        cv2.imwrite(os.path.join(root_dir, "0", "images", f"{serial_num}.jpg"), image)
        
    out_pose_dir = os.path.join(root_dir, "0", "colmap")
    os.makedirs(out_pose_dir, exist_ok=True)

    with open("config/colmap_options.yaml", "r") as f:
        options = yaml.safe_load(f)

    database_path = out_pose_dir + "/database.db"
    if os.path.exists(database_path):
        os.remove(database_path)
        
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    # # Initial camera intrinsics
    

    kypt_offset = {}
    image_id_dict = {}
    camera_id_dict = {}
    for i in range(1, num_cameras+1): # add 50 cameras
        cur_serial = camera_index_inv[i]
        intrinsic = intrinsics_dict[cur_serial]
        
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

        # OPENCV, (fx, fy, cx, cy, k1, k2, p1, p2) considered
        camera_id = db.add_camera(4, width, height, np.array([fx,fy, cx, cy, k1, k2, p1, p2]), 0)
        image_id = db.add_image(f"{cur_serial}.jpg", camera_id)
        kypt_offset[cur_serial] = 0
        image_id_dict[cur_serial] = image_id
        camera_id_dict[cur_serial] = camera_id

    img_keys = db.get_images()
    cam_keys = db.get_camera()

    merged_kypt_dict = {}
    merged_kypt_matches = {}
    
    tot_kypt_dict = {}
    tot_kypt_matches = {}

    tot_cor_offset = 0
    serial_offset = {0 for serial in camera_index.keys()}

    for index, kypt_data in keypoint_dict.items():
        serial_list = list(kypt_data.keys())
        serial_list.sort()
        
        for i in range(len(serial_list)):
            for j in range(i+1, len(serial_list)):
                serial_1 = serial_list[i]
                serial_2 = serial_list[j]

                common_ids, idx1, idx2 = np.intersect1d(kypt_data[serial_1]["ids"], kypt_data[serial_2]["ids"], return_indices=True)
                idx1 += kypt_offset[serial_1]
                idx2 += kypt_offset[serial_2]
                
                if len(common_ids) > 0:
                    matches = np.column_stack((idx1, idx2))
                    if (serial_1, serial_2) not in tot_kypt_matches.keys():
                        tot_kypt_matches[(serial_1, serial_2)] = []
                    tot_kypt_matches[(serial_1, serial_2)].append(matches)

        for i in range(len(serial_list)):
            serial_1 = serial_list[i]
            if serial_1 not in tot_kypt_dict:
                tot_kypt_dict[serial_1] = []
            tot_kypt_dict[serial_1].append(kypt_data[serial_1]["corners"])
            kypt_offset[serial_1] += len(kypt_data[serial_1]["ids"])
        
    
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

    # colmap failes to much try colmap gui
    # mapperOptions = pycolmap.IncrementalPipelineOptions(options['MapperOptions'])
    # maps = pycolmap.incremental_mapping(database_path, ".", out_pose_dir, options = mapperOptions)
    # maps[0].write_text(out_pose_dir)

    # recon = maps[0]
