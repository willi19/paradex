import argparse
import os
import json
import yaml
import copy
import shutil
from os.path import join as pjoin
from itertools import chain
from multiprocessing import Pool
from glob import glob
import pycolmap


import sys
sys.path.append("..")

from calibration.database import *

shared_dir = "/home/capture18/captures1"

# def build_framewise_directories(root):
#     all_frames_dir = os.path.join(root, "frames")
#     if not os.path.exists(all_frames_dir):
#         os.mkdir(all_frames_dir)

#     # get number of frames
#     cameras = [path for path in glob(root + "/*") if os.path.isdir(path) and path.split("/")[-1].isdigit()] # goes only upto 9
#     if len(cameras) == 0:
#         print("Directory has no camera folders!")
#         exit()
#     # num_frames = len(glob(os.path.join(cameras[0], "raw", "*")))

#     # build directories
#     num_frames= 1 # build camera once
#     for i in range(1, num_frames+1):
#         frame_dir = os.path.join(all_frames_dir, str(i).zfill(3))
#         if not os.path.exists(frame_dir):
#             os.mkdir(frame_dir)
#         for cam in cameras:
#             camera_id = cam.split("/")[-1]
#             if os.path.exists(os.path.join(cam, "raw")):
#                 shutil.copy(os.path.join(cam, "raw", str(i).zfill(5)+".png"), os.path.join(frame_dir, camera_id+".png")) # from camera image to frame folder

def get_two_view_geometries(cam1, cam2, pix1, pix2, indices, pair): # get tuple of cameras
    pycam1 = pycolmap.Camera(model="OPENCV", width=cam1["width"], height=cam1["height"], params=list(cam1["params"].reshape(-1)))
    pycam2 = pycolmap.Camera(model="OPENCV", width=cam2["width"], height=cam2["height"], params=list(cam2["params"].reshape(-1)))
    E = pycolmap.estimate_essential_matrix(pix1, pix2, pycam1, pycam2)
    F = pycolmap.estimate_fundamental_matrix(pix1, pix2)['F']
    H = pycolmap.estimate_homography_matrix(pix1, pix2)['H']
    # database is shared resource here
    return pair[0], pair[1], indices, F, E['E'], H, 3 # ways to getting two-view-geometries

# frames_dir : <some_dir>/calibration/frames, root_dir : <some_dir>/calibration
def sift_merge_database(root_dir, num_cam):
    # open and connect existing frames' datasets
    db_path = os.path.join(root_dir, "database_merged.db")

    if os.path.exists(db_path):
        os.remove(db_path)
        print("Database file already exists! Delete the file and re-run")

    db_merged = COLMAPDatabase.connect(db_path)
    db_merged.create_tables()
    num_cameras = num_cam


    with open(f"{shared_dir}/camera_lens.json", "r") as f:
        camera_lens = json.load(f) # {camera_serial : lens_type}

    with open(f"{shared_dir}/camera_index.json", "r") as f:
        camera_index = json.load(f)  # {camera_serial : camera_index}

    with open(f"{shared_dir}/lens_info.json", "r") as f:
        lens_info = json.load(f)

    camera_index_inv = dict()
    for k,v in camera_index.items():
        camera_index_inv[v] = k

    # Initial camera intrinsics
    width, height = 2048, 1536
    cx, cy = width // 2, height // 2



    match_pair, keypoints, descriptors = dict(), dict(), dict()
    offset_per_camera = dict( (i,0) for i in range(1, num_cameras+1))

    for i in range(1, num_cameras+1): # add 50 cameras
        cur_serial = camera_index_inv[i]
        cur_lens_type = camera_lens[cur_serial]

        if str(cur_lens_type) not in lens_info:
            print("Choose appropriate lens type!")
            exit(-1)

        cur_lens_info = lens_info[str(cur_lens_type)]
        fx = cur_lens_info["fx"]
        fy = cur_lens_info["fy"]
        k1 = cur_lens_info["k1"]
        k2 = cur_lens_info["k2"]
        p1 = cur_lens_info["p1"]
        p2 = cur_lens_info["p2"]

        # OPENCV, (fx, fy, cx, cy, k1, k2, p1, p2) considered
        camera_id = db_merged.add_camera(4, width, height, np.array([fx,fy, cx, cy, k1, k2, p1, p2]), 0)
        _ = db_merged.add_image(f"{cur_serial}.png", camera_id)
        
    for db_path in [root_dir]:
        frame_num = 0 #int(db_path.split("/")[-1])
        print("----------Current frame : {}----------".format(frame_num))
        #frame_list.append((frame_num, copy.deepcopy(offset_per_camera))) # track changes
        cur_db = COLMAPDatabase.connect(os.path.join(db_path, "database.db"))
        cur_images, cur_keypoints, cur_matches, cur_descriptors = cur_db.get_images(), cur_db.get_keypoints(), cur_db.get_matches(), cur_db.get_descriptors()

        for img_id_pairs, indices in cur_matches.items():
            image_id1, image_id2 = img_id_pairs[0], img_id_pairs[1]
            image_id1_serial, image_id2_serial = cur_images[image_id1]["name"].split(".")[0], cur_images[image_id2]["name"].split(".")[0]

            cam_id1 = camera_index[image_id1_serial]
            cam_id2 = camera_index[image_id2_serial]
            # cam_id1, cam_id2 = cur_images[image_id1]["camera_id"], cur_images[image_id2]["camera_id"]
            cam_match_pair = tuple(sorted((cam_id1, cam_id2)))
            # add offset to keypoint indices and append keypoints & descriptors
            keypoint_idx1, keypoint_idx2 = indices[:,0], indices[:,1]
            keypoint_idx1 += offset_per_camera[cam_id1] # (m, )
            keypoint_idx2 += offset_per_camera[cam_id2] # (m, )
            if cam_id1 < cam_id2:
                idx_pair = np.column_stack([keypoint_idx1, keypoint_idx2])
            else:
                idx_pair = np.column_stack([keypoint_idx2, keypoint_idx1])
            if cam_match_pair in match_pair:
                match_pair[cam_match_pair] = np.concatenate((match_pair[cam_match_pair], idx_pair), axis=0)
            else:
                match_pair[cam_match_pair] = idx_pair

        image_idx = dict()
        # cumulate offsets, keypoints, descriptors
        for img_id in range(1, num_cameras+1): # image id
            cur_camera_serial = cur_images[img_id]["name"].split(".")[0]
            cur_camera_id = camera_index[cur_camera_serial]
            image_idx[img_id] = cur_camera_id
            N_keypoints = cur_keypoints[img_id].shape[0]
            if cur_camera_id in keypoints:
                keypoints[cur_camera_id] = np.concatenate((keypoints[cur_camera_id], cur_keypoints[img_id]), axis=0)
            else:
                keypoints[cur_camera_id] = cur_keypoints[img_id]
            if cur_camera_id in descriptors:
                descriptors[cur_camera_id] = np.concatenate((descriptors[cur_camera_id], cur_descriptors[img_id]), axis=0)
            else:
                descriptors[cur_camera_id] = cur_descriptors[img_id]
            offset_per_camera[cur_camera_id] += N_keypoints
        # cumulate matched indices & points


    # pycolmap camera pair information
    cameras = db_merged.get_camera()
    pair_infos = []
    print("Adding matches and prepare arguments")
    for pair, indices in match_pair.items():
        print("Add matches and prepare args {}".format(pair))
        db_merged.add_matches(pair[0], pair[1], indices) # image_id1, image_id2, match_indices
        cam1, cam2 = cameras[pair[0]], cameras[pair[1]]
        idx1, idx2 = indices[:,0], indices[:,1]
        kp1, kp2 = keypoints[pair[0]], keypoints[pair[1]]
        pix1, pix2 = kp1[idx1, :2], kp2[idx2, :2]
        pair_infos.append((cam1, cam2, pix1, pix2, indices, pair))
    pair_infos = chain(pair_infos)

    # write to database_merged.db and calculate two_view_geometries (initial pairs)
    with Pool(processes=12) as pool:
        print("calculating two-view geometry...")
        twoviewgeom = list(pool.starmap(get_two_view_geometries, pair_infos))

    print("Adding into database")
    for val in twoviewgeom : db_merged.add_two_view_geometry(*val)


    for i in range(1, 1+num_cameras):
        print("Adding cumulated keypoints & descriptors for camera {}...".format(i))
        db_merged.add_keypoints(i, keypoints[i])
        db_merged.add_descriptors(i, descriptors[i])
    db_merged.commit()
    db_merged.close()
    return

# def extract_and_match_features(all_frame_dir, match_options=dict(), extract_options=dict()):
#     all_frames = glob(all_frame_dir+"/*")
#     sift_extract_options = pycolmap.SiftExtractionOptions(extract_options)
#     sift_matching_options = pycolmap.SiftMatchingOptions(match_options)
#     database_path = os.path.join(all_frame_dir, "database.db")

#     pycolmap.extract_features(database_path, all_frame_dir, camera_model="OPENCV", sift_options=sift_extract_options)
#     pycolmap.match_exhaustive(database_path, sift_options=sift_matching_options)

def create_database(database_path):
    cmd = f"colmap database_creator --database_path {database_path}"
    os.system(cmd)

def feature_extraction(database_path, image_path, camera_model):
    cmd = f"colmap feature_extractor \
            --database_path {database_path} \
            --image_path {image_path} \
            --ImageReader.camera_model {camera_model} \
            --SiftExtraction.use_gpu 0" # Change this when GPU is available
    os.system(cmd)

def feature_matching(database_path):
    cmd = f"colmap sequential_matcher --database_path {database_path} --SiftMatching.use_gpu 0"
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", help="Keep data structure as formulated") # <some_dir>/sequence_# or <some_dir>/calibration
    args = parser.parse_args()

    root_dir = args.root_dir
    out_pose_dir = pjoin(root_dir, "out")
    frame_dir = pjoin(root_dir, "frames")

    num_cam = len(os.listdir(frame_dir))
    if not os.path.exists(out_pose_dir):
        os.mkdir(out_pose_dir)

    with open("options.yaml", "r") as f:
        options = yaml.safe_load(f)

    databasepth = root_dir + "/database.db"

    create_database(database_path=databasepth)
    feature_extraction(database_path=databasepth, image_path=frame_dir, camera_model="OPENCV") # we use undistorted cameras
    feature_matching(database_path=databasepth)

    sift_merge_database(root_dir, num_cam)
    db_path = glob(root_dir+"/*.db")[0]
    mapperOptions = pycolmap.IncrementalPipelineOptions(options['MapperOptions'])
    maps = pycolmap.incremental_mapping(db_path, ".", out_pose_dir, options = mapperOptions)
    maps[0].write_text(out_pose_dir)