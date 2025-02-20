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
import cv2
from cv2 import aruco

import sys
sys.path.append("..")

from .database import *


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)


with open("./config/charuco_info.json", "r") as f:
    boardinfo = json.load(f)

board_idx = 1
cb = boardinfo[str(board_idx)]
board = aruco.CharucoBoard( (cb["numX"], cb["numY"]), cb["checkerLength"], cb["markerLength"], aruco_dict, np.array(cb["markerIDs"]))
board_info = (board, int(cb["numMarker"]), 0) 

corners3d = board_info[0].getChessboardCorners()
arucoDetector = aruco.ArucoDetector(aruco_dict)
arucoDetector_tuned = aruco.ArucoDetector(aruco_dict)
params = arucoDetector_tuned.getDetectorParameters()
# # set detect parameters
"""
Edge detection
"""
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 28
params.adaptiveThreshWinSizeStep = 26

"""
Contour filtering
"""
params.minMarkerPerimeterRate = 0.01
params.maxMarkerPerimeterRate = 6.0
params.minCornerDistanceRate = 0.01
params.minMarkerDistanceRate = 0.01
params.polygonalApproxAccuracyRate = 0.07

"""
Marker Identification
"""
params.minOtsuStdDev = 10.0
params.perspectiveRemovePixelPerCell = 6
params.perspectiveRemoveIgnoredMarginPerCell = 0.25
params.errorCorrectionRate = 0.8 

"""
Corner Refinement
"""
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 100
params.cornerRefinementMinAccuracy = 0.1

"""
Using newer version
"""
params.useAruco3Detection = True
#params.useGlobalThreshold = True
arucoDetector_tuned.setDetectorParameters(params) 

shared_dir = "/home/capture18/paradex/config"

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
    F = pycolmap.estimate_fundamental_matrix(pix1, pix2)
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


    with open(f"{shared_dir}/camera.json", "r") as f:
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
        cur_lens_type = camera_lens[cur_serial]["lens"]

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




def detect_charuco_features(frame_path):
    global board_info

    ret = {"detected_corners": {}, "detected_markers": {}, "detected_ids": {}, "detected_mids": {}}
    
    scene_list = os.listdir(frame_path)
    scene_list.sort()

    b = board_info
    cur_board, cur_board_id = b[0], b[2]
    charDet = aruco.CharucoDetector(cur_board)

    for scene_idx, scene_name in enumerate(scene_list[:2]):
        scene_path = pjoin(frame_path, scene_name)

        for img_name_tot in os.listdir(scene_path):
            img_name = img_name_tot.split(".")[0]
            img_path = pjoin(scene_path, img_name_tot)
            img = cv2.imread(img_path)
            charCorner, charIDs, markerCorner, markerIDs = charDet.detectBoard(img)
            
            if img_name not in ret["detected_corners"]:
                ret["detected_corners"][img_name] = []
                ret["detected_ids"][img_name] = []
                ret["detected_markers"][img_name] = []
                ret["detected_mids"][img_name] = []

            if charIDs is not None:
                ret["detected_corners"][img_name].append(charCorner[:,0,:])
                ret["detected_ids"][img_name].append((charIDs + 70*scene_idx)[:,0])
                for val in markerCorner:
                    ret["detected_markers"][img_name].append(val)
                ret["detected_mids"][img_name].append(markerIDs)


    for img_name in list(ret["detected_corners"].keys()):    
        if len(ret["detected_corners"][img_name]) > 0:
            ret["detected_corners"][img_name] = np.concatenate(ret["detected_corners"][img_name], axis=0)
            ret["detected_ids"][img_name] = np.concatenate(ret["detected_ids"][img_name], axis=0)
            #detected_markers = np.concatenate(detected_markers, axis=0)
            ret["detected_mids"][img_name] = np.concatenate(ret["detected_mids"][img_name], axis=0)
    return ret

def get_colmap_images(database_path):
    """
    Fetches image IDs and names from the COLMAP database using an SQL query.
    If no images exist in the database, return an empty dictionary.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, name FROM images")
    images = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()
    return images

def feature_matching(database_path):
    cmd = f"colmap sequential_matcher --database_path {database_path} --SiftMatching.use_gpu 0"
    os.system(cmd)


if __name__ == "__main__":
    root_dir = "/home/capture18/captures1/calib_0217"
    
    out_pose_dir = pjoin(root_dir, "out")
    frame_dir = pjoin(root_dir, "frames")

    num_cam = 24#len(os.listdir(frame_dir))
    if not os.path.exists(out_pose_dir):
        os.mkdir(out_pose_dir)

    with open("src/calibration/options.yaml", "r") as f:
        options = yaml.safe_load(f)

    database_path = root_dir + "/database.db"

    #charuco_feature_extraction(database_path=databasepth, image_path=frame_dir) # we use undistorted cameras
    # feature_matching(database_path=databasepth)

    # sift_merge_database(root_dir, num_cam)
    # db_path = glob(root_dir+"/*.db")[0]
    # mapperOptions = pycolmap.IncrementalPipelineOptions(options['MapperOptions'])
    # maps = pycolmap.incremental_mapping(db_path, ".", out_pose_dir, options = mapperOptions)
    # maps[0].write_text(out_pose_dir)

    scene_list = os.listdir(frame_dir)
    scene_list.sort()

    ret = detect_charuco_features(frame_dir)
    print(ret["detected_corners"].keys())
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

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
    num_cameras = 24

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
        camera_id = db.add_camera(4, width, height, np.array([fx,fy, cx, cy, k1, k2, p1, p2]), 0)
        _ = db.add_image(f"{cur_serial}.png", camera_id)
        db.add_keypoints(camera_id, ret["detected_corners"][f"{cur_serial}"])

    img_keys = db.get_images()
    cam_keys = db.get_camera()

    for i in range(1, num_cameras+1):
        for j in range(i+1, num_cameras+1):
            
            cam_id1 = img_keys[i]["camera_id"]
            cam_id2 = img_keys[j]["camera_id"]

            cam_serial1 = camera_index_inv[cam_id1]
            cam_serial2 = camera_index_inv[cam_id2]

            corners1 = ret["detected_corners"][f"{cam_serial1}"]
            ids1 = ret["detected_ids"][f"{cam_serial1}"]

            corners2 = ret["detected_corners"][f"{cam_serial2}"]
            ids2 = ret["detected_ids"][f"{cam_serial2}"]

            common_ids, idx1, idx2 = np.intersect1d(ids1, ids2, return_indices=True)
            if len(common_ids) > 0:
                matches = np.column_stack((idx1, idx2))  # Pair indices of matching keypoints
                
                db.add_matches(i, j, matches)
                print(f"Added {len(matches)} matches between {i} and {j}")

                try:
                    twoviewgeom = get_two_view_geometries(cam_keys[cam_id1], cam_keys[cam_id2], corners1[idx1], corners2[idx2], matches, (i,j))

                    db.add_two_view_geometry(*twoviewgeom)
                except:
                    pass
    db.commit()
    db.close()

    mapperOptions = pycolmap.IncrementalPipelineOptions(options['MapperOptions'])
    maps = pycolmap.incremental_mapping(database_path, ".", out_pose_dir, options = mapperOptions)
    maps[0].write_text(out_pose_dir)


    # for img_name, img_idx in enumerate(list(ret["detected_corners"].keys())):
    #     img_idx, 


    #     img_id = db.add_image(img_name, 1)
    #     db.add_keypoints(img_id, ret["detected_corners"][img_name])
    #     db.add_descriptors(img_id, np.zeros((ret["detected_corners"][img_name].shape[0], 128)))
    #     db.add_points(ret["detected_corners"][img_name], ret["detected_ids"][img_name], ret["detected_mids"][img_name])
