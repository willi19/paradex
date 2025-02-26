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


board_idx_list = [1, 2, 3, 4]
cb_list = [boardinfo[str(board_idx)] for board_idx in board_idx_list]

board_list = [aruco.CharucoBoard( (cb["numX"], cb["numY"]), cb["checkerLength"], cb["markerLength"], aruco_dict, np.array(cb["markerIDs"])) for cb in cb_list]
board_info_list = [(board, int(cb["numMarker"]), bi) for cb, board, bi in zip(cb_list, board_list, board_idx_list)]

corners3d = board_info_list[0][0].getChessboardCorners()
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


def detect_charuco_features(frame_path):
    global board_info_list

    ret = {"detected_corners": {}, "detected_markers": {}, "detected_ids": {}, "detected_mids": {}}
    
    scene_list = os.listdir(frame_path)
    scene_list.sort()


    for scene_idx, scene_name in enumerate(scene_list):
        scene_path = pjoin(frame_path, scene_name)
        os.makedirs(pjoin(root_dir, "keypoints", scene_name), exist_ok=True)

        for img_name_tot in os.listdir(scene_path):
            img_name = img_name_tot.split(".")[0]
            img_path = pjoin(scene_path, img_name_tot)
            img = cv2.imread(img_path)
            for board_idx, b in enumerate(board_info_list):
                cur_board, cur_board_id = b[0], b[2]

                if os.path.exists(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_charCorner.npy")):
                    charCorner = np.load(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_charCorner.npy"))
                    charIDs = np.load(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_charIDs.npy"))
                    markerCorner = np.load(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_markerCorner.npy"))
                    markerIDs = np.load(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_markerIDs.npy"))

                else:
                    charDet = aruco.CharucoDetector(cur_board)
                    charCorner, charIDs, markerCorner, markerIDs = charDet.detectBoard(img)
                    if charIDs is None:
                        charIDs = np.array([])
                        charCorner = np.array([])
                    if markerIDs is None:
                        markerIDs = np.array([])
                        markerCorner = np.array([])

                    np.save(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_charCorner.npy"), charCorner)
                    np.save(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_charIDs.npy"), charIDs)
                    np.save(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_markerCorner.npy"), markerCorner)
                    np.save(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_markerIDs.npy"), markerIDs)
                
                if img_name not in ret["detected_corners"]:
                    ret["detected_corners"][img_name] = []
                    ret["detected_ids"][img_name] = []
                    ret["detected_markers"][img_name] = []
                    ret["detected_mids"][img_name] = []

                if charIDs.shape != (0,):
                    ret["detected_corners"][img_name].append(charCorner[:,0,:])
                    ret["detected_ids"][img_name].append((charIDs + 70*len(board_info_list)*scene_idx + 70*cur_board_id)[:,0])
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
    root_dir = "/home/capture18/captures1/calib_0221_2"
    
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

    os.makedirs(root_dir + "/keypoints", exist_ok=True)
    ret = detect_charuco_features(frame_dir)

    print(ret["detected_corners"].keys())
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

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
    num_cameras = 24

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
        cx = cur_lens_info["cx"]
        cy = cur_lens_info["cy"]

        # OPENCV, (fx, fy, cx, cy, k1, k2, p1, p2) considered
        camera_id = db.add_camera(4, width, height, np.array([fx,fy, cx, cy, k1, k2, p1, p2]), 0)
        _ = db.add_image(f"{cur_serial}.png", camera_id)
        db.add_keypoints(camera_id, ret["detected_corners"][f"{cur_serial}"])

    img_keys = db.get_images()
    cam_keys = db.get_camera()

    os.makedirs(root_dir + "/matches", exist_ok=True)

    for i in range(1, num_cameras+1):
        for j in range(i+1, num_cameras+1):
            
            cam_id1 = img_keys[i]["camera_id"]
            cam_id2 = img_keys[j]["camera_id"]

            cam_serial1 = camera_index_inv[cam_id1]
            cam_serial2 = camera_index_inv[cam_id2]

            if os.path.exists(root_dir + "/matches" + f"/{cam_serial1}_{cam_serial2}.npy"):
                matches = np.load(root_dir + "/matches" + f"/{cam_serial1}_{cam_serial2}.npy")
                db.add_matches(i, j, matches)
                print(f"Added {len(matches)} matches between {i} and {j}")
                try:
                    twoviewgeom = get_two_view_geometries(cam_keys[cam_id1], cam_keys[cam_id2], corners1[idx1], corners2[idx2], matches, (i,j))

                    db.add_two_view_geometry(*twoviewgeom)
                except:
                    pass
            else:   
                corners1 = ret["detected_corners"][f"{cam_serial1}"]
                ids1 = ret["detected_ids"][f"{cam_serial1}"]

                corners2 = ret["detected_corners"][f"{cam_serial2}"]
                ids2 = ret["detected_ids"][f"{cam_serial2}"]

                common_ids, idx1, idx2 = np.intersect1d(ids1, ids2, return_indices=True)
                if len(common_ids) > 0:
                    matches = np.column_stack((idx1, idx2))  # Pair indices of matching keypoints
                    np.save(root_dir + "/matches" + f"/{cam_serial1}_{cam_serial2}.npy", matches)

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
