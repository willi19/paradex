
import json

import cv2
import numpy as np
from cv2 import aruco
from typing import Tuple, List, Dict
from pathlib import Path
import glob

from paradex.utils.io import load_cam_param, load_images
# from flir_python.utils.file_io import load_cam_param, load_images


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)


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
# params.useGlobalThreshold = True
arucoDetector_tuned.setDetectorParameters(params)


def detect_aruco_process(img_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    cam_num = int(img_path.split("/")[-3])
    frame_num = int(img_path.split("/")[-1][:-4])
    img = cv2.imread(img_path)
    corners, corners_t, IDs, IDs_t = detect_aruco_tuned(img)
    return corners, corners_t, IDs, IDs_t, cam_num, frame_num


def detect_aruco_tuned(img) -> Tuple[np.ndarray, np.ndarray]:
    global arucoDetector_tuned
    corners, IDs, _ = arucoDetector.detectMarkers(img)
    corners_t, IDs_t, _ = arucoDetector_tuned.detectMarkers(img)  # marker corners 의미?
    # Merge detected
    return corners, corners_t, IDs, IDs_t  # if not detected, returns none


def detect_aruco(img) -> Tuple[np.ndarray, np.ndarray]:
    corners, IDs, _ = arucoDetector.detectMarkers(img)
    return corners, IDs


# Detect charuco board
def detect_charuco(img):

    all_boards = []  # to be used in detection
    # for j in range(1, args.board_num+1): # read all board info

    homedir = Path().home()
    path = str(
        homedir
        / "dexterous-hri/dexterous_hri/utils/flir_python/flir_python/charuco_info.json"
    )

    with open(
        path,
        "r",
    ) as f:
        boardinfo = json.load(f)

    board_num = [0, 2, 3, 4]  #

    for board_cnt, idx in enumerate(board_num):
        cb = boardinfo[str(idx)]  # idx : board index
        board = aruco.CharucoBoard(
            (cb["numX"], cb["numY"]),
            cb["checkerLength"],
            cb["markerLength"],
            aruco_dict,
            np.array(cb["markerIDs"]),
        )
        
        all_boards.append((board, int(cb["numMarker"]), idx))  # j for offset ID

    corners3d = all_boards[0][0].getChessboardCorners()

    detected_corners, detected_ids = [], []
    detected_markers, detected_mids = [], []
    # initial board num
    cur_board, cur_board_id = all_boards[0][0], all_boards[0][2]
    
    charDet = aruco.CharucoDetector(cur_board)
    charCorner, charIDs, markerCorner, markerIDs = charDet.detectBoard(img)
    if charIDs is not None:
        aruco.drawDetectedCornersCharuco(img, charCorner, charIDs)
        detected_corners.append(charCorner)
        detected_ids.append(charIDs)
        for val in markerCorner:
            detected_markers.append(val)  # tuple
        detected_mids.append(markerIDs)

    for b in all_boards[1:]:
        # import ipdb; ipdb.set_trace()
        cur_board, cur_board_id = b[0], b[2]
        charDet = aruco.CharucoDetector(cur_board)
        charCorner, charIDs, markerCorner, markerIDs = charDet.detectBoard(img)
        if charIDs is not None:

            aruco.drawDetectedCornersCharuco(
                img, charCorner, charIDs + 70 * (cur_board_id - 2) + 48
            )
            detected_corners.append(charCorner)
            detected_ids.append(charIDs + 70 * (cur_board_id - 2) + 48)
            for val in markerCorner:
                detected_markers.append(val)  # tuple
            detected_mids.append(markerIDs)
            # cv2.imshow("asdf",img)
            # cv2.waitKey(0)

    if len(detected_corners) > 0:
        detected_corners = np.concatenate(detected_corners, axis=0)
        detected_ids = np.concatenate(detected_ids, axis=0)
        # detected_markers = np.concatenate(detected_markers, axis=0)
        detected_mids = np.concatenate(detected_mids, axis=0)
    return (detected_corners, detected_markers), (detected_ids, detected_mids)


def triangulate(corners: dict, projections: list):  # Previus code from jh
    """
    N : number of images with same marker
    corners : {1: (N,2) array, 2:(N,2) array, 3:(N,2) array, 4: (N,2) array}
    projections : list of 3x4 matrices of length N
    """
    numImg = len(projections)
    kp3d = dict()
    # print("triangulate")
    for corner_id, kps in corners.items():
        A = []
        for i in range(numImg):
            curX, curY = kps[i, 0], kps[i, 1]
            cur_proj = projections[i]
            A.append(curY * cur_proj[2] - cur_proj[1])
            A.append(curX * cur_proj[2] - cur_proj[0])
        # print(A, numImg)
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        kp3d[corner_id] = V[3][0:3] / V[3][3]  #
    return kp3d


def get_marker_from_imageset(
    img_path, cam_param_path
):  # give marker3d from distorted cameraset
    img_list = load_images(img_path)
    intrinsic, extrinsic = load_cam_param(cam_param_path)

    undist_cam_param = {}

    for cam_name, cam in intrinsic.items():
        undist_cam_param[cam_name] = cam["intrinsics_undistort"]

    markers = {}
    proj_mtx = {}

    for img_name, img in img_list.items():
        corners, ids = detect_aruco(img)
        original_intrinsics = np.array(
            intrinsic[img_name]["intrinsics_original"]
        ).reshape(3, 3)
        if len(corners) == 0:
            continue
        for i, id in enumerate(ids):
            id = int(id)
            if id not in markers:
                markers[id] = []
                proj_mtx[id] = []

            normalized_coords = cv2.undistortPoints(
                corners[i], original_intrinsics, intrinsic[img_name]["dist_param"]
            )
            mapped_pixels = normalized_coords.squeeze() * np.array(
                [[undist_cam_param[img_name][0, 0], undist_cam_param[img_name][1, 1]]]
            ) + np.array(
                [[undist_cam_param[img_name][0, 2], undist_cam_param[img_name][1, 2]]]
            )
            markers[id].append(mapped_pixels)
            proj_mtx[id].append(
                undist_cam_param[img_name]
                @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
                @ extrinsic[img_name]["w2c"]
            )
    pts_3d = {}
    for marker_ind in markers.keys():
        markers[marker_ind] = np.array(markers[marker_ind])
        proj_mtx[marker_ind] = np.array(proj_mtx[marker_ind])
        N = markers[marker_ind].shape[0]
        if N < 2:
            continue
        pts_3d[marker_ind] = triangulate_marker(
            markers[marker_ind], proj_mtx[marker_ind]
        )
    # visualize_marker(pts_3d, extrinsic)
    return pts_3d


def get_marker_from_undistorted_img(
    img_path, cam_param_path
):  # give marker3d from undistorted cameraset
    img_list = load_images(img_path)
    intrinsic, extrinsic = load_cam_param(cam_param_path)

    markers = {}
    proj_mtx = {}

    for img_name, img in img_list.items():
        corners, ids = detect_aruco(img)

        if len(corners) == 0:
            continue
        for i, id in enumerate(ids):
            id = int(id)
            if id not in markers:
                markers[id] = []
                proj_mtx[id] = []
            # normalized_coords = cv2.undistortPoints(corners[i], intrinsic[img_name]['Intrinsics'], intrinsic[img_name]['dist_param'])
            # mapped_pixels = normalized_coords.squeeze() * np.array([[undist_cam_param[img_name][0,0], undist_cam_param[img_name][1,1]]]) + np.array([[undist_cam_param[img_name][0,2], undist_cam_param[img_name][1,2]]])
            markers[id].append(corners[i])
            proj_mtx[id].append(
                intrinsic[img_name]["Intrinsics"]
                @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
                @ extrinsic[img_name]["w2c"]
            )

    pts_3d = {}
    for marker_ind in markers.keys():
        markers[marker_ind] = np.array(markers[marker_ind])
        proj_mtx[marker_ind] = np.array(proj_mtx[marker_ind])
        pts_3d[marker_ind] = triangulate_marker(
            markers[marker_ind], proj_mtx[marker_ind]
        )

    return pts_3d


def triangulate_marker(corners, projections):  # triangulate marker
    """
    N : number of images with same marker
    corners : (N, 4, 2) array
    projections : (N, 3, 4) array
    """
    N = corners.shape[0]
    assert corners.shape == (N, 4, 2)
    assert projections.shape == (N, 3, 4)
    # if N < 3:
    #    print("Not enough images to triangulate")
    corner_3d = []
    for corner_id in range(4):
        A = []
        for i in range(N):
            curX, curY = corners[i, corner_id, 0], corners[i, corner_id, 1]
            cur_proj = projections[i]
            A.append(curY * cur_proj[2] - cur_proj[1])
            A.append(curX * cur_proj[2] - cur_proj[0])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        # print(S, U.shape, V.shape)
        kp3d = V[3][0:3] / V[3][3]  #
        corner_3d.append(kp3d)
    return np.array(corner_3d)  # (4, 3)


def reprojection_error(corners, corners_3d, projections):
    """
    corners : (N, 4, 2) array
    corners_3d : (4, 3) array
    projections : (N, 3, 4) array
    """
    N = corners.shape[0]
    if isinstance(corners, np.ndarray):
        corners = torch.tensor(corners).float()
    if isinstance(corners_3d, np.ndarray):
        corners_3d = torch.tensor(corners_3d).float()
    if isinstance(projections, np.ndarray):
        projections = torch.tensor(projections).float()
    assert corners.shape == (N, 4, 2)
    assert corners_3d.shape == (4, 3)
    assert projections.shape == (N, 3, 4)

    # detach corners, projections for gradient calculation
    corners.requires_grad = False
    projections.requires_grad = False

    # Add homogeneous coordinate to corners_3d
    corners_3d_hom = torch.cat([corners_3d, torch.ones(4, 1)], dim=1)  # (C, 4)

    # Project corners_3d using the projection matrices
    proj_corners = torch.einsum("nij,mj->nmi", projections, corners_3d_hom)  # (N, 4, 3)

    # Normalize by the third (homogeneous) coordinate
    proj_corners = proj_corners / proj_corners[:, :, 2].unsqueeze(2)  # (N, 4, 3)
    # Compute the reprojection error
    error = torch.norm(proj_corners[:, :, :2] - corners, dim=(2))  # (N, 4)
    error = torch.mean(error, dim=1)
    return error  # Convert back to numpy array for compatibility


def choose_inlier(corners, projections, corners_3d):
    """
    Perform RANSAC to select the best set of cameras
    corners : (N, 4, 2) array
    projections : (N, 3, 4) array
    """
    N = corners.shape[0]

    # corners_3d = triangulate_marker(corners, projections)

    corners = torch.tensor(corners).float()
    corners_3d = torch.tensor(corners_3d).float()
    projections = torch.tensor(projections).float()

    error = reprojection_error(corners, corners_3d, projections).numpy()
    # print(error, "error")
    inlier_mask = np.where(error < 1.5)
    # print(inlier_mask, error,"error")
    return inlier_mask


def triangulate_marker_ransac(corners, projections):
    """
    corners : (N, 4, 2) array
    projections : (N, 3, 4) array
    """
    N = corners.shape[0]
    inlier_cnt = 0
    best_3d = None
    inlier = None
    # print("triangulate_marker_ransac", N)
    for i in range(N):
        for j in range(i + 1, N):
            selected_corners = np.array([corners[i], corners[j]])
            selected_projections = np.array([projections[i], projections[j]])

            init_corners_3d = triangulate_marker(selected_corners, selected_projections)
            inlier_mask = choose_inlier(corners, projections, init_corners_3d)

            # print(len(inlier_mask[0]), "inlier_update", i, j, N)
            if len(inlier_mask[0]) > inlier_cnt:
                inlier_cnt = len(inlier_mask[0])
                best_3d = triangulate_marker(
                    corners[inlier_mask], projections[inlier_mask]
                )
                inlier = inlier_mask
    # print(inlier, "inlier")
    # print(inlier_cnt, "inlier_cnt")
    # print(best_3d, "best_3d")

    corners_t = torch.tensor(corners[inlier]).float()
    corners_3d_t = torch.tensor(best_3d).float()
    corners_3d_t.requires_grad = True
    projections_t = torch.tensor(projections[inlier]).float()

    # Perform optimization
    optimizer = Adam([corners_3d_t], lr=0.001)
    for iter in range(10):
        optimizer.zero_grad()
        error = reprojection_error(corners_t, corners_3d_t, projections_t).mean()
        # print(error)
        error.backward()
        optimizer.step()

    # diff = np.linalg.norm(init_corners_3d - corners_3d_t.detach().numpy(), axis=1)
    # print(diff, "diff")
    return corners_3d_t.detach().numpy(), inlier


def draw_aruco(img):
    corners, ids = detect_aruco(img)
    if ids is not None:
        # import ipdb; ipdb.set_trace()
        for idx, (cor, i) in enumerate(zip(corners, ids)):
            for j in range(4):
                # cv2.line(img, tuple(corners[i][0][j]), tuple(corners[i][0][(j+1)%4]), (0, 0, 255), 2)
                pt1 = tuple(int(x) for x in cor[0][j])
                pt2 = tuple(int(x) for x in cor[0][(j + 1) % 4])
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)

                # cv2.text(img, str(i), tuple(corners[i][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(
                    img,
                    str(j),
                    (int(cor[0][j][0]), int(cor[0][j][1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            center = np.mean(cor, axis=1)[0]
            cv2.putText(
                img,
                str(i[0]),
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

    cv2.imshow("aruco", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_marker_pts(scene_path):
    pc_npz_list = glob.glob(scene_path + "/*.npz")

    timestamp_marker_pts = {}  # Timestamp : {Marker_id : {camera, pts}}

    for pc_path in pc_npz_list:
        pc_file = np.load(pc_path, allow_pickle=True)
        for timestamp, detect_pts_hist in pc_file.items():
            detect_pts_hist = detect_pts_hist.item()
            for cam_name, marker_pts in detect_pts_hist.items():
                if timestamp not in timestamp_marker_pts:
                    timestamp_marker_pts[timestamp] = {}
                for marker_id, marker_pts in marker_pts.items():
                    # print(marker_id)
                    if marker_id not in timestamp_marker_pts[timestamp]:
                        timestamp_marker_pts[timestamp][marker_id] = {}

                    timestamp_marker_pts[timestamp][marker_id][cam_name] = marker_pts

    return timestamp_marker_pts
