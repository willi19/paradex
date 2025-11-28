"""
ArUco and CharUco marker detection module (optimized version)
- Caching for faster repeated detection
- Safe handling of empty detections
- Cleaner merging of multi-board results
"""

from cv2 import aruco
from typing import Tuple, List, Dict
import numpy as np
import cv2
import json
import os

from paradex.utils.system import config_dir
# ------------------------
# Dictionary / Detector cache
# ------------------------
aruco_dict = {
    "4X4_50": aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
    "4X4_100": aruco.getPredefinedDictionary(aruco.DICT_4X4_100),
    "4X4_250": aruco.getPredefinedDictionary(aruco.DICT_4X4_250),
    "4X4_1000": aruco.getPredefinedDictionary(aruco.DICT_4X4_1000),
    "5X5_50": aruco.getPredefinedDictionary(aruco.DICT_5X5_50),
    "5X5_100": aruco.getPredefinedDictionary(aruco.DICT_5X5_100),
    "5X5_250": aruco.getPredefinedDictionary(aruco.DICT_5X5_250),
    "5X5_1000": aruco.getPredefinedDictionary(aruco.DICT_5X5_1000),
    "6X6_50": aruco.getPredefinedDictionary(aruco.DICT_6X6_50),
    "6X6_100": aruco.getPredefinedDictionary(aruco.DICT_6X6_100),
    "6X6_250": aruco.getPredefinedDictionary(aruco.DICT_6X6_250),
    "6X6_1000": aruco.getPredefinedDictionary(aruco.DICT_6X6_1000),
    "7X7_50": aruco.getPredefinedDictionary(aruco.DICT_7X7_50),
    "7X7_100": aruco.getPredefinedDictionary(aruco.DICT_7X7_100),
    "7X7_250": aruco.getPredefinedDictionary(aruco.DICT_7X7_250),
    "7X7_1000": aruco.getPredefinedDictionary(aruco.DICT_7X7_1000),
}

boardinfo_dict = json.load(open(os.path.join(config_dir, "charuco_info.json"), "r"))


_aruco_detector_cache = {}
_charuco_detector_cache = {}
_charuco_board_cache = {}


def get_aruco_detector(dict_type: str):
    if dict_type not in _aruco_detector_cache:
        _aruco_detector_cache[dict_type] = aruco.ArucoDetector(aruco_dict[dict_type])
    return _aruco_detector_cache[dict_type]


def get_charuco_detector():
    key_list = list(boardinfo_dict.keys())
    for key in key_list:
        if key not in _charuco_detector_cache:
            boardinfo = boardinfo_dict[key]
            check_boardinfo_valid({key: boardinfo})
            
            board = aruco.CharucoBoard(
                (boardinfo["numX"], boardinfo["numY"]),
                boardinfo["checkerLength"],
                boardinfo["markerLength"],
                aruco_dict[boardinfo["dict_type"]],
                np.array(boardinfo["markerIDs"])
            )
            _charuco_board_cache[key] = board
            _charuco_detector_cache[key] = aruco.CharucoDetector(board)

    return _charuco_detector_cache

# --------------------------------------------------------------


def detect_aruco(img, dict_type='6X6_1000') -> Tuple[List[np.ndarray], np.ndarray]:
    detector = get_aruco_detector(dict_type)
    corners, IDs, _ = detector.detectMarkers(img)
    if IDs is None:
        return [], np.zeros((0, 1), dtype=np.int32)
    return corners, IDs


def check_boardinfo_valid(boardinfo):
    required = {"dict_type", "numX", "numY", "checkerLength", "markerLength", "markerIDs"}
    for b, cfg in boardinfo.items():
        missing = required - cfg.keys()
        assert len(missing) == 0, f"Missing fields {missing} for board {b}"


def detect_charuco(img):
    detection_results = {}
    detector = get_charuco_detector()
    
    for b_id, det in detector.items():
        checkerCorner, checkerIDs, _, _ = det.detectBoard(img)

        if checkerIDs is None or len(checkerIDs) == 0:
            continue

        obj_pts = _charuco_board_cache[b_id].getChessboardCorners().reshape(-1, 2)  # Nx2
        img_pts = checkerCorner.reshape(-1, 2)

        H, inliers = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, ransacReprojThreshold=1.0)

        corners_filtered = img_pts[inliers.ravel() == 1]
        ids_filtered = checkerIDs[inliers.ravel() == 1]        

        detection_results[b_id] = {
            "checkerCorner": corners_filtered[:, 0, :],
            "checkerIDs": ids_filtered[:, 0],
        }
    
    return detection_results


def merge_charuco_detection(detection_list):

    offset_map = {}
    offset = 0
    for b_id, cfg in boardinfo_dict.items():
        offset_map[b_id] = offset
        offset += (cfg["numX"] - 1) * (cfg["numY"] - 1)

    corners_all = []
    ids_all = []

    for b_id, det in detection_list.items():
        ck = det["checkerCorner"]
        ids = det["checkerIDs"] + offset_map[b_id]

        corners_all.append(ck)
        ids_all.append(ids)

    if len(corners_all) == 0:
        return {"checkerCorner": np.zeros((0, 2)), "checkerIDs": np.zeros((0, 1), dtype=np.int32)}

    return {
        "checkerCorner": np.concatenate(corners_all, axis=0),
        "checkerIDs": np.concatenate(ids_all, axis=0),
    }


def draw_charuco(image, corners, color=(0, 255, 255), radius=4, thickness=2, ids=None):
    color = tuple(map(int, color))
    for i in range(len(corners)):
        x, y = map(int, corners[i])
        cv2.circle(image, (x, y), radius, color, thickness)
        if ids is not None:
            cv2.putText(image, str(int(ids[i])), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)


def draw_aruco(img, kypt, ids=None, color=(255, 0, 0)):
    color = tuple(map(int, color))
    for idx, corner in enumerate(kypt):
        corner = corner.squeeze().astype(int)
        if ids is not None:
            c_xy = tuple(np.mean(corner, axis=0).astype(int))
            cv2.putText(img, str(ids[idx]), c_xy, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for i in range(4):
            cv2.circle(img, tuple(corner[i]), 5, color, -1)
            cv2.putText(img, str(i), (corner[i][0] + 10, corner[i][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def draw_keypoint(img, kypt, color=(255, 0, 0)):
    color = tuple(map(int, color))
    pts = np.atleast_2d(kypt).astype(int)
    for x, y in pts:
        cv2.circle(img, (x, y), 1, color, -1)
    return img
