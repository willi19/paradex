from cv2 import aruco
from typing import Tuple, List, Dict
import numpy as np
import cv2

aruco_type = ["4X4_50", "4X4_100", "4X4_250", "4X4_1000",
                "5X5_50", "5X5_100", "5X5_250", "5X5_1000",
                "6X6_50", "6X6_100", "6X6_250", "6X6_1000",
                "7X7_50", "7X7_100", "7X7_250", "7X7_1000"]

aruco_dict = {"4X4_50": aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
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
                "7X7_1000": aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)}

arucoDetector_dict = {dict_name: aruco.ArucoDetector(aruco_dict[dict_name]) for dict_name in aruco_type}

def detect_aruco(img, dict_type='6X6_1000') -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Detects ArUco markers in the given image.
    
    :param img: Input image in which to detect markers.
    :param dict_type: Type of ArUco dictionary to use for detection.
    :return: 
        corners (List[np.ndarray]): List of (4,2) float32 arrays for each detected marker's corners.
        IDs (np.ndarray): (N,1) int32 array of detected marker IDs.
    """
    corners, IDs, _ = arucoDetector_dict[dict_type].detectMarkers(img)
    return corners, IDs

def check_boardinfo_valid(boardinfo):
    for board_idx in boardinfo.keys():
        assert "dict_type" in boardinfo[board_idx].keys(), f"dict_type not found in boardinfo for board {board_idx}"
        assert "numX" in boardinfo[board_idx].keys(), f"numX not found in boardinfo for board {board_idx}"
        assert "numY" in boardinfo[board_idx].keys(), f"numY not found in boardinfo for board {board_idx}"
        assert "checkerLength" in boardinfo[board_idx].keys(), f"checkerLength not found in boardinfo for board {board_idx}"
        assert "markerLength" in boardinfo[board_idx].keys(), f"markerLength not found in boardinfo for board {board_idx}"
        assert "markerIDs" in boardinfo[board_idx].keys(), f"markerIDs not found in boardinfo for board {board_idx}"

def merge_charuco_detection(detection_list, boardinfo):
    check_boardinfo_valid(boardinfo)

    marker_id_offset = {}
    offset_sum = 0

    for board_idx in boardinfo.keys():
        cb = boardinfo[board_idx]
        
        marker_id_offset[int(board_idx)] = offset_sum
        offset_sum += (cb["numX"] - 1) * (cb["numY"] - 1)
    
    detected_corners, detected_ids = [], []
    detected_markers, detected_mids = [], []

    for board_id in detection_list.keys():
        offset = marker_id_offset[board_id]
        checkerCorner = detection_list[board_id]["checkerCorner"]
        checkerIDs = detection_list[board_id]["checkerIDs"]
        # markerCorner = detection_list[board_id]["markerCorner"]
        # markerIDs = detection_list[board_id]["markerIDs"]

        detected_corners.append(checkerCorner)
        detected_ids.append(checkerIDs + offset)

        # for val in markerCorner:
        #     detected_markers.append(val)
        # detected_mids.append(markerIDs)

    if len(detected_corners) > 0:
        detected_corners = np.concatenate(detected_corners, axis=0)
        detected_ids = np.concatenate(detected_ids, axis=0)
    
    else:
        detected_corners = np.array([])
        detected_ids = np.array([])

    return {
        "checkerCorner": detected_corners,
        "checkerIDs": detected_ids,
    }

def detect_charuco(img, boardinfo):
    """
    Detects Charuco corners and ArUco markers in the given image using multiple boards.

    Args:
        img (np.ndarray): Input image containing the markers. Shape (H, W, 3) or (H, W).
        boardinfo (Dict[int, Dict]): Dictionary of board settings.
            Each board must contain: 'dict_type', 'numX', 'numY', 'checkerLength', 'markerLength', 'markerIDs'.

    Returns:
        Tuple[
            Tuple[np.ndarray, List[np.ndarray]],
            Tuple[np.ndarray, np.ndarray]
        ]:
            - detected_corners (np.ndarray): Detected Charuco corner positions. Shape (M, 2).
            - detected_markers (List[np.ndarray]): List of detected ArUco marker corners. Each element has shape (4, 2).
            - detected_ids (np.ndarray): Detected Charuco corner IDs. Shape (M, 1).
            - detected_mids (np.ndarray): Detected ArUco marker IDs. Shape (N, 1).
    """

    check_boardinfo_valid(boardinfo)
    detector_dict = {}
    detection_results = {}

    for board_idx in boardinfo.keys():
        cb = boardinfo[board_idx]
        board = aruco.CharucoBoard(
            (cb["numX"], cb["numY"]),
            cb["checkerLength"],
            cb["markerLength"],
            aruco_dict[cb['dict_type']],
            np.array(cb["markerIDs"])
        )
        charDet = aruco.CharucoDetector(board)

        detector_dict[int(board_idx)] = charDet
        
    for board_id in detector_dict.keys():
        charDet = detector_dict[board_id]
        checkerCorner, checkerIDs, markerCorner, markerIDs = charDet.detectBoard(img)
        
        if checkerIDs is None:
            continue
        
        detection_results[board_id] = {
            "checkerCorner": checkerCorner,
            "checkerIDs": checkerIDs,
            # "markerCorner": markerCorner,
            # "markerIDs": markerIDs
        }

    return detection_results

def draw_charuco(image, corners, color=(0, 255, 255), radius=4, thickness=2, ids=None):
    color = [int(x) for x in color]
    for i in range(len(corners)):
        corner = tuple(int(x) for x in corners[i])
        cv2.circle(image, corner, radius, color, thickness)
        if ids is not None:
            cv2.putText(image, str(int(ids[i])), (corner[0] + 5, corner[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)

def draw_aruco():
    pass

def triangulate_marker():
    pass