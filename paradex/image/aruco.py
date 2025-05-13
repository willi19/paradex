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

    for board_idx in boardinfo.keys():
        assert "dict_type" in boardinfo[board_idx].keys(), f"dict_type not found in boardinfo for board {board_idx}"
        assert "numX" in boardinfo[board_idx].keys(), f"numX not found in boardinfo for board {board_idx}"
        assert "numY" in boardinfo[board_idx].keys(), f"numY not found in boardinfo for board {board_idx}"
        assert "checkerLength" in boardinfo[board_idx].keys(), f"checkerLength not found in boardinfo for board {board_idx}"
        assert "markerLength" in boardinfo[board_idx].keys(), f"markerLength not found in boardinfo for board {board_idx}"
        assert "markerIDs" in boardinfo[board_idx].keys(), f"markerIDs not found in boardinfo for board {board_idx}"

    all_boards = {}
    marker_id_offset = 0

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

        all_boards[int(board_idx)] = {
            "marker_id_offset": marker_id_offset,
            "detector": charDet
        }
        marker_id_offset += (cb["numX"] - 1) * (cb["numY"] - 1)

    detected_corners, detected_ids = [], []
    detected_markers, detected_mids = [], []

    for board_id in all_boards.keys():
        offset = all_boards[board_id]["marker_id_offset"]
        charDet = all_boards[board_id]["detector"]

        charCorner, charIDs, markerCorner, markerIDs = charDet.detectBoard(img)

        if charIDs is not None:
            detected_corners.append(charCorner)
            detected_ids.append(charIDs + offset)

            for val in markerCorner:
                detected_markers.append(val)
            detected_mids.append(markerIDs)

    if len(detected_corners) > 0:
        detected_corners = np.concatenate(detected_corners, axis=0)
        detected_ids = np.concatenate(detected_ids, axis=0)
        detected_mids = np.concatenate(detected_mids, axis=0)
    
    else:
        detected_corners = np.array([])
        detected_ids = np.array([])
        detected_mids = np.array([])

    return (detected_corners, detected_markers), (detected_ids, detected_mids)
