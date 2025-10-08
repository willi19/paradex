"""
ArUco and CharUco marker detection module.

This module provides functions for detecting and tracking ArUco and CharUco markers
in images from multi-camera systems. Supports multiple dictionary types and 
multi-board CharUco detection.
"""

from cv2 import aruco
from typing import Tuple, List, Dict
import numpy as np
import cv2

from paradex.image.undistort import undistort_img
from paradex.geometry.triangulate import ransac_triangulation, triangulate
from paradex.image.projection import get_cammtx

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
    Detect ArUco markers in an image.
    
    Args:
        img (np.ndarray): Input image in BGR format. Shape (H, W, 3) or (H, W).
        dict_type (str, optional): ArUco dictionary type. 
            Supported: '4X4_50' to '7X7_1000'. Defaults to '6X6_1000'.
    
    Returns:
        Tuple[List[np.ndarray], np.ndarray]: 
            - corners: List of detected marker corners. Each element is (4, 2) float32 array.
            - IDs: Detected marker IDs. Shape (N, 1) int32 array. None if no markers detected.
    
    Example:
        >>> corners, ids = detect_aruco(image, dict_type='6X6_1000')
        >>> print(f"Detected {len(corners)} markers")
    """
    corners, IDs, _ = arucoDetector_dict[dict_type].detectMarkers(img)
    return corners, IDs

def check_boardinfo_valid(boardinfo):
    """
    Validate CharUco board information dictionary.
    
    Args:
        boardinfo (Dict[int, Dict]): Board configuration dictionary.
    
    Raises:
        AssertionError: If required fields are missing.
    
    Note:
        Required fields for each board: 'dict_type', 'numX', 'numY', 
        'checkerLength', 'markerLength', 'markerIDs'.
    """
    for board_idx in boardinfo.keys():
        assert "dict_type" in boardinfo[board_idx].keys(), f"dict_type not found in boardinfo for board {board_idx}"
        assert "numX" in boardinfo[board_idx].keys(), f"numX not found in boardinfo for board {board_idx}"
        assert "numY" in boardinfo[board_idx].keys(), f"numY not found in boardinfo for board {board_idx}"
        assert "checkerLength" in boardinfo[board_idx].keys(), f"checkerLength not found in boardinfo for board {board_idx}"
        assert "markerLength" in boardinfo[board_idx].keys(), f"markerLength not found in boardinfo for board {board_idx}"
        assert "markerIDs" in boardinfo[board_idx].keys(), f"markerIDs not found in boardinfo for board {board_idx}"

def merge_charuco_detection(detection_list, boardinfo):
    """
    Merge detection results from multiple CharUco boards into a single coordinate system.
    
    Args:
        detection_list (Dict[int, Dict]): Detection results per board.
            Each entry contains 'checkerCorner' and 'checkerIDs'.
        boardinfo (Dict[int, Dict]): Board configuration dictionary.
    
    Returns:
        Dict[str, np.ndarray]: Merged detection results containing:
            - 'checkerCorner': All detected corners. Shape (M, 2).
            - 'checkerIDs': Corresponding corner IDs with offsets applied. Shape (M, 1).
    
    Note:
        Corner IDs are offset based on board index to avoid ID collisions.
        Offset for each board = (numX-1) * (numY-1) * board_index.
    """
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

        detected_corners.append(checkerCorner)
        detected_ids.append(checkerIDs + offset)

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
    Detect CharUco board corners and ArUco markers from multiple boards in an image.
    
    Args:
        img (np.ndarray): Input image containing CharUco boards. Shape (H, W, 3) or (H, W).
        boardinfo (Dict[int, Dict]): Dictionary of board configurations.
            Each board must contain:
                - 'dict_type' (str): ArUco dictionary type (e.g., '6X6_1000')
                - 'numX' (int): Number of squares in X direction
                - 'numY' (int): Number of squares in Y direction
                - 'checkerLength' (float): Checker square side length (meters)
                - 'markerLength' (float): ArUco marker side length (meters)
                - 'markerIDs' (List[int]): List of marker IDs used in the board
    
    Returns:
        Dict[int, Dict]: Detection results per board ID containing:
            - 'checkerCorner' (np.ndarray): Detected corner positions. Shape (M, 1, 2).
            - 'checkerIDs' (np.ndarray): Detected corner IDs. Shape (M, 1).
    
    Example:
        >>> boardinfo = {
        ...     0: {
        ...         'dict_type': '6X6_1000',
        ...         'numX': 10,
        ...         'numY': 7,
        ...         'checkerLength': 0.025,
        ...         'markerLength': 0.0185,
        ...         'markerIDs': list(range(35))
        ...     }
        ... }
        >>> detections = detect_charuco(image, boardinfo)
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
        }

    return detection_results

def draw_charuco(image, corners, color=(0, 255, 255), radius=4, thickness=2, ids=None):
    """
    Draw detected CharUco corners on an image.
    
    Args:
        image (np.ndarray): Image to draw on. Modified in-place.
        corners (np.ndarray): Corner positions to draw. Shape (M, 2).
        color (Tuple[int, int, int], optional): BGR color. Defaults to (0, 255, 255) cyan.
        radius (int, optional): Circle radius in pixels. Defaults to 4.
        thickness (int, optional): Circle line thickness. Defaults to 2.
        ids (np.ndarray, optional): Corner IDs to display as text. Shape (M,). Defaults to None.
    
    Note:
        Image is modified in-place. No return value.
    """
    color = [int(x) for x in color]
    for i in range(len(corners)):
        corner = tuple(int(x) for x in corners[i])
        cv2.circle(image, corner, radius, color, thickness)
        if ids is not None:
            cv2.putText(image, str(int(ids[i])), (corner[0] + 5, corner[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)

def draw_aruco(img, kypt, ids=None, color=(255, 0, 0)):
    """
    Draw detected ArUco markers on an image with corner indices.
    
    Args:
        img (np.ndarray): Image to draw on.
        kypt (List[np.ndarray]): List of marker corners. Each element shape (4, 2).
        ids (np.ndarray, optional): Marker IDs to display. Shape (N,). Defaults to None.
        color (Tuple[int, int, int], optional): BGR color for corners. Defaults to (255, 0, 0) blue.
    
    Returns:
        np.ndarray: Image with drawn markers.
    
    Note:
        Each corner is numbered 0-3 and drawn as a colored circle.
        Marker ID is displayed at the center if provided.
    """
    for idx, corner in enumerate(kypt):
        corner = corner.squeeze().astype(int)
        if ids is not None:
            cv2.putText(img, str(ids[idx]), tuple(np.mean(corner, axis=0).astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i in range(4):
            cv2.circle(img, tuple(corner[i]), 5, color, -1)
            cv2.putText(img, str(i), (corner[i][0] + 10, corner[i][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def draw_keypoint(img, kypt, color=(255, 0, 0)):
    """
    Draw keypoints as small circles on an image.
    
    Args:
        img (np.ndarray): Image to draw on. Modified in-place.
        kypt (np.ndarray): Keypoint coordinates. Shape (2,) or (N, 2).
        color (Tuple[int, int, int], optional): BGR color. Defaults to (255, 0, 0) blue.
    
    Returns:
        np.ndarray: Image with drawn keypoints.
    
    Note:
        Single point or multiple points can be provided.
    """
    kypt = kypt.astype(int)
    
    if len(kypt.shape) == 1:
        return cv2.circle(img, tuple(kypt), 1, color, -1)
    else:
        for corner in kypt:
            cv2.circle(img, tuple(corner), 1, color, -1)
    return img

def triangulate_marker(img_dict, intrinsic, extrinsic, dict_type='6X6_1000'):
    """
    Triangulate 3D positions of ArUco markers from multiple camera views.
    
    Args:
        img_dict (Dict[str, np.ndarray]): Dictionary mapping camera serial numbers to images.
        intrinsic (Dict[str, Dict]): Camera intrinsic parameters for each serial number.
            Each entry must contain 'original_intrinsics', 'dist_params', 'intrinsics_undistort'.
        extrinsic (Dict[str, np.ndarray]): Camera extrinsic parameters (3x4 or 4x4 matrices).
        dict_type (str, optional): ArUco dictionary type. Defaults to '6X6_1000'.
    
    Returns:
        Dict[int, np.ndarray]: Dictionary mapping marker IDs to 3D positions.
            Each position is shape (1, 3) in world coordinates.
    
    Example:
        >>> img_dict = {'cam_01': img1, 'cam_02': img2}
        >>> marker_3d = triangulate_marker(img_dict, intrinsics, extrinsics)
        >>> print(f"Marker 5 position: {marker_3d[5]}")
    
    Note:
        - Images are automatically undistorted before detection
        - Uses RANSAC triangulation for robustness
        - Markers must be visible in at least 2 cameras
    """
    cammat = get_cammtx(intrinsic, extrinsic)
    
    id_cor = {}
    for serial_num, img in img_dict.items():
        if serial_num not in cammat:
            continue
        
        undist_img = undistort_img(img.copy(), intrinsic[serial_num])
        undist_kypt, ids = detect_aruco(undist_img, dict_type)
        
        if ids is None:
            continue
        
        ids = ids.reshape(-1)
        for id, k in zip(ids, undist_kypt):
            k = k.squeeze()
            
            if id not in id_cor:
                id_cor[id] = {"2d": [], "cammtx": []}
            id_cor[id]["2d"].append(k)
            id_cor[id]["cammtx"].append(cammat[serial_num])
            
    cor_3d = {id: ransac_triangulation(np.array(id_cor[id]["2d"]), np.array(id_cor[id]["cammtx"])) 
              for id in id_cor}
    return cor_3d