import json

import cv2
import numpy as np
from cv2 import aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

all_boards = [] # to be used in detection
#for j in range(1, args.board_num+1): # read all board info

with open("config/charuco_info.json", "r") as f:
    boardinfo = json.load(f)

board_num = [1,2,3,4]
for board_cnt, idx in enumerate(board_num):
    cb = boardinfo[str(idx)] # idx : board index
    board = aruco.CharucoBoard( (cb["numX"], cb["numY"]), cb["checkerLength"], cb["markerLength"], aruco_dict, np.array(cb["markerIDs"]))
    all_boards.append((board, int(cb["numMarker"]), board_cnt)) # j for offset ID

corners3d = all_boards[0][0].getChessboardCorners()
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

def detect_aruco(img):
    global arucoDetector_tuned
    corners, IDs, _ = arucoDetector.detectMarkers(img)
    return corners, IDs

# Detect charuco board
def detect_charuco(img):
    global all_boards
    detected_corners, detected_ids = [], []
    detected_markers, detected_mids = [], []
    for b in all_boards:
        cur_board, cur_board_id = b[0], b[2]
        charDet = aruco.CharucoDetector(cur_board)
        charCorner, charIDs, markerCorner, markerIDs = charDet.detectBoard(img)
        if charIDs is not None:
            aruco.drawDetectedCornersCharuco(img, charCorner, charIDs+70*cur_board_id)
            detected_corners.append(charCorner)
            detected_ids.append(charIDs+70*cur_board_id)
            for val in markerCorner:
                detected_markers.append(val) # tuple
            detected_mids.append(markerIDs)

    if len(detected_corners) > 0:
        detected_corners = np.concatenate(detected_corners, axis=0)
        detected_ids = np.concatenate(detected_ids, axis=0)
        #detected_markers = np.concatenate(detected_markers, axis=0)
        detected_mids = np.concatenate(detected_mids, axis=0)
    return (detected_corners, detected_markers), (detected_ids, detected_mids)

def triangulate(corners: np.ndarray, projections: np.ndarray):
    """
    corners : Nx4x2 matrix (N cameras, 4 keypoints, 2D image coordinates)
    projections : Nx3x4 matrix (N cameras)

    Returns:
        kp3d : (4, 3) matrix with triangulated 3D points for each corner
    """
    numImg = projections.shape[0]
    if numImg < 2:
        return None
    
    numPts = corners.shape[1]

    curX = corners[:, :, None, 0]
    curY = corners[:, :, None, 1]
    A = np.zeros((numPts, numImg * 2, 4))    
    
    A = np.vstack([
        curY @ projections[:, 2:3, :] - projections[:, 1:2, :],
        curX @ projections[:, 2:3, :] - projections[:, 0:1, :]
    ])
    
    A = A.transpose(1, 0, 2)
    _, _, Vt = np.linalg.svd(A)

    X = Vt[:, -1]  # Last row of V (smallest singular value)
    return (X[:, :3] / X[:, 3:4])  # Normalize by X[:,3]


def ransac_triangulation(corners: np.ndarray, projections: np.ndarray, threshold=1.5, iterations=100):
    """
    RANSAC-based triangulation to filter out outliers.
    
    corners : Nx4x2 matrix (N cameras, 4 keypoints, 2D image coordinates)
    projections : Nx3x4 matrix (N cameras)
    threshold : Inlier threshold for reprojection error
    iterations : Number of RANSAC iterations
    
    Returns:
        best_kp3d : (4, 3) matrix with filtered 3D keypoints
    """
    best_inliers = 0
    best_kp3d = None
    
    numPts = corners.shape[1]
    numImg = projections.shape[0]
    if numImg < 2:
        return None
    for _ in range(iterations):
        # Randomly sample a subset of cameras
        sample_idx = np.random.choice(numImg, size=max(2, numImg // 2), replace=False)
        sampled_corners = corners[sample_idx]
        sampled_projections = projections[sample_idx]
        
        # Triangulate points
        kp3d = np.array(triangulate(sampled_corners, sampled_projections))
        
        # Reproject points to all cameras and compute errors

        kp3d_h = np.hstack((kp3d, np.ones((numPts, 1))))  # Convert to homogeneous coordinates
        reprojections = projections @ kp3d_h.T  # Shape: (N, 3, numPts)
        
        reprojections = reprojections[:, :2, :] / reprojections[:, 2:3, :]  # Normalize
        
        # Compute reprojection errors
        errors = np.linalg.norm(reprojections - corners.transpose(0, 2, 1), axis=1)
        inliers = np.sum(errors < threshold, axis=0)
        
        # Update best inlier count and result
        total_inliers = np.sum(inliers)
        if total_inliers > best_inliers:
            best_inliers = total_inliers
            best_kp3d = kp3d
    return best_kp3d
