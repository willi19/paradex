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
    #arucoDetector = aruco.ArucoDetector(aruco_dict)
    # corners, IDs, _ = arucoDetector.detectMarkers(img) # marker corners 의미?
    corners, IDs, _ = arucoDetector.detectMarkers(img)
    #corners_t, IDs_t, _ = arucoDetector_tuned.detectMarkers(img) # marker corners 의미?
    # Merge detected
    # return corners_t, IDs_t #corners, corners_t, IDs, IDs_t
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


def triangulate(corners:dict, projections:list):
    """
    N : number of images with same marker
    corners : {1: (N,2) array, 2:(N,2) array, 3:(N,2) array, 4: (N,2) array}
    projections : list of 3x4 matrices of length N
    """
    numImg = len(projections)
    kp3d = dict()
    #print("triangulate")
    for corner_id, kps in corners.items():
        A = [] 
        for i in range(numImg):
            curX, curY = kps[i,0], kps[i,1]
            cur_proj = projections[i]
            A.append(curY*cur_proj[2] - cur_proj[1])
            A.append(curX*cur_proj[2] - cur_proj[0])
        #print(A, numImg)
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        kp3d[corner_id] = V[3][0:3]/V[3][3] #
    return kp3d

def triangulate_table(corners, projections):
    """
    N : number of images with same marker
    corners : (N, 4, 2) array
    projections : (N, 3, 4) array
    """
    N = corners.shape[0]
    assert corners.shape == (N, 4, 2)
    assert projections.shape == (N, 3, 4)
    corner_3d = []
    for corner_id in range(4):
        A = [] 
        for i in range(N):
            curX, curY = corners[i,corner_id,0], corners[i,corner_id,1]
            cur_proj = projections[i]
            A.append(curY*cur_proj[2] - cur_proj[1])
            A.append(curX*cur_proj[2] - cur_proj[0])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        #print(S, U.shape, V.shape)
        kp3d = V[3][0:3]/V[3][3] #
        corner_3d.append(kp3d)
    return np.array(corner_3d) # (4, 3)

def detect_charuco_features(image_path):
    """
    Detects ChArUco markers and extracts features.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    
    if ids is None:
        return None, None  # No markers detected

    # Detect ChArUco corners
    charuco = aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco.detectBoard(gray)

    if charuco_corners is None or charuco_ids is None:
        return None, None  # No ChArUco corners detected

    return charuco_corners, charuco_ids

if __name__ == "__main__":

    im1 = cv2.imread("/home/capture2/Videos/22640993.png")
    res = detect_charuco(im1)
    print(res[1][0], res[1][0].shape, res[0][0], res[0][0].shape)