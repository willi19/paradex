import os
import cv2
import numpy as np
import tqdm

from paradex.image.image_dict import ImageDict
from paradex.utils.path import home_path
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, merge_charuco_detection, get_board_cor
from paradex.transforms.conversion import SOLVE_XA_B

def compute_charuco_board_center(charuco_3d, board_3d):
    """
    Compute board center using detected corners with board definition.
    
    Args:
        charuco_3d: dict with 'checkerIDs' and 'checkerCorner'
        board_info: dict with board parameters
            - 'squares_x': number of squares in x (e.g., 7)
            - 'squares_y': number of squares in y (e.g., 5)
            - 'square_length': size of each square in meters
            - 'marker_length': size of ArUco markers
    
    Returns:
        board_pose: 4x4 transformation matrix
    """
    ids = charuco_3d['checkerIDs'].flatten()
    corners_3d = charuco_3d['checkerCorner']
    
    bids = board_3d['checkerIDs'].flatten()
    bcorners = board_3d['checkerCorner']
    
    ids_comm, bids_comm = find_common_indices(ids, bids)
    
    obj_points = bcorners[bids_comm]
    img_points = corners_3d[ids_comm]
    
    # Solve for transformation: board_frame -> world_frame
    board_pose = SOLVE_XA_B(obj_points, img_points)
    board_center = np.mean(obj_points, axis=0) + np.array([0,0,-0.02])  # slightly above the board
    
    board_center_hom = np.hstack([board_center, 1.0])
    board_center_world = board_pose @ board_center_hom
    return board_center_world[:3]


root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
obj_list = ['pepper_tuna']
board_cor_dict = get_board_cor()

for obj_name in obj_list:
    index_list = os.listdir(os.path.join(root_dir, obj_name))
    for index in index_list:
        demo_path = os.path.join("capture/object_turntable", obj_name, index)
        image_dir = os.path.join(home_path, "paradex_download", demo_path, "images")
        if not os.path.exists(image_dir):
            continue
        out_dir = os.path.join(home_path, "paradex_download", demo_path, "object_center")
        os.makedirs(out_dir, exist_ok=True)

        intrinsic, extrinsic = load_camparam(os.path.join(home_path, "paradex_download", demo_path))

        serial_list = os.listdir(image_dir)
        max_idx = max([len(os.listdir(os.path.join(image_dir, serial))) for serial in serial_list])
        
        for idx in tqdm.tqdm(range(1, max_idx + 1)):
            imgs = {}
            for serial in serial_list:
                img_path = os.path.join(image_dir, serial, f"{idx:04d}.jpg")
                if os.path.exists(img_path):
                    imgs[serial] = cv2.imread(img_path)
            img_dict = ImageDict(imgs, intrinsic, extrinsic)
            charuco_3d = img_dict.triangulate_charuco()
            board_id = list(charuco_3d.keys())[0]
            
            board_cor = board_cor_dict[board_id]
            board_center_world = compute_charuco_board_center(charuco_3d[board_id], board_cor)
            
            proj_center = img_dict.project_pointcloud(board_center_world)
            for serial in proj_center:
                os.makedirs(os.path.join(out_dir, f"{serial}"), exist_ok=True)
                np.save(os.path.join(out_dir, f"{serial}", f"{idx:04d}.npy"), proj_center[serial])