import os
import cv2
import numpy as np
import tqdm
import multiprocessing as mp
import shutil

from paradex.image.image_dict import ImageDict
from paradex.utils.path import shared_dir, home_path
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, get_board_cor
from paradex.transforms.conversion import SOLVE_XA_B

def process(demo_path):
    """Process one sequence with parallel frame processing."""
    
    out_dir = os.path.join(demo_path, "rotation")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    os.makedirs(out_dir, exist_ok=True)
    
    charuco_dir = os.path.join(demo_path, "charuco_3d")
    
    base_charuco = get_board_cor()['2']
    base_id = base_charuco['checkerIDs']
    base_cor = base_charuco['checkerCorner']
    
    for charuco_file in tqdm.tqdm(sorted(os.listdir(charuco_dir))):
        if not charuco_file.endswith("_cor.npy"):
            continue
        frame_idx = charuco_file.split("_")[0]
        
        charuco_cor = np.load(os.path.join(charuco_dir, f"{frame_idx}_cor.npy"))
        charuco_id = np.load(os.path.join(charuco_dir, f"{frame_idx}_id.npy"))
        
        common_idx = find_common_indices(base_id, charuco_id)
        if common_idx[0] is None:
            print(f"  Warning: No common points in frame {frame_idx}, skipping.")
            continue
            
        if len(common_idx[0]) < 4:
            print(f"  Warning: Not enough common points in frame {frame_idx}, skipping.")
            continue
        
        base_pts = base_cor[common_idx[0]]
        cur_pts = charuco_cor[common_idx[1]]
        T = SOLVE_XA_B(base_pts, cur_pts) # board in world

        np.save(os.path.join(out_dir, f"{frame_idx}.npy"), T)
    
    return demo_path

def main():
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
    obj_list = sorted(os.listdir(root_dir))
    
    for obj_name in obj_list:
        obj_path = os.path.join(root_dir, obj_name)
        if not os.path.isdir(obj_path):
            continue
        
        index_list = sorted(os.listdir(obj_path))
        for index in index_list:
            process(os.path.join(root_dir, obj_name, index))
            
if __name__ == "__main__":
    main()