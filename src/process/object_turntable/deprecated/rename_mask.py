import os
import cv2
import numpy as np
import tqdm
import multiprocessing as mp
import tqdm

from paradex.image.undistort import undistort_img
from paradex.utils.path import shared_dir, home_path
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, merge_charuco_detection, detect_charuco
from paradex.transforms.conversion import SOLVE_XA_B


if __name__ == "__main__":
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
    for obj_name in sorted(os.listdir(root_dir)):
        obj_path = os.path.join(root_dir, obj_name)
        for index in os.listdir(obj_path):
            mask_path = os.path.join(root_dir, obj_name, index, "masks")
            
            for serial_num in tqdm.tqdm(os.listdir(mask_path)):
                serial_path = os.path.join(mask_path, serial_num)
                mask_files = os.listdir(serial_path)
                for mask_file in mask_files:
                    # mask : frame_{idx}.png -> frame_{idx:06d}.png
                    idx = int(mask_file.split("frame_")[-1].split(".png")[0])
                    new_mask_file = f"frame_{idx:06d}.png"
                    os.rename(os.path.join(serial_path, mask_file), os.path.join(serial_path, new_mask_file))