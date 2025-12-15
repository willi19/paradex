import os
import cv2
import numpy as np
import tqdm
import multiprocessing as mp
from paradex.image.image_dict import ImageDict
from paradex.utils.path import shared_dir, home_path
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, merge_charuco_detection, detect_charuco
from paradex.transforms.conversion import SOLVE_XA_B

if __name__ == "__main__":
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")

    for obj_name in sorted(os.listdir(root_dir)):
        obj_path = os.path.join(root_dir, obj_name)
        for index in os.listdir(obj_path):
            if not os.path.exists(os.path.join(root_dir, obj_name, index, "colmap", "database.db")):
                print(f"COLMAP database missing for {obj_name}/{index}")
            