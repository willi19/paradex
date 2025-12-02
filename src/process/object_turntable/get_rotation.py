import os
import cv2
import numpy as np
import tqdm

from paradex.image.image_dict import ImageDict
from paradex.utils.path import shared_dir, home_path
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, merge_charuco_detection
from paradex.transforms.conversion import SOLVE_XA_B

root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
obj_list = ['pepper_tuna']

for obj_name in obj_list:
    index_list = os.listdir(os.path.join(root_dir, obj_name))
    for index in index_list:
        demo_path = os.path.join("capture/object_turntable", obj_name, index)
        image_dir = os.path.join(home_path, "paradex_download", demo_path, "images")
        if not os.path.exists(image_dir):
            continue
        intrinsic, extrinsic = load_camparam(os.path.join(shared_dir, demo_path))
        
        serial_list = os.listdir(image_dir)
        max_idx = max([len(os.listdir(os.path.join(image_dir, serial))) for serial in serial_list])
        marker6D = []
        last_charuco_3d = None

        for idx in tqdm.tqdm(range(1, max_idx + 1)):
            imgs = {}
            for serial in serial_list:
                img_path = os.path.join(image_dir, serial, f"frame_{idx:06d}.jpg")
                if os.path.exists(img_path):
                    imgs[serial] = cv2.imread(img_path)
            img_dict = ImageDict(imgs, intrinsic, extrinsic)
            charuco_3d = merge_charuco_detection(img_dict.triangulate_charuco())
            if idx == 1:
                last_charuco_3d = charuco_3d
                marker6D.append(np.eye(4))
            else:
                idx1, idx2 = find_common_indices(last_charuco_3d["checkerIDs"], charuco_3d['checkerIDs'])
                delta = SOLVE_XA_B(
                    charuco_3d["checkerCorner"][idx2],
                    last_charuco_3d["checkerCorner"][idx1],
                )
                marker6D.append(delta @ marker6D[-1])
                last_charuco_3d = charuco_3d
        np.save(os.path.join(shared_dir, demo_path, "rot.npy"), np.array(marker6D))