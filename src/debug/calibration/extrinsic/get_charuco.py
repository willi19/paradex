import argparse
import os
import cv2
from paradex.utils.file_io import find_latest_directory, home_path, config_dir, shared_dir
import numpy as np
from paradex.geometry.triangulate import triangulate
from paradex.image.merge import merge_image
from paradex.image.aruco import draw_keypoint, detect_charuco, merge_charuco_detection
import json

board_info = json.load(open(os.path.join(config_dir, "environment", "charuco_info.json"), "r"))

extrinsic_dir = os.path.join(shared_dir,"extrinsic")
cam_param_dir = os.path.join(shared_dir, "cam_param")
dir = "config"

if __name__ == "__main__":
    img_dir = os.path.join(shared_dir, "extrinsic_test")
    for index in os.listdir(img_dir):
        img_list = os.listdir(os.path.join(img_dir, index, "image"))
        for img_name in img_list:
            img = cv2.imread(os.path.join(img_dir, index, "image", img_name))
            
            serial_num = img_name.split(".")[0]
            results = detect_charuco(img, board_info)
            result = merge_charuco_detection(results, board_info)
            
            kypts = result["checkerCorner"]
            ids = result["checkerIDs"]
            
            os.makedirs(os.path.join(img_dir, index, "keypoint"), exist_ok=True)
            np.save(os.path.join(img_dir, index, "keypoint", f"{serial_num}_cor.npy"), kypts)
            np.save(os.path.join(img_dir, index, "keypoint", f"{serial_num}_id.npy"), ids)
            
