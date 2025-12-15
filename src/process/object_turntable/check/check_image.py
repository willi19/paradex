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

def process_task(args):
    obj_name, index = args
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable", obj_name, index)
    image_dir = os.path.join(root_dir, "selected")
    mask_dir = os.path.join(root_dir, "masked_images")
    
    if not os.path.exists(image_dir):
        return
    
    out_dir = os.path.join(root_dir, "charuco_3d")
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing {obj_name}/{index}", len(os.listdir(image_dir)))
    if len(os.listdir(image_dir)) != 24:
        return
        

if __name__ == "__main__":
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
    corrupted_files = []
    for obj_name in sorted(os.listdir(root_dir)):
        obj_path = os.path.join(root_dir, obj_name)
        for index in os.listdir(obj_path):
            image_path = os.path.join(root_dir, obj_name, index, "images")
            if not os.path.exists(image_path):
                continue
            for serial in os.listdir(image_path):
                serial_path = os.path.join(image_path, serial)
                img_files = os.listdir(serial_path)
                for img_file in tqdm.tqdm(img_files, desc=f"Checking {obj_name}/{index}/{serial}"):
                    img = cv2.imread(os.path.join(serial_path, img_file))
                    if img is None:
                        print(f"❌ CV2 Error: {os.path.join(serial_path, img_file)}")
                        corrupted_files.append(os.path.join(serial_path, img_file))
    print("Corrupted files:")
    for f in corrupted_files:
        print(f)