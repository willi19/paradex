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

    # Collect tasks
    tasks = []
    cnt = 0
    tot = 0
    vid_tot = 0
    vid_cnt = 0
    
    for obj_name in sorted(os.listdir(root_dir)):
        obj_path = os.path.join(root_dir, obj_name)
        for index in os.listdir(obj_path):
            vid_tot += 24
            tot += 1            
            image_path = os.path.join(root_dir, obj_name, index, "selected")
            if not os.path.exists(image_path):
                continue
            else:
                masked_image_path = os.path.join(root_dir, obj_name, index, "masked_images")
                if not os.path.exists(masked_image_path):
                    continue
                serial_list = os.listdir(image_path)
                succ_cnt = 0
                for serial_num in serial_list:
                    if not os.path.exists(os.path.join(masked_image_path, serial_num)):
                        print(f"Missing masked images for {obj_name}/{index}/{serial_num}, skip")
                        continue
                    real_selected_cnt = 0
                    for img_file in os.listdir(os.path.join(image_path, serial_num)):
                        idx = int(img_file.split("frame_")[-1].split(".jpg")[0])
                        if idx % 40 == 1:
                            real_selected_cnt += 1
                    if len(os.listdir(os.path.join(image_path, serial_num))) == real_selected_cnt:
                        vid_cnt += 1
                        succ_cnt += 1
                    else:
                        print(f"Mismatch in {obj_name}/{index}/{serial_num}: {real_selected_cnt} vs {len(os.listdir(os.path.join(masked_image_path, serial_num)))}")
                if succ_cnt == len(serial_list):
                    cnt += 1
    # print(f"Processing {cnt}/{tot} sequences with {mp.cpu_count()} workers")
    print(f"{vid_cnt}/{vid_tot} finished ")
    print(f"Total sequences: {tot}, valid sequences: {cnt}")