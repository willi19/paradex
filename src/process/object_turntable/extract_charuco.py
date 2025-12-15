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

def process_frame(args):
    idx, root_dir  = args
    if os.path.exists(os.path.join(root_dir, "charuco_3d", f"{idx:06d}_id.npy")) and \
              os.path.exists(os.path.join(root_dir, "charuco_3d", f"{idx:06d}_cor.npy")):
            return
        
    intrinsic, extrinsic = load_camparam(os.path.join(root_dir))
    all_serial_list = sorted(os.listdir(os.path.join(root_dir, "images")))
    serial_list = [s for s in all_serial_list if os.path.exists(os.path.join(root_dir, "images", s, f"frame_{idx:06d}.jpg"))]
    if len(serial_list) < 5:
        return
    
    imgs = {}
    for serial in serial_list:
        img = cv2.imread(os.path.join(root_dir, "images", serial, f"frame_{idx:06d}.jpg"))
        if img is None:
            print(f"  Warning: Image read error for {serial} frame {idx:06d} in {root_dir}, skipping.")
            return
        imgs[serial] = img

    intrinsic_partial = {s: intrinsic[s] for s in serial_list}
    extrinsic_partial = {s: extrinsic[s] for s in serial_list}

    img_dict = ImageDict(imgs, intrinsic_partial, extrinsic_partial)
    
    charuco_3d = img_dict.triangulate_charuco()
    if len(charuco_3d) == 0:
        print(f"  No charuco detected for idx {idx} in {obj_name}/{index}")
        return
    
    board_id = list(charuco_3d.keys())[0]
    
    np.save(os.path.join(root_dir, "charuco_3d", f"{idx:06d}_id.npy"), charuco_3d[board_id]['checkerIDs'])
    np.save(os.path.join(root_dir, "charuco_3d", f"{idx:06d}_cor.npy"), charuco_3d[board_id]['checkerCorner'])

def process_task(args):
    obj_name, index = args
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable", obj_name, index)
    image_dir = os.path.join(root_dir, "images")
    
    if not os.path.exists(image_dir):
        return
    
    out_dir = os.path.join(root_dir, "charuco_3d")
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing {obj_name}/{index}", len(os.listdir(image_dir)))
    if len(os.listdir(image_dir)) != 24:
        return
    
    serial_list = sorted(os.listdir(image_dir))    
    max_idx = -1
    for serial in serial_list:
        img_files = os.listdir(os.path.join(image_dir, serial))
        for img_file in img_files:
            if img_file.startswith("frame_") and img_file.endswith(".jpg"):
                idx = int(img_file[6:12])
                if idx > max_idx:
                    max_idx = idx
    
    task_list = []
    for idx in range(1, max_idx + 1):
        if os.path.exists(os.path.join(root_dir, "charuco_3d", f"{idx:06d}_id.npy")) and \
              os.path.exists(os.path.join(root_dir, "charuco_3d", f"{idx:06d}_cor.npy")):
            continue

        task_list.append((idx, root_dir))

    with mp.Pool(processes=mp.cpu_count()-10) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_frame, task_list), total=len(task_list)))

if __name__ == "__main__":
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")

    # Collect tasks
    tasks = []
    for obj_name in sorted(os.listdir(root_dir)):
        obj_path = os.path.join(root_dir, obj_name)
        if os.path.isdir(obj_path):
            for index in os.listdir(obj_path):
                tasks.append((obj_name, index))
    
    print(f"Processing {len(tasks)} sequences with {mp.cpu_count()-10} workers")
    
    # Parallel
    for task in tqdm.tqdm(tasks):
        process_task(task)