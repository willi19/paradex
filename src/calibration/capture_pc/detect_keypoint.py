import os
import argparse
from tqdm import tqdm
from paradex.utils.marker import detect_charuco
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from paradex.utils.io import find_latest_directory, home_dir, calib_path_list

def process_image(img_dir):
    """
    Process a single image: detect keypoints and save results.
    - Loads an image, detects Charuco keypoints, and saves keypoint data.
    """
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            tqdm.write(f"❌ Failed to load {img_path}")
            return None

        # Define keypoint directory
        kypt_dir = os.path.dirname(img_path.replace("images", "keypoints"))
        os.makedirs(kypt_dir, exist_ok=True)

        # Define keypoint filenames
        kypt_file_name = os.path.basename(img_path).split(".")[0]
        kypt_path = os.path.join(kypt_dir, f"{kypt_file_name}_ids.npy")

        # # Skip if already processed
        # if os.path.exists(kypt_path):
        #     continue

        # Detect charuco markers
        (detected_corners, detected_markers), (detected_ids, detected_mids) = detect_charuco(img)

        if len(detected_corners) == 0:
            detected_ids = np.array([])
            detected_corners = np.array([])
        
        else:
            detected_corners = detected_corners[:,0,:]

        # Save keypoints
        np.save(os.path.join(kypt_dir, f"{kypt_file_name}_ids.npy"), detected_ids)
        np.save(os.path.join(kypt_dir, f"{kypt_file_name}_corners.npy"), detected_corners)

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--name", type=str, help="Name of the directory to detect keypoint.")
    parser.add_argument("--latest", action="store_true", help="Split the latest video files.")

    args = parser.parse_args()

    if not args.latest and not args.name:
        print("Please specify either --latest or --name.")
        exit()
    
    if args.latest:
        name = find_latest_directory(calib_path_list[0])
    else:
        name = args.name

    image_dir_list = []
    for calib_path in calib_path_list:
        if not os.path.exists(os.path.join(calib_path, name)):
            continue
        index_list = os.listdir(os.path.join(calib_path, name))
        for index in index_list:
            image_dir = os.path.join(calib_path, name, index, "images")
            if not os.path.exists(image_dir):
                continue
            image_dir_list += [os.path.join(image_dir, d) for d in os.listdir(image_dir)]
    image_dir_list = sorted(image_dir_list, key=lambda x: (int(x.split("/")[-3]), int(x.split("/")[-1])))

    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=len(image_dir_list), desc="Processing Images", unit="img_dir") as pbar:
            for _ in pool.imap_unordered(process_image, image_dir_list):
                pbar.update(1)  # Update progress bar