import os
import argparse
from paradex.utils.upload_file import get_total_size, copy_to_nfs
import numpy as np
from tqdm import tqdm
from paradex.utils.io import find_latest_directory

home_dir = os.path.expanduser("~")
calib_path_list = [os.path.join(home_dir, f"captures{i}", "calibration") for i in range(1,3)]


def load_keypoints(keypoint_dir):
    """
    Load keypoints from a directory.
    """
    keypoints = {}
    for file in os.listdir(keypoint_dir):
        if file.endswith("_ids.npy"):
            key = file.split("_")[0]
            keypoints[key] = np.load(os.path.join(keypoint_dir, file))
    return keypoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively copy files and directories, skipping identical files but overwriting corrupt ones.")
    parser.add_argument("--latest", action="store_true", help="Path to the source file or directory")
    parser.add_argument("--name", help="Path to the NFS-mounted directory")
    parser.add_argument("--image", action="store_true", help="Copy images also")
    parser.add_argument("--video", action="store_true", help="Copy videos also")
    
    args = parser.parse_args()

    if not args.latest and not args.name:
        print("Please specify either --latest or --name.")
        exit()
    
    if args.latest:
        name = find_latest_directory(calib_path_list[0])
    else:
        name = args.name

    dest_root_dir = os.path.join(home_dir, "shared_data", "calibration", name)
    
    kypt_dir_list = []
    for calib_path in calib_path_list:
        index_list = os.listdir(os.path.join(calib_path, name))
        for index in index_list:
            kypt_dir = os.path.join(calib_path, name, index, "keypoints")
            kypt_dir_list += [os.path.join(kypt_dir, d) for d in os.listdir(kypt_dir)]
    kypt_dir_list = sorted(kypt_dir_list, key=lambda x: (int(x.split("/")[-3]), int(x.split("/")[-1])))
    
    # pbar = tqdm(kypt_dir_list, desc="Copying keypoints", unit="directory")
    with tqdm(total=len(kypt_dir_list), desc="Copying keypoints", unit="directory") as pbar:
        for kypt_dir in kypt_dir_list:
            index = kypt_dir.split("/")[-3]
            timestamp = kypt_dir.split("/")[-1]
            
            dest_dir = os.path.join(dest_root_dir, index, "keypoints", timestamp)
            total_size = get_total_size(kypt_dir, dest_dir)
            copy_to_nfs(kypt_dir, dest_dir, total_size)
            pbar.update(1)

    # Todo : Image and video copying