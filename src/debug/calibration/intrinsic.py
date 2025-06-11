import numpy as np
import cv2
import json
import os
import argparse
from paradex.utils.file_io import config_dir, shared_dir, find_latest_directory

# --- Load corner data and board info ---
parser = argparse.ArgumentParser(description="Compute intrinsic matrix from ChArUco corners.")
parser.add_argument(
    "--serial",
    type=str,
    required=True,
    help="Camera serial number.",
)
parser.add_argument(
    "--date",
    type=str,
    required=False,
    help="Path to the saved npy file with board corners.",
)

args = parser.parse_args()

num_trial = 20
sample_size = 100

serial_num = args.serial
root_path = os.path.join(shared_dir, "intrinsic", serial_num)
kypt_path = os.path.join(root_path, "keypoint")

date = args.date

if date is None:
    date = find_latest_directory(kypt_path)

kypt = np.load(os.path.join(kypt_path, date), allow_pickle=True)

chessboard_size = (10, 7)  # number of inner corners
square_size = 0.025  # in meters or whatever unit you choose

objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

tot_objp = np.array([objp] * len(kypt))
tot_imgp = np.array([corners.reshape(-1, 2) for corners in kypt])

dist_list = []
K_list = []

for _ in range(num_trial):
    # --- Perform camera calibration ---
    image_size = (2048, 1536)  
    index = np.random.choice(len(tot_objp), sample_size, replace=False)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        tot_objp[index], tot_imgp[index], image_size, None, None
    )
    dist_list.append(dist.flatten())
    K_list.append(K)

print(np.mean(dist_list, axis=0))
print(np.std(dist_list, axis=0))

print(np.mean(K_list, axis=0))
print(np.std(K_list, axis=0))
