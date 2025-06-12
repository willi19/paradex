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
    required=False,
    help="Camera serial number.",
)
parser.add_argument(
    "--date",
    type=str,
    required=False,
    help="Path to the saved npy file with board corners.",
)

args = parser.parse_args()

serial_list = []
if args.serial is None:
    serial_list = os.listdir(os.path.join(shared_dir, "intrinsic"))

else:
    serial_list.append(args.serial)

for serial_num in serial_list:
    print(f"Processing serial number: {serial_num}")
    root_path = os.path.join(shared_dir, "intrinsic", serial_num)
    kypt_path = os.path.join(root_path, "keypoint")
    
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(os.path.join(root_path, "param")):
        os.makedirs(os.path.join(root_path, "param")) 
        
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
    # --- Perform camera calibration ---
    image_size = (2048, 1536)  
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        tot_objp, tot_imgp, image_size, None, None
    )
    print(f"Calibration RMS error: {ret}")
    print(f"Intrinsic matrix K:\n{K}")
    print(f"Distortion coefficients:\n{dist.flatten()}")

    intrinsics_data = {
        "RMS_error": ret,
        "K": K.tolist(),
        "distortion": dist.flatten().tolist(),
        "width": image_size[0],
        "height": image_size[1]
    }

    file_name = date.split(".")[0]

    with open(f"{root_path}/param/{file_name}.json", "w") as f:
        json.dump(intrinsics_data, f, indent=4)