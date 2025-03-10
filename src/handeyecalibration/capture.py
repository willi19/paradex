import argparse
from paradex.capture.image import capture_images_from_all_cameras
import os
from paradex.process.marker_detector import detect_aruco
import numpy as np

calib_path = os.path.join(os.path.expanduser("~"), "shared_data", "handeye_calibration")
def find_latest_directory():
    dirs = [d for d in os.listdir(calib_path)] 
       
    if not dirs:
        print("No valid directories found.")
        return
        
    # Sort directories based on name (assuming names are time-based)
    latest_dir = max(dirs, key=str)
    index_list = [int(d) for d in os.listdir(os.path.join(calib_path, latest_dir))]

    return latest_dir, max(index_list)+1 if index_list else 0

if __name__ == "__main__":
    camera_config = "config/lens.json"
    lens_info = "config/camera.json"

    name, index = find_latest_directory()

    home_dir = os.path.expanduser("~")
    save_path = os.path.join(home_dir, "shared_data", "handeye_calibration", name, str(index))
    os.makedirs(os.path.join(save_path,"image"), exist_ok=True)

    images = capture_images_from_all_cameras(os.path.join(save_path,"image"), lens_info, camera_config)

    os.makedirs(os.path.join(save_path,"marker"), exist_ok=True)

    for serialnum, image in images.items():
        corners, ids = detect_aruco(image)
        id_dict = {}
        if ids is not None:
            id_dict = {i[0]: c for i, c in zip(ids, corners)}
        np.save(os.path.join(save_path, "marker", f"{serialnum}"), id_dict)