import argparse
from paradex.capture.image import capture_images_from_all_cameras
from pathlib import Path
import os
from paradex.utils.io import calib_path_list
import datetime

def find_latest_directory(is_init):
    if is_init:
        now = datetime.datetime.now()
        dir_name = now.strftime("%Y%m%d%H%M%S")
        
        return dir_name, 0
    else:
        # Get list of directories in the current working directory
        dirs = [d for d in os.listdir(calib_path_list[0])] 
        print(os.listdir(calib_path_list[0]), dirs)   
        if not dirs:
            print("No valid directories found.")
            return
        
        # Sort directories based on name (assuming names are time-based)
        latest_dir = max(dirs, key=str)
        index_list = [int(d) for d in os.listdir(os.path.join(calib_path_list[0], latest_dir))]

        return latest_dir, max(index_list)+1 if index_list else 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a single image from multiple cameras.")
    parser.add_argument("--init", action="store_true", help="Initialize a new directory with current timestamp.")

    args = parser.parse_args()
    is_init = args.init
    name, index = find_latest_directory(is_init)

    camera_config = "config/lens.json"
    lens_info = "config/camera.json"

    
    os.makedirs(os.path.join(calib_path_list[0], str(name), str(index), "images","00001"), exist_ok=True)
    

    save_path = f"{calib_path_list[0]}/{name}/{index}/images/00001"
    os.makedirs(save_path, exist_ok=True)
#     save_path.mkdir(parents=True, exist_ok=True)

    capture_images_from_all_cameras(save_path, lens_info, camera_config)
