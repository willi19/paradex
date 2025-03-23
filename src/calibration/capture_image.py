import argparse
import os
import datetime
import paradex
from paradex.camera.camera_loader import CameraManager
from paradex.utils.io import home_dir, calib_path_list

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

def main():
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--init", action="store_true", help="Initialize a new directory with current timestamp.")
    
    args = parser.parse_args()
    is_init = args.init
    name, index = find_latest_directory(is_init)

    for calib_path in calib_path_list:
        os.makedirs(os.path.join(calib_path, str(name), str(index), "images","00001"), exist_ok=True)
        
    dir_name = f"calibration/{name}/{index}/images/00001"
    manager = CameraManager(num_cameras=4, name=dir_name, is_streaming=False)
    manager.start()

if __name__ == "__main__":
    main()
