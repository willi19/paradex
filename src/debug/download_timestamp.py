import os
import shutil
import argparse
import time
from tqdm import tqdm
from paradex.utils.upload_file import get_total_size, copy_to_nfs
from paradex.utils.io import shared_dir, download_dir

home_dir = os.path.expanduser("~")
capture_path_list = [os.path.join(home_dir, f"captures{i}") for i in range(1,3)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively copy files and directories, skipping identical files but overwriting corrupt ones.")
    parser.add_argument("--name", required=True, help="Path to the source file or directory")
    
    args = parser.parse_args()

    source_dir = os.path.join(shared_dir, args.name)
    obj_list = os.listdir(source_dir)
    for obj_name in obj_list:
        index_list = os.listdir(os.path.join(source_dir, obj_name))
        for index_name in index_list:
            source_path = os.path.join(source_dir, obj_name, index_name)
            destination_path = os.path.join(download_dir, args.name, obj_name, index_name,"videos")

            total_size = get_total_size(source_path, destination_path)
            print(source_path, destination_path)
            copy_to_nfs(source_path, destination_path, total_size, end_with=".json")

    print("Copy completed.")
        
