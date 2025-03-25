import os
import shutil
import argparse
import time
from tqdm import tqdm
from paradex.utils.upload_file import get_total_size, copy_to_nfs
from paradex.utils.io import capture_path_list

home_dir = os.path.expanduser("~")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively copy files and directories, skipping identical files but overwriting corrupt ones.")
    parser.add_argument("--name", required=True, help="Path to the source file or directory")
    
    args = parser.parse_args()
    destination_path = os.path.join(home_dir, "shared_data", args.name)

    for capture_path in capture_path_list:
        source_path = os.path.join(capture_path, args.name)
        obj_list = os.listdir(source_path)
        for obj_name in obj_list:
            index_list = os.listdir(os.path.join(source_path, obj_name))
            for index_name in index_list:
                source_path = os.path.join(capture_path, args.name, obj_name, index_name,"videos")
                destination_path = os.path.join(home_dir, "shared_data", args.name, obj_name, index_name,"videos")
                total_size = get_total_size(source_path, destination_path)

                copy_to_nfs(source_path, destination_path, total_size, end_with=".json")

        
    print("Copy completed.")
        
