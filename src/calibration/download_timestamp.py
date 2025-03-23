import os
import shutil
import argparse
import time
from tqdm import tqdm
from paradex.utils.upload_file import get_total_size, copy_to_nfs

home_dir = os.path.expanduser("~")
capture_path_list = [os.path.join(home_dir, f"captures{i}") for i in range(1,3)]

def get_total_size(path, dest_path):
    """Calculate total size of files that need to be copied (excluding identical existing files)."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            src_file = os.path.join(dirpath, f)
            dest_file = os.path.join(dest_path, os.path.relpath(src_file, path))

            # Count only files that need to be copied (new or different)
            if not os.path.exists(dest_file) or os.path.getsize(dest_file) != os.path.getsize(src_file):
                total_size += os.path.getsize(src_file)
    
    return total_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively copy files and directories, skipping identical files but overwriting corrupt ones.")
    parser.add_argument("--name", required=True, help="Path to the source file or directory")
    
    args = parser.parse_args()
    source_path = os.path.join(home_dir, "shared_data", "capture", args.name)
    destination_path = os.path.join(home_dir, "download", "capture", args.name)

    total_size = get_total_size(source_path, destination_path)
        
    copy_to_nfs(source_path, destination_path, total_size, end_with=".json")

    print("Copy completed.")
        
