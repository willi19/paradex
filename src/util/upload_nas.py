import os
import shutil
import argparse
import time
from tqdm import tqdm

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

def copy_file_with_progress(src, dst, total_size, copied_size):
    """Copy a file with progress tracking, overwriting in case of corruption."""
    file_size = os.path.getsize(src)

    # Skip if the file already exists and is identical
    # if os.path.exists(dst) and os.path.getsize(dst) == file_size:
    #     # print(f"Skipping identical file: {src}")
    #     return copied_size

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    start_time = time.time()
    with open(src, "rb") as f_src, open(dst, "wb") as f_dst, tqdm(
        total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"Copying: {os.path.basename(src)}", ascii=True
    ) as pbar:
        while chunk := f_src.read(1024 * 1024):  # Read in 1MB chunks
            f_dst.write(chunk)
            copied_size += len(chunk)
            pbar.update(len(chunk))
            
            elapsed_time = time.time() - start_time
            speed = copied_size / elapsed_time if elapsed_time > 0 else 0  # Speed in B/s
            percent_complete = (copied_size / total_size) * 100
            remaining_size = total_size - copied_size  # Remaining data to copy

            pbar.set_postfix(
                percentage=f"{percent_complete:.2f}%",
                speed=f"{speed / (1024 * 1024):.2f} MB/s",
                remaining=f"{remaining_size / (1024 * 1024):.2f} MB left"
            )
        pbar.close()
    return copied_size

def copy_to_nfs(source_path, destination_path, total_size, copied_size=0):
    """Recursively copy files and directories while skipping identical files but overwriting corrupt ones."""
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return copied_size

    try:
        if os.path.isfile(source_path):
            copied_size = copy_file_with_progress(source_path, destination_path, total_size, copied_size)
        else:
            os.makedirs(destination_path, exist_ok=True)
            file_list = os.listdir(source_path)
            for file in file_list:
                source_file = os.path.join(source_path, file)
                destination_file = os.path.join(destination_path, file)
                copied_size = copy_to_nfs(source_file, destination_file, total_size, copied_size)
    except Exception as e:
        print(f"Error copying {source_path}, retrying: {e}")
        # Retry: Overwrite the file in case of corruption
        try:
            shutil.copy2(source_path, destination_path)
            print(f"File '{source_path}' overwritten successfully.")
        except Exception as retry_error:
            print(f"Failed to overwrite '{source_path}': {retry_error}")

    return copied_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively copy files and directories, skipping identical files but overwriting corrupt ones.")
    parser.add_argument("--name", required=True, help="Path to the source file or directory")
    
    args = parser.parse_args()
    destination_path = os.path.join(home_dir, "shared_data", args.name)

    for capture_path in capture_path_list:
        source_path = os.path.join(capture_path, args.name)
        total_size = get_total_size(source_path, destination_path)
        
        copy_to_nfs(source_path, destination_path, total_size)

    print("Copy completed.")
        
