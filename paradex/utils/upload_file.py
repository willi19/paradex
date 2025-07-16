import os
import shutil
import argparse
import time
from tqdm import tqdm

def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    with open(src, "rb") as f_src, open(dst, "wb") as f_dst:
        while chunk := f_src.read(1024 * 1024):  # Read in 1MB chunks
            f_dst.write(chunk)
            

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
