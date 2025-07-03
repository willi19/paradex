import os
from paradex.utils.file_io import get_video_list
from paradex.video.compress_video import compress_video
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from paradex.utils.file_io import find_latest_directory, shared_dir, load_cam_param, capture_path_list, get_video_list
import json

def process_video(args):
    """
    Process a single video: split video into frames.
    - Splits a video into frames and saves them in a directory.
    """
    video_path = args
    compress_video(video_path)
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    # parser.add_argument("--name", type=str, required=True, help="Name of the directory to split.")
    parser.add_argument("--name_list", type=str, nargs="+", default=None, help="List of directories to split.")
    args = parser.parse_args()

    if args.name_list == None:
        args.name_list = os.listdir(os.path.join(capture_path_list[0], "capture"))
    
    for name in args.name_list:
        try:
            if not os.path.exists(os.path.join(capture_path_list[0], "capture", name)) or not os.path.exists(os.path.join(capture_path_list[1], "capture", name)):
                print(f"Directory {name} not found.")
                continue    
            
            index_list = os.listdir(os.path.join(shared_dir, "capture", name))
            index_list.sort(key=lambda x: int(x))
            
            video_list = []
            index_offset = 0
            for index in index_list:
                for capture_path in capture_path_list:
                    
                    video_dir = os.path.join(capture_path, "capture", name, index, "videos")
                    if not os.path.exists(video_dir):
                        continue

                    for vp in get_video_list(video_dir):
                        serial = os.path.basename(vp[0]).split("-")[0]
                        video_list.append((vp))

            with Pool(processes=cpu_count()) as pool:
                with tqdm(total=len(video_list), desc="Total Progress", unit="dir") as outer_bar:
                    for _ in pool.imap_unordered(process_video, video_list):
                        outer_bar.update(1)
                
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue