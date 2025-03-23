from paradex.utils.io import get_video_list
from paradex.process.select_frame import split_video
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from paradex.utils.io import find_latest_directory, shared_dir, load_cam_param, capture_path_list, get_video_list
import json

def process_video(args):
    """
    Process a single video: split video into frames.
    - Splits a video into frames and saves them in a directory.
    """
    video_path, intrinsics, selected_frame, index_offset = args
    split_video(video_path, intrinsics, selected_frame, index_offset)
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    # parser.add_argument("--name", type=str, required=True, help="Name of the directory to split.")
    parser.add_argument("--name_list", type=str, nargs="+", default=None, help="List of directories to split.")
    parser.add_argument("--cam_param", type=str, default=None, help="Camera parameter file.")
    args = parser.parse_args()

    if args.name_list == None:
        args.name_list = os.listdir(os.path.join(capture_path_list[0], "capture"))
    intrinsics, extrinsics = load_cam_param(args.cam_param)

    for name in args.name_list:

        if not os.path.exists(os.path.join(capture_path_list[0], "capture", name)) or not os.path.exists(os.path.join(capture_path_list[1], "capture", name)):
            print(f"Directory {name} not found.")
            continue    
        
        index_list = os.listdir(os.path.join(capture_path_list[0], "capture", name))
        
        video_list = []
        index_offset = 0
        for index in index_list:
            for capture_path in capture_path_list:
                
                selected_frame = json.load(open(os.path.join(shared_dir, "capture", name, index, "selected_frame.json")))
                video_dir = os.path.join(capture_path, "capture", name, index)

                for vp in get_video_list(video_dir):
                    serial = os.path.basename(vp[0]).split("_")[0]
                    video_list.append((vp, intrinsics[serial], selected_frame, index_offset))

            index_offset += len(selected_frame.keys())

        with Pool(processes=cpu_count()) as pool:
            with tqdm(total=len(video_list), desc="Total Progress", unit="dir") as outer_bar:
                for _ in pool.imap_unordered(process_video, video_list):
                    outer_bar.update(1)
            
