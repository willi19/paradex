from paradex.utils.io import get_video_list
from paradex.utils.video import split_video
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
   
home_dir = os.path.expanduser("~")
calib_path_list = [os.path.join(home_dir, f"captures{i}", "calibration") for i in range(1,3)]

def find_latest_directory():
    dirs = [d for d in os.listdir(calib_path_list[0])] 
    if not dirs:
        print("No valid directories found.")
        return
        
        # Sort directories based on name (assuming names are time-based)
    latest_dir = max(dirs, key=str)
    
    return latest_dir

def process_video(video_path):
    """
    Process a single video: split video into frames.
    - Splits a video into frames and saves them in a directory.
    """
    image_dir = os.path.join(os.path.dirname(os.path.dirname(video_path[0])), "images")
    split_video(video_path, image_dir)
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--latest", action="store_true", help="Split the latest video files.")
    parser.add_argument("--name", type=str, help="Name of the directory to split.")

    args = parser.parse_args()

    if not args.latest and not args.name:
        print("Please specify either --latest or --name.")
        exit()

    if args.latest:
        name = find_latest_directory()
    else:
        name = args.name

    if not os.path.exists(os.path.join(calib_path_list[0], name)) or not os.path.exists(os.path.join(calib_path_list[1], name)):
        print(f"Directory {name} not found.")
        exit()
    
    video_list = []
    for calib_path in calib_path_list:
        index_list = os.listdir(os.path.join(calib_path, name))
        for index in index_list:
            video_dir = os.path.join(calib_path, name, index, "video")
            video_list += get_video_list(video_dir)

        # process_dir_list += [os.path.join(calib_path, name, d) for d in os.listdir(os.path.join(calib_path, name))]
    video_list = sorted(video_list, key=lambda x: int(x[0].split("/")[-3]))
    
    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=len(video_list), desc="Total Progress", unit="dir") as outer_bar:
            for _ in pool.imap_unordered(process_video, video_list):
                outer_bar.update(1)
        
