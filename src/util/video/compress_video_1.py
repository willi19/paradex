import os
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from paradex.video.compress_video import compress_video

def find_avi_json_pairs(folder):
    video_json_pairs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".avi"):
                video_path = os.path.join(root, f)
                json_path = video_path.replace(".avi", ".json")  # 예: 같은 폴더에 같은 이름의 .json이 있다고 가정
                video_json_pairs.append((video_path, json_path))
    return video_json_pairs

def process_video(video_json_tuple):
    compress_video(video_json_tuple)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress all .avi videos in a given folder.")
    parser.add_argument("--folder", type=str, required=True, help="Target folder containing .avi files.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Provided path '{args.folder}' is not a valid directory.")
        exit(1)

    video_list = find_avi_json_pairs(args.folder)
    print(f"Found {len(video_list)} .avi files to compress in {args.folder}")

    if len(video_list) == 0:
        print("No .avi files found. Exiting.")
        exit(0)

    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=len(video_list), desc="Compressing Videos") as pbar:
            for _ in pool.imap_unordered(process_video, video_list):
                pbar.update(1)
