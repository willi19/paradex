import os
import json
from paradex.video.merge_video import merge_video_synced
from paradex.utils.io import home_path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge video files.")
    parser.add_argument("--obj_list", type=str, nargs="+", help="List of objects to merge.")

    args = parser.parse_args()

    for obj in args.obj_list:
        root_dir = os.path.join(home_path, "download", "processed", obj)
        ind_list = os.listdir(root_dir)
        for ind in ind_list:
            input_dir = os.path.join(root_dir, ind, "video")
            output_file = os.path.join("video","merged", obj, f"{ind}.mp4")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            merge_video_synced(input_dir, output_file)