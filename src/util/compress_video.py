import subprocess
import argparse
from paradex.utils.io import capture_path_list
from paradex.utils.video import convert_avi_to_mp4
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively copy files and directories, skipping identical files but overwriting corrupt ones.")
    parser.add_argument("--name", required=True, nargs="+", help="Path to the source file or directory")
    
    args = parser.parse_args()

    for capture_path in capture_path_list:
        for name in args.name:
            for idx in os.listdir(os.path.join(capture_path, name)):
                source_dir = os.path.join(capture_path, name, idx)
                output_dir = os.path.join(capture_path, name, idx, "compressed")
                os.makedirs(output_dir, exist_ok=True)
                video_list = [f for f in os.listdir(source_dir) if f.endswith(".avi")]
                for video in video_list:
                    source_path = os.path.join(source_dir, video)
                    output_path = os.path.join(output_dir, video.replace(".avi", ".mp4"))

                    convert_avi_to_mp4(source_path, output_path)
                    print(f"Compressed {source_path} to {output_path}")
