import os
import json
from paradex.utils.merge_video import merge_video_synced

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge video files.")
    parser.add_argument("--input_dir", type=str, help="Input directory containing video files.")
    parser.add_argument("--output_file", type=str, help="Output file path.")

    args = parser.parse_args()

    merge_video_synced(args.input_dir, args.output_file)