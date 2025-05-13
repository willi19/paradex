
import argparse
import os
import json
from paradex.utils.file_io import config_dir

parser = argparse.ArgumentParser(description="Capture intrinsic camera calibration.")
parser.add_argument(
    "--serial",
    type=str,
    required=True,
    help="Directory to save the video.",
)

pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
args = parser.parse_args()
serial_num = int(args.serial)

pc_name = None
for pc in pc_info.keys():
    if serial_num in pc_info[pc]['cam_list']:
        pc_name = pc
        break

if pc_name is None:
    raise ValueError(f"Serial number {serial_num} not found in PC list.")

print(pc_name)
# Get serial number

# Connect pc with according serial number

# Get and plot current marker position

# Capture when it is inside the image
