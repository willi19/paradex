from threading import Event
import time
import argparse
import os

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir

from paradex.calibration.utils import save_current_camparam

from paradex.image.image_dict import ImageDict

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', default=None) # images are saved under shared_dir/image_path/images
parser.add_argument('--save_path', required=True)

args = parser.parse_args()

if args.image_path is None:
    rcc = remote_camera_controller("image_main.py")
    save_dir = os.path.join(shared_dir, args.save_path)

    save_current_camparam(os.path.join(shared_dir, args.save_path))
    print(f"Capturing image to {args.save_path}")

    rcc.start("image", False, f'shared_data/{args.save_path}/raw')
    rcc.stop()

    img_dict = ImageDict.from_path(os.path.join(shared_dir, args.save_path))
    img_dict.undistort(os.path.join(shared_dir, args.save_path))

    # img is saved under os.path.join(shared_dir, args.save_path, "images")
rcc.end()