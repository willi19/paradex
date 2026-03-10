from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
print(sys.path)

import argparse
import os

from paradex.utils.path import shared_dir
from paradex.image.image_dict import ImageDict
from paradex.calibration.utils import save_current_camparam, save_current_C2R

def undistort(image_path, save_camparam=True, save_c2r=True):
    img_dict = ImageDict.from_path(os.path.join(shared_dir, image_path))
    img_dict.undistort(os.path.join(shared_dir, image_path))

    if save_camparam:
        save_current_camparam(os.path.join(shared_dir, image_path))
    if save_c2r:
        save_current_C2R(os.path.join(shared_dir, image_path))
    
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to the image folder to undistort",
    )
    args = parser.parse_args()

    # root_path=args.root_path
    # for obj_nm in os.listdir(root_path):
    #     for scene_nm in os.listdir(os.path.join(root_path, obj_nm)):
    #         image_path = os.path.join(root_path, f'{obj_nm}/{scene_nm}')
    undistort(args.save_path)
            # save_current_C2R(image_path,)