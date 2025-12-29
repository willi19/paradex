from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
print(sys.path)

from threading import Event
import time
import argparse
import os

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir

from paradex.calibration.utils import save_current_camparam
from paradex.image.image_dict import ImageDict

from _object_6d_tracking.run.run_multiview_matching26d import CaptureAndOptimizePipeline
from _object_6d_tracking.multiview_utils.template_matcher import MatchProcessor
from _object_6d_tracking.pose_utils.scene import Scene
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', default=None) # images are saved under shared_dir/image_path/images
parser.add_argument('--save_path', default=None)

parser.add_argument("--obj_names", type=str, required=True, nargs="+")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--vis_final", action="store_true")
parser.add_argument("--loss_thres", type=float, default=20)
parser.add_argument("--paircount", type=int, default=10)
parser.add_argument("--inliers_threshold", type=int, default=10)
parser.add_argument("--saveimg", action="store_true")
parser.add_argument(
    "--max_workers", type=int, default=8, help="thread pool size for matching"
)

args = parser.parse_args()

assert args.image_path is not None or args.save_path is not None, "Either image_path or save_path must be provided"
if args.image_path is None:
    rcc = remote_camera_controller("image_main.py")
    save_dir = os.path.join(shared_dir, args.save_path)

    save_current_camparam(os.path.join(shared_dir, args.save_path))
    print(f"Capturing image to {args.save_path}")

    rcc.start("image", False, f'shared_data/{args.save_path}/raw')
    rcc.stop()

    image_dict = ImageDict.from_path(os.path.join(shared_dir, args.save_path))
    image_dict.undistort(os.path.join(shared_dir, args.save_path))
    # img is saved under os.path.join(shared_dir, args.save_path, "images")
    scene_path = os.path.join(shared_dir, args.save_path)
else:
    image_dict = ImageDict.from_path(os.path.join(shared_dir, args.image_path))
    if not os.path.exists(os.path.join(shared_dir, args.image_path, "images")):
        image_dict.undistort(os.path.join(shared_dir, args.image_path))
    scene_path = os.path.join(shared_dir, args.image_path)

scene = Scene(scene_path=Path(scene_path), mask_dir_nm=None, mask_module=None,\
                yolo_prompt=None, use_pretrained_yolo=False, image_dir=None, device=DEVICE)
cam_intr = scene.cam2intr
cam_extr = scene.cam2extr_4X4
cam_proj = scene.proj_matrix

# img_bucket = {}
# for cam_id in scene.cam_ids:
#     img_bucket[cam_id] = scene.get_image(cam_id, 0)

# import ipdb; ipdb.set_trace()

scene.get_batched_renderer(tg_cam_list=scene.cam_ids)

# Matcher finding matches between given image and pre-made template.
match_processor = MatchProcessor(obj_names=args.obj_names, paircount_threshold=args.paircount, \
                                                            saveimg=args.saveimg, max_workers=args.max_workers)
pipeline = CaptureAndOptimizePipeline(args, match_processor, \
                                    scene=scene, cam_intr=cam_intr, cam_extr=cam_extr, cam_proj=cam_proj, debug=args.debug)
parsed_ttl_output_dict = pipeline.run(img_bucket=image_dict, idx=0)
torch.save(parsed_ttl_output_dict, os.path.join(scene_path, "obj_T_multiview_matching.pkl"))
