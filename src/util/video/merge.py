import os
import numpy as np
import argparse
import trimesh
from multiprocessing import Pool

from paradex.utils.file_io import rsc_path, shared_dir, load_camparam, get_robot_urdf_path
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.image.merge import merge_image
from paradex.image.overlay import overlay_mask
from paradex.video.process_video import process_video_list

from paradex.video.convert_codec import change_to_h264
from paradex.utils.upload_file import copy_file

from paradex.visualization_.renderer import BatchRenderer
from paradex.visualization_.robot_module import Robot_Module
from paradex.robot.mimic_joint import parse_inspire
import copy

def load_info(video_dir):
    return None

def process_frame(img_dict, video_path, fid, data):
    frame = merge_image(img_dict)
    return frame

def process_frame_list(img_dict, video_dir, fid):
    frame = merge_image(img_dict)
    return frame

# td = 0.09 # latency difference between camera and sensor
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    
    args = parser.parse_args()

    process_video_list(args.path, 
            args.out_path, 
            None, 
            process_frame)
    change_to_h264(f"{args.out_path.split('.')[0]}_tmp.avi", args.out_path)