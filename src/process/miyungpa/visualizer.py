import os
import shutil
import numpy as np
import cv2
import tqdm
import time
import trimesh
import torch

from paradex.utils.path import shared_dir, home_path
from paradex.dataset_acqusition.match_sync import get_synced_data, fill_framedrop
from paradex.image.image_dict import ImageDict
from paradex.calibration.utils import load_camparam, load_c2r
from paradex.visualization.robot import RobotModule
from paradex.robot.utils import get_robot_urdf_path
from paradex.image.merge import merge_image
from paradex.utils.upload_file import rsync_copy
from paradex.video.util import convert_avi_to_mp4
from paradex.robot.inspire import parse_inspire
from paradex.visualization.visualizer.viser import ViserViewer
import pickle

# shared_data/capture/miyungpa/clock/2025-12-02_02-01-36/clock_optim/final/obj_output_after_optim_total_fibonacci_5_.pickle

# /home/temp_id/shared_data/capture/miyungpa/clock/2025-12-02_02-01-36/clock_optim/final




demo_root_path = os.path.join(shared_dir, "capture/miyungpa")
for obj_name in ['clock']:# os.listdir(demo_root_path)[-2:]:
    index_list = os.listdir(os.path.join(demo_root_path, obj_name))
    for index in index_list:
        demo_path = os.path.join(demo_root_path, obj_name, index)
        
        c2r = load_c2r(os.path.join(demo_path))
        arm_pos = np.load(os.path.join(demo_path, "arm", "position.npy"))
        hand_pos = np.load(os.path.join(demo_path, "hand", "position.npy"))
        hand_pos = parse_inspire(hand_pos)
        
        obj_pos = pickle.load(open(os.path.join(demo_path, f"{obj_name}_optim", "final", "obj_output_after_optim_total_fibonacci_5_.pickle"), "rb"))
        obj_T = np.eye(4)
        obj_T[:3, 3] = obj_pos['t'].cpu().numpy()
        obj_T[:3, :3] = obj_pos['R'].cpu().numpy()
        
        obj_T = np.linalg.inv(c2r) @ obj_T
        
        mesh = trimesh.load(os.path.join(shared_dir, "mesh", f"{obj_name}.ply"))
        # mesh.apply_transform(np.linalg.inv(c2r))

        pos = np.concatenate([arm_pos, hand_pos], axis=1)
        vis = ViserViewer()

        vis.add_robot('robot', get_robot_urdf_path('xarm', 'inspire'))
        vis.add_object('object', mesh, obj_T)
        vis.add_traj('asdf', {'robot':pos})
        vis.add_floor(-0.0525)
        vis.add_contact_module('robot', 'object')
        vis.start_viewer()
