import os
import glob
from pathlib import Path
import pickle
import json
import re
import numpy as np 
import trimesh

def get_camera_params(input_folder):
    
    intrinsics_dict = json.load(open(os.path.join(input_folder, 'cam_param', 'intrinsics.json')))
    extrinsics_dict = json.load(open(os.path.join(input_folder, 'cam_param', 'extrinsics.json')))
    proj_matrix = {}
    cam_params = {}
    KRT = {}
    for cam_id in extrinsics_dict:
        extrinsic_np = np.array(extrinsics_dict[cam_id]) # 3X4
        intrinsic_np = np.array(intrinsics_dict[cam_id]['intrinsics_undistort'])
        height, width = intrinsics_dict[cam_id]['height'], intrinsics_dict[cam_id]['width']
        dist = np.array(intrinsics_dict[cam_id]['dist_params'])
        K_original = np.array(intrinsics_dict[cam_id]["original_intrinsics"])

        cam_params[cam_id] = {'extrinsic':extrinsic_np, 'intrinsic':intrinsic_np.reshape(3,3), 'size':(height, width) , 'dist': dist, 'K_original': K_original}

        proj_matrix[cam_id] = cam_params[cam_id]['intrinsic']@cam_params[cam_id]['extrinsic']
        K = intrinsic_np
        R = extrinsic_np[:, :3]  # 3x3 회전행렬
        t = extrinsic_np[:, 3] 
        KRT[cam_id] = {'K': K, 'R': R, 't': t}
    return cam_params, proj_matrix, KRT

def get_optimal_T(directory_path, obj_name):
    # obj_name = directory_path.split("/")[-2]
    result_dir = Path(directory_path)/f'{obj_name}_optim'/'final'
    all_files = glob.glob(str(result_dir/'obj_final_loss_*.json'), recursive=False)
    if len(all_files)<=0:
        return None
    min_loss = None
    min_file_nm = None
    for file_path in all_files:
        final_loss = json.load(open(file_path))
        if min_loss is None or final_loss['rgb_loss'] < min_loss:
            min_loss = final_loss['rgb_loss']
            min_file_nm = file_path
    print(min_file_nm)
    print(min_file_nm)
    if min_file_nm is not None:
        min_id = min_file_nm.split("/")[-1].split(".")[0].split("loss_")[-1]
        print(min_id)
        return str(result_dir/f'obj_output_after_optim_total_{min_id}.pickle')
    else:
        return None

def get_obj_trajectory(directory_path):
    tracking_result_path = Path(directory_path)/'object_tracking'/'tracking_output.pickle'


from paradex.pose_utils.vis_utils import read_mesh, get_initial_mesh

import importlib.util, sys

def _ensure_numpy_pickle_aliases():
    """Make numpy.core and numpy._core interchangeable for pickle compatibility."""
    has_core   = importlib.util.find_spec("numpy.core")   is not None
    has__core  = importlib.util.find_spec("numpy._core")  is not None

    # If only one exists, alias the missing one to the existing one.
    if has_core and not has__core:
        import numpy.core as _core
        sys.modules["numpy._core"] = _core
    elif has__core and not has_core:
        import numpy._core as _core
        sys.modules["numpy.core"] = _core

def get_obj_info(scene_path, object_nm, obj_status_path, return_type='trimesh', device=None):            
    import open3d as o3d
    print(scene_path)
    print(object_nm)
    obj_mesh, scaled = get_initial_mesh(object_nm, return_type=return_type, simplify=True, device=device)
    obj_scale = 1.0
    obj_trajectory = {}
    # Get Trajectory
    
    if os.path.exists(os.path.join(scene_path, 'object_tracking/trajectory.pickle')):
        _ensure_numpy_pickle_aliases()
        obj_trajectory = pickle.load(open(os.path.join(scene_path, 'object_tracking/trajectory.pickle'),'rb'))

    if obj_status_path is None:
        obj_status_path = get_optimal_T(str(scene_path), object_nm)
    else:
        obj_status_path = obj_status_path

    obj_T = None
    # initial object status
    if obj_status_path is not None:
        obj_optim_output = pickle.load(open(obj_status_path,'rb'))
        obj_scale = obj_optim_output['scale'].detach().cpu().numpy().item(0)

        obj_R = obj_optim_output['R'].detach().cpu().numpy()
        obj_t = obj_optim_output['t'].detach().cpu().numpy()
        obj_T = np.eye(4)
        obj_T[:3,:3] = obj_R
        obj_T[:3,3] = obj_t

    # scale obj_mesh
    if not scaled:
        if isinstance(obj_mesh, trimesh.Trimesh):
            obj_mesh.apply_scale(obj_scale)
        elif isinstance(obj_mesh, o3d.geometry.TriangleMesh):
            obj_mesh = obj_mesh.scale(obj_scale, center=np.zeros(3))
        else:
            print("Not implemented YET")

    return obj_mesh, obj_T, obj_trajectory


def get_obj_realtime_t(scene_path, object_nm, return_type='trimesh', device=None):
    # /home/jisoo/teserract_nas/demo_250618/pringles/0/pringles_optim/final/init_transl_0.pickle

    # def get_latest_file(directory):
    #     files = [os.path.join(directory, f) for f in os.listdir(directory)]
    #     files = [f for f in files if os.path.isfile(f)]

    #     if not files:
    #         return None

    #     latest_file = max(files, key=os.path.getctime)
    #     return latest_file

    latest_file = scene_path/'transl/transl.npy'
    if latest_file:
        try:
            transl = np.load(latest_file)
        except:
            return None
        return transl
    else:
        return None


import cv2

def makevideo(img_paths, output_video_path, delete_imgs=True):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(img_paths[0])    
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (frame.shape[1],frame.shape[0]) )
    if not video_writer.isOpened():
        print("Failed to open VideoWriter")
    for img_path in img_paths:
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)
    video_writer.release()

    if delete_imgs:
        for img_path in img_paths:
            try:
                os.remove(img_path)
            except OSError as e:
                print(f"Error deleting {img_path}: {e}")

import copy 
def get_binary_mask(mask):
    result = copy.deepcopy(mask)
    result[result>0] = 1

    return result