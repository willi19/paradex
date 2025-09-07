import os
import glob
from pathlib import Path
import pickle
import json

def get_optimal_T(directory_path, obj_name=None):
    if obj_name is None:
        obj_name = directory_path.split("/")[-2]
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

    if min_file_nm is not None:
        min_id = min_file_nm.split("/")[-1].split(".")[0].split("loss_")[-1]
        print(f'min idx: {min_id}')
        return str(result_dir/f'obj_output_after_optim_total_{min_id}.pickle')
    else:
        return None

def get_obj_trajectory(directory_path):
    tracking_result_path = Path(directory_path)/'object_tracking'/'tracking_output.pickle'

import numpy as np 
import trimesh
from paradex.object_detection.obj_utils.vis_utils import read_mesh, get_initial_mesh

def get_obj_info(scene_path, object_nm, obj_status_path, return_type='trimesh', device=None):            
    import open3d as o3d
    obj_mesh, scaled = get_initial_mesh(object_nm, return_type=return_type, simplify=True, device=device)
    obj_scale = 1.0
    obj_trajectory = {}
    # Get Trajectory
    if os.path.exists(scene_path/'object_tracking/trajectory.pickle'):
        obj_trajectory = pickle.load(open(scene_path/'object_tracking/trajectory.pickle','rb'))

    if obj_status_path is None:
        obj_status_path = get_optimal_T(str(scene_path))
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

def resize_with_aspect_ratio(image, maximumL=1920):
    target_width, target_height = maximumL, maximumL
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    if scale>=1:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    return resized

    # Pad to target size
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


def makevideo(img_paths, output_video_path, fps=5, delete_imgs=True):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = resize_with_aspect_ratio(cv2.imread(img_paths[0]))
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(frame.shape[1],frame.shape[0]) )
    if not video_writer.isOpened():
        print("Failed to open VideoWriter")
    for img_path in img_paths:
        frame = resize_with_aspect_ratio(cv2.imread(img_path))
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



def read_camerainfo(camerainfo_dir):
    extrinsics_dict = json.load(open(Path(camerainfo_dir)/'cam_param'/'extrinsics.json'))
    intrinsics_dict = json.load(open(Path(camerainfo_dir)/'cam_param'/'intrinsics.json'))

    height, width = None, None
    proj_matrix = {}
    cam_params = {}
    cam2extr = {}
    cam2intr = {}
    # camera_centers = {}

    # Setting Camera Parameters
    for cam_id in extrinsics_dict:
        extrinsic_np = np.array(extrinsics_dict[cam_id]) # 3X4
        intrinsic_key = 'intrinsics_undistort' if 'intrinsics_undistort' in intrinsics_dict[cam_id] else 'original_intrinsics'
        intrinsic_np = np.array(intrinsics_dict[cam_id][intrinsic_key]).reshape(3,3)
        dist = intrinsics_dict[cam_id]['dist_params'] if 'dist_params' in intrinsics_dict[cam_id] else  intrinsics_dict[cam_id]['dist_param'] 
        cam_params[cam_id] = {'extrinsic':extrinsic_np, 'intrinsic':intrinsic_np.reshape(3,3), 'dist':dist}
        cam2extr[cam_id] = extrinsic_np
        cam2intr[cam_id] = intrinsic_np

        # cam_center = -np.linalg.inv(extrinsic_np[:3,:3])@extrinsic_np[:3,3]
        # camera_centers[cam_id] = cam_center

        proj_matrix[cam_id] = cam_params[cam_id]['intrinsic']@cam_params[cam_id]['extrinsic']
        height, width = intrinsics_dict[cam_id]['height'], intrinsics_dict[cam_id]['width']
    return height, width, proj_matrix, cam_params, cam2extr, cam2intr