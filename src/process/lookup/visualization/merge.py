import os
import numpy as np
import argparse
import trimesh
from multiprocessing import Pool
import time
import math
import cv2

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
    root_path = os.path.dirname(video_dir) # obj_name/grasp_type/index/video
    serial_list = [vid_name.split('.')[0] for vid_name in os.listdir(os.path.join(root_path, "videos"))]
    
    name = os.path.basename(os.path.dirname(os.path.dirname(root_path)))
    grasp_type =os.path.basename(os.path.dirname(root_path))
    index = os.path.basename(root_path)
    
    mesh = trimesh.load(os.path.join(rsc_path, "object", name, f"{name}.obj"))
    nas_path = os.path.join(shared_dir, "capture", "lookup", name, grasp_type, index)
    intrinsic, extrinsic = load_camparam(nas_path)
    
    obj_T = np.load(os.path.join(nas_path, "obj_T.npy"))
    cor_3d = np.load(os.path.join(nas_path, "cor_3d.npy"),allow_pickle=True).item()
    c2r = np.load(os.path.join(nas_path, "C2R.npy"))
    
    qpos_arm = np.load(os.path.join(nas_path, "xarm", "qpos.npy"))
    qpos_hand = np.load(os.path.join(nas_path, "hand", "qpos.npy"))
    
    # qpos_hand = parse_inspire(qpos_hand)
    qpos = np.concatenate([qpos_arm, qpos_hand], axis=1)

    extrinsic_list = []
    intrinsic_list = []
    cammtx_list = []

    num_images = len(serial_list)
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)
    
    new_W = 2048 // grid_rows
    new_H = 1536 // grid_rows
    
    for serial_name in serial_list:
        extmat = extrinsic[serial_name]
        extrinsic_list.append(extmat @ c2r)
        intmat = intrinsic[serial_name]['intrinsics_undistort'].copy()

        intrinsic_list.append(intmat)
        cammtx_list.append(intrinsic_list[-1] @ extrinsic_list[-1])
    rm = Robot_Module(get_robot_urdf_path("xarm", "allegro"), state=qpos)
    renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')

    return mesh, renderer, cor_3d, obj_T, rm, serial_list, cammtx_list, c2r

def process_frame(img_dict, video_path, fid, data):
    (mesh, renderer, cor_3d, obj_T, rm, serial_list, cammtx_list, c2r) = data
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.apply_transform(np.linalg.inv(c2r) @ obj_T[fid])
    # tmp = (cammtx_list[0][:3,:3] @ transformed_mesh.vertices.T + cammtx_list[0][:3,3:]).T
    # print(tmp[:,:2] / tmp[:,2:])
    # import pdb; pdb.set_trace()
    num_images = len(img_dict)
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)
    
    new_W = 2048 // grid_rows
    new_H = 1536 // grid_rows

    # for serial_num in serial_list:
    #     img_dict[serial_num] = cv2.resize(img_dict[serial_num], (new_W, new_H))

    if np.linalg.norm(obj_T) > 0.1:
        start_time = time.time()
        frame, mask = project_mesh_nvdiff(transformed_mesh, renderer)
        # print(time.time()-start_time, "render obj")
        mask = mask.detach().cpu().numpy()[:,:,:,0]

        start_time = time.time()
        for i, serial_num in enumerate(serial_list):
            img_dict[serial_num] = overlay_mask(img_dict[serial_num], mask[i], 0.3, (255,0, 0))
        # print(time.time()-start_time, "obj overlay")
    
    robot_mesh = rm.get_mesh(fid)
    print("asdf")
    for mesh in robot_mesh:
        start_time = time.time()
        frame, mask = project_mesh_nvdiff(mesh, renderer)
        # print(time.time()-start_time, "render robot")
        mask = mask.detach().cpu().numpy()[:,:,:,0]
        # resized_mask = []
        # for i in range(len(mask)):
        #     resized_mask.append(cv2.resize(mask[i], (new_W, new_H)))

        start_time = time.time()
        for i, serial_num in enumerate(serial_list):
            overlay_mask(img_dict[serial_num], mask[i], 0.3, (0, 255, 0))
        print(time.time()-start_time, "robot overlay-----")

    # for id, cor in cor_3d[fid+1].items():
    #     if cor is None:
    #         continue
    #     cor_h = np.concatenate([cor, np.ones((cor.shape[0], 1))], axis=1)
    #     cor = (np.linalg.inv(c2r) @ cor_h.T).T[:,:3]
    #     for i, serial_num in enumerate(serial_list):
    #         img_dict[serial_num] = project_point(cor, cammtx_list[i], img_dict[serial_num])
    
    frame = merge_image(img_dict)
    return frame

def process_frame_list(img_dict, video_dir, fid):
    frame = merge_image(img_dict)
    return frame

# td = 0.09 # latency difference between camera and sensor
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_name', nargs="+", type=str, default=None)
    parser.add_argument('--grasp_type', nargs="+", type=str, default=None)
    args = parser.parse_args()

    process_list = []
    
    if args.obj_name == None:
        name_list = os.listdir(os.path.join(shared_dir, 'capture', "lookup"))
        name_list.sort()

    else:
        name_list = args.obj_name
        
    for name in name_list:
        grasp_list = os.listdir(os.path.join(shared_dir, "capture", "lookup", name))
        if args.grasp_type is not None:
            for grasp_name in args.grasp_type:
                if grasp_name in grasp_list:
                    process_list.append((name, grasp_name))
        if args.grasp_type is None:
            for grasp_name in grasp_list:
                process_list.append((name, grasp_name))
    
    arg_list = []
    
    for name, grasp_type in process_list:
        root_dir = os.path.join(shared_dir, "capture", "lookup", name, grasp_type)
        index_list = os.listdir(root_dir)
        
        for index in index_list[:2]:
            index_dir = os.path.join(os.path.join(root_dir, str(index)))
            out_dir = os.path.join("capture", name, grasp_type, index)
            os.makedirs(os.path.join(out_dir, "overlay"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
            for video_name in os.listdir(os.path.join(index_dir, "videos")):
                copy_file(os.path.join(index_dir, "videos", video_name), os.path.join(out_dir, "videos", video_name))
            
            # if os.path.exists(os.path.join(out_dir, "merge_overlay.mp4")):
            #     continue

            process_video_list(os.path.join(out_dir, "videos"), 
                    os.path.join(out_dir, "overlay"), 
                    load_info(os.path.join(out_dir, "videos")), 
                    process_frame)
            change_to_h264(os.path.join(out_dir, "overlay_tmp.avi"), os.path.join(out_dir, "merge_overlay.mp4"))