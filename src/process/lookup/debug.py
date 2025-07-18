import os
import numpy as np
import argparse
import trimesh

from paradex.utils.file_io import rsc_path, shared_dir, load_camparam
from paradex.image.projection import get_cammtx, project_point, project_mesh
from paradex.geometry.triangulate import ransac_triangulation
from paradex.geometry.math import rigid_transform_3D
from paradex.video.process_video import process_video

def load_info(video_path):
    root_path = os.path.dirname(os.path.dirname(video_path)) # obj_name/grasp_type/index/video/*.avi
    serial_num = os.path.basename(video_path).split(".")[0]
    
    name = os.path.basename(os.path.dirname(os.path.dirname(root_path)))
    mesh = trimesh.load(os.path.join(rsc_path, "object", name, f"{name}.obj"))
    intrinsic, extrinsic = load_camparam(root_path)
    # ext = np.eye(4)
    # ext[:3,:] = extrinsic[serial_num]
    obj_T = np.load(os.path.join(root_path, "obj_T.npy"))
    cor_3d = np.load(os.path.join(root_path, "cor_3d.npy"),allow_pickle=True).item()
    
    return mesh, intrinsic[serial_num]['intrinsics_undistort'], extrinsic[serial_num], cor_3d, obj_T

def process_frame(frame, video_path, fid, data):
    (mesh, intrinsic, extrinsic, cor_3d, obj_T) = data
    cammtx = intrinsic @ extrinsic
    vertex = mesh.vertices
    vertex = (obj_T[fid][:3, :3] @ vertex.T + obj_T[fid][:3, 3:]).T
    
    # frame = project_point(vertex, cammtx, frame) 
    frame = project_mesh(frame, mesh, intrinsic, extrinsic, obj_T[fid])
    for id, cor in cor_3d[fid+1].items():
        if cor is None:
            continue
        frame = project_point(cor, cammtx, frame)
    return frame

# td = 0.09 # latency difference between camera and sensor
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_name', nargs="+", type=str, default=None)
    parser.add_argument('--grasp_type', nargs="+", type=str, default=None)
    args = parser.parse_args()

    process_list = []
    
    if args.obj_name == None:
        name_list = os.listdir(os.path.join(shared_dir, 'capture_', "lookup"))
        name_list.sort()

    else:
        name_list = [args.obj_name]
        
    for name in name_list:
        grasp_list = os.listdir(os.path.join(shared_dir, "capture_", "lookup", name))
        if args.grasp_type is not None and args.grasp_type in grasp_list:
            process_list.append((name, args.grasp_type))
        if args.grasp_type is None:
            for grasp_name in grasp_list:
                process_list.append((name, grasp_name))
    
    for name, grasp_type in process_list:
        root_dir = os.path.join(shared_dir, "capture_", "lookup", name, grasp_type)
        index_list = os.listdir(root_dir)
        
        for index in index_list:
            index_dir = os.path.join(os.path.join(root_dir, str(index)))
            for video_name in os.listdir(os.path.join(index_dir, "videos")):
                process_video(os.path.join(index_dir, "videos", video_name), "tmp.mp4", load_info, process_frame)
