import os
import numpy as np
import argparse
import trimesh
from multiprocessing import Pool

from paradex.utils.file_io import rsc_path, shared_dir, load_camparam, get_robot_urdf_path
from paradex.image.projection import get_cammtx, project_point, project_mesh
from paradex.geometry.triangulate import ransac_triangulation
from paradex.geometry.math import rigid_transform_3D
from paradex.video.process_video import process_video
from paradex.visualization_.robot_module import Robot_Module
from paradex.robot.mimic_joint import parse_inspire

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
    
    qpos_arm = np.load(os.path.join(root_path, "xarm", "qpos.npy"))
    qpos_hand = np.load(os.path.join(root_path, "inspire", "qpos.npy"))
    qpos_hand = parse_inspire(qpos_hand)
    qpos = np.concatenate([qpos_arm, qpos_hand], axis=1)
    
    rm = Robot_Module(get_robot_urdf_path("xarm", "inspire"), state=qpos)
    c2r = np.load(os.path.join(root_path, "C2R.npy"))
    import pyrender
    renderer = pyrender.OffscreenRenderer(
            viewport_width=2048,
            viewport_height=1536
        )
    return mesh, intrinsic[serial_num]['intrinsics_undistort'], extrinsic[serial_num], cor_3d, obj_T, qpos, rm, c2r, renderer

def process_frame(frame, video_path, fid, data):
    (mesh, intrinsic, extrinsic, cor_3d, obj_T, qpos, rm, c2r, renderer) = data
    cammtx = intrinsic @ extrinsic
    vertex = mesh.vertices
    vertex = (obj_T[fid][:3, :3] @ vertex.T + obj_T[fid][:3, 3:]).T
    
    # frame = project_point(vertex, cammtx, frame) 
    # print(fid, obj_T[fid])
    if np.linalg.norm(obj_T[[fid]]) > 0.1:
        frame = project_mesh(frame, mesh, intrinsic, extrinsic, obj_T[fid], renderer)
        
    for id, cor in cor_3d[fid+1].items():
        if cor is None:
            continue
        frame = project_point(cor, cammtx, frame)
    
    robot_mesh = rm.get_mesh(fid, base_T=np.linalg.inv(c2r))
    for mesh in robot_mesh:
        frame = project_mesh(frame, mesh, intrinsic, extrinsic, np.eye(4), renderer)
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
    
    arg_list = []
    
    for name, grasp_type in process_list:
        root_dir = os.path.join(shared_dir, "capture_", "lookup", name, grasp_type)
        index_list = os.listdir(root_dir)
        
        for index in index_list:
            index_dir = os.path.join(os.path.join(root_dir, str(index)))
            for video_name in os.listdir(os.path.join(index_dir, "videos")):
                os.makedirs(os.path.join(name, grasp_type, index, "overlay"), exist_ok=True)
                arg_list.append((os.path.join(index_dir, "videos", video_name), os.path.join(name, grasp_type, index, "overlay", video_name), load_info, process_frame))

    # process_video(*arg_list[0])
    with Pool() as pool:
        pool.starmap(process_video, arg_list)  