import os
import numpy as np
import trimesh

from paradex.utils.file_io import rsc_path, shared_dir, load_camparam, get_robot_urdf_path, home_path
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

from .util import get_process_list, get_argument, get_path

def load_info(video_dir):
    root_path = os.path.dirname(video_dir) # obj_name/grasp_type/index/video
    serial_list = [vid_name.split('.')[0] for vid_name in os.listdir(os.path.join(root_path, "videos"))]
    
    name = os.path.basename(os.path.dirname(os.path.dirname(root_path)))
    grasp_type =os.path.basename(os.path.dirname(root_path))
    index = os.path.basename(root_path)
    
    mesh = trimesh.load(os.path.join(rsc_path, "object", name, f"{name}.obj"))
    nas_path = os.path.join(shared_dir, "capture_", "lookup", name, grasp_type, index)
    intrinsic, extrinsic = load_camparam(nas_path)
    
    obj_T = np.load(os.path.join(nas_path, "obj_T.npy"))
    cor_3d = np.load(os.path.join(nas_path, "cor_3d.npy"),allow_pickle=True).item()
    c2r = np.load(os.path.join(nas_path, "C2R.npy"))
    
    qpos_arm = np.load(os.path.join(nas_path, "xarm", "qpos.npy"))
    qpos_hand = np.load(os.path.join(nas_path, "inspire", "qpos.npy"))
    qpos_hand = parse_inspire(qpos_hand)
    qpos = np.concatenate([qpos_arm, qpos_hand], axis=1)

    extrinsic_list = []
    intrinsic_list = []
    cammtx_list = []
    
    for serial_name in serial_list:
        extmat = extrinsic[serial_name]
        extrinsic_list.append(extmat @ c2r)
        
        intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
        print(extrinsic_list[-1].shape, intrinsic_list[-1].shape)
        cammtx_list.append(intrinsic_list[-1] @ extrinsic_list[-1])
    rm = Robot_Module(get_robot_urdf_path("xarm", "inspire"), state=qpos)
    renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')

    return mesh, renderer, cor_3d, obj_T, rm, serial_list, cammtx_list, c2r

def process_frame(img_dict, video_path, fid, data):
    (mesh, renderer, cor_3d, obj_T, rm, serial_list, cammtx_list, c2r) = data
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.apply_transform(np.linalg.inv(c2r) @ obj_T[fid])
    # tmp = (cammtx_list[0][:3,:3] @ transformed_mesh.vertices.T + cammtx_list[0][:3,3:]).T
    # print(tmp[:,:2] / tmp[:,2:])
    # import pdb; pdb.set_trace()
    
    if np.linalg.norm(obj_T) > 0.1:
        frame, mask = project_mesh_nvdiff(transformed_mesh, renderer)
        mask = mask.detach().cpu().numpy()[:,:,:,0]
        for i, serial_num in enumerate(serial_list):
            img_dict[serial_num] = overlay_mask(img_dict[serial_num], mask[i], 0.3, (255,0, 0))
    
    robot_mesh = rm.get_mesh(fid)
    for mesh in robot_mesh:
        frame, mask = project_mesh_nvdiff(mesh, renderer)
        mask = mask.detach().cpu().numpy()[:,:,:,0]
        for i, serial_num in enumerate(serial_list):
            img_dict[serial_num] = overlay_mask(img_dict[serial_num], mask[i], 0.3, (0, 255, 0))
        
    for id, cor in cor_3d[fid+1].items():
        if cor is None:
            continue
        cor_h = np.concatenate([cor, np.ones((cor.shape[0], 1))], axis=1)
        cor = (np.linalg.inv(c2r) @ cor_h.T).T[:,:3]
        for i, serial_num in enumerate(serial_list):
            img_dict[serial_num] = project_point(cor, cammtx_list[i], img_dict[serial_num])
    
    frame = merge_image(img_dict)
    return frame

def process_frame_list(img_dict, video_dir, fid):
    frame = merge_image(img_dict)
    return frame

if __name__ == '__main__':
    args = get_argument()
    process_list = get_process_list(args)
    path = get_path(args)
    
    nas_dir = os.path.join(shared_dir, path)
    download_dir = os.path.join(home_path, "download", path)
    
    for obj, hand, index in process_list:
        index_dir = os.path.join(os.path.join(root_dir, str(index)))
        os.makedirs(os.path.join(name, grasp_type, index, "overlay"), exist_ok=True)
        os.makedirs(os.path.join(name, grasp_type, index, "videos"), exist_ok=True)
        
        # download video to temporary path
        for video_name in os.listdir(os.path.join(index_dir, "videos")):
            copy_file(os.path.join(index_dir, "videos", video_name), os.path.join(name, grasp_type, index, "videos", video_name))
        
        
        if os.path.exists(os.path.join(name, grasp_type, index, "merge_overlay.mp4")):
            continue
        process_video_list(os.path.join(name, grasp_type, index, "videos"), 
                os.path.join(name, grasp_type, index, "overlay"), 
                load_info(os.path.join(name, grasp_type, index, "videos")), 
                process_frame)
        change_to_h264(os.path.join(name, grasp_type, index, "overlay_tmp.avi"), os.path.join(name, grasp_type, index, "merge_overlay.mp4"))