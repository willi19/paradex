
import os
import numpy as np
import cv2
from numpy import concatenate as ncat
import json
import pickle
from collections import deque
from pathlib import Path
import copy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


colors = [
(0, 0, 255),    # 빨강
(0, 165, 255),  # 주황
(0, 255, 255),  # 노랑
(0, 255, 0),    # 초록
(255, 0, 0),    # 파랑
(128, 0, 0),    # 남색
(128, 0, 128)   # 보라
]


from paradex.pose_utils.geometry import transform_by_extrinsic

import trimesh
import torch
# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import TexturesVertex, TexturesUV
# from pytorch3d.io import load_obj

import torch
import torchvision.utils as vutils

def make_grid_image(images, grid_rows, grid_cols):
    N, H, W, C = images.shape
    assert N <= grid_rows * grid_cols, "Grid too small for number of images"

    # Convert (N, H, W, 3) -> (N, 3, H, W) for PyTorch
    images = images.permute(0, 3, 1, 2)

    # Pad if needed (e.g. if grid is larger than N)
    if N < grid_rows * grid_cols:
        pad_num = grid_rows * grid_cols - N
        pad_images = torch.zeros((pad_num, C, H, W), dtype=images.dtype).to(images.device)
        images = torch.cat([images, pad_images], dim=0)

    # Make grid
    grid = vutils.make_grid(images, nrow=grid_cols, padding=0)
    
    # Convert back to (H, W, 3)
    grid = grid.permute(1, 2, 0)
    return grid

def make_grid_image_np(images, grid_rows, grid_cols):
    N, H, W, C = images.shape
    assert N <= grid_rows * grid_cols, "Grid too small for number of images"

    # Pad if needed
    if N < grid_rows * grid_cols:
        pad_num = grid_rows * grid_cols - N
        pad_images = np.zeros((pad_num, H, W, C), dtype=images.dtype)
        images = np.concatenate([images, pad_images], axis=0)

    # Reshape into grid
    images = images.reshape(grid_rows, grid_cols, H, W, C)
    images = images.transpose(0, 2, 1, 3, 4)  # (rows, H, cols, W, C) -> (rows*H, cols*W, C)
    grid_image = images.reshape(grid_rows * H, grid_cols * W, C)

    return grid_image


def repeat_pytorch3d_mesh(pytorch3d_mesh, batch_num):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex, TexturesUV
    from pytorch3d.io import load_obj

    device = pytorch3d_mesh.device

    p3d_verts = pytorch3d_mesh.verts_padded().float().to(device).requires_grad_(False)
    p3d_faces = pytorch3d_mesh.faces_padded().to(device).requires_grad_(False)
    # _p3d_textures = pytorch3d_mesh.textures.verts_features_padded().float().to(device).requires_grad_(False)
    p3d_textures = pytorch3d_mesh.textures

    p3d_verts_batch = p3d_verts.repeat(batch_num, 1, 1)
    p3d_faces_batch = p3d_faces.repeat(batch_num, 1, 1)

    if type(p3d_textures) == TexturesUV:
        maps_batched = p3d_textures.maps_padded().repeat(batch_num, 1, 1, 1)
        faces_uvs_batched = p3d_textures.faces_uvs_padded().repeat(batch_num, 1, 1)
        verts_uvs_batched = p3d_textures.verts_uvs_padded().repeat(batch_num, 1, 1)

        p3d_texture_batch = TexturesUV(
            maps=maps_batched,
            faces_uvs=faces_uvs_batched,
            verts_uvs=verts_uvs_batched
        )

    else: # type(p3d_textures) == TexturesVertex
        verts_features_batched = p3d_textures.verts_features_padded().repeat(batch_num, 1, 1)
        p3d_texture_batch = TexturesVertex(verts_features=verts_features_batched)

    pytorch3d_mesh = Meshes(verts=p3d_verts_batch, faces=p3d_faces_batch, textures=p3d_texture_batch)

    return pytorch3d_mesh


MESH_DIR = os.path.join(os.environ['NAS_PATH'], 'mesh')
MESH_DIR_BEFORE_PROCESSING = MESH_DIR.replace('mesh','mesh_before_processed')

def get_initial_mesh(obj_name, return_type='open3d', post_processing=False, simplify=False, centered=True, device='cuda'):

    scaled = False
    print(MESH_DIR)
    if os.path.exists(os.path.join(MESH_DIR, obj_name+".ply")):
        mesh_path = os.path.join(MESH_DIR, obj_name+".ply")
        scaled = True
    elif os.path.exists(os.path.join(MESH_DIR, obj_name, obj_name+".obj")):
        mesh_path = os.path.join(MESH_DIR, obj_name, obj_name+".obj")
        scaled = True
        post_processing = True # For
    elif os.path.exists(os.path.join(MESH_DIR_BEFORE_PROCESSING, obj_name+".ply")):
        mesh_path = os.path.join(MESH_DIR_BEFORE_PROCESSING, obj_name+".ply")
    elif os.path.exists(os.path.join(MESH_DIR_BEFORE_PROCESSING, obj_name)):
        mesh_path = os.path.join(MESH_DIR_BEFORE_PROCESSING, obj_name, obj_name+".obj")
        post_processing = True # For
    else:
        raise "Mesh file not found"
    print(mesh_path)

    return read_mesh(mesh_path, return_type=return_type, post_processing=post_processing, simplify=simplify, centered=centered, device=device), scaled


def read_mesh(mesh_path, return_type='open3d', post_processing=False, simplify=False, centered=True, device='cuda'):
    import open3d as o3d
    assert return_type in ['open3d','trimesh','pytorch3d'], 'check mesh resturn type'

    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=post_processing)
    mesh.compute_vertex_normals()
    if centered:
        center = (mesh.get_max_bound()+mesh.get_min_bound())/2
        mesh.translate(-center)
    post_fix = mesh_path.split(".")[-1]

    if simplify and 'obj' not in mesh_path: # only simplify if ply file. (which has vertex normals)
        # if not os.path.exists(mesh_path.replace(f'.{post_fix}',f'_simplified.{post_fix}')):
        simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=2000)
        # o3d.io.write_triangle_mesh(mesh_path.replace(f'.{post_fix}',f'_simplified.{post_fix}'), simplified_mesh)
        mesh = simplified_mesh

    if mesh.has_vertex_colors():
        vertex_colors_bgr = np.array(mesh.vertex_colors)
        vertex_colors = vertex_colors_bgr.copy()
    elif mesh.has_triangle_uvs():
        mesh_uvs = np.array(mesh.triangle_uvs).reshape((-1, 3, 2))
        mesh_texture = np.array(mesh.textures[0])[:,:,::-1]

    else:
        raise 'The Mesh does not have any texutre'
    
    if return_type == 'open3d':
        # print("visualization")
        return mesh    
    elif return_type =='trimesh':
        if mesh.has_vertex_colors():
            trimesh_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles), vertex_colors=vertex_colors, process=post_processing)
        elif mesh.has_triangle_uvs():
            visual = trimesh.visual.texture.TextureVisuals(uv=mesh_uvs.reshape(-1, 2),
                                                        image=mesh_texture)
            trimesh_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles), visual=visual, process=post_processing)
        trimesh_mesh.export('trimesh_test.obj')
        return trimesh_mesh
    else: # pytorch3d
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex, TexturesUV
        from pytorch3d.io import load_obj

        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device).float().requires_grad_(False)
        faces = torch.tensor(mesh.triangles, dtype=torch.int64, device=device).float().requires_grad_(False)

        if mesh.has_vertex_colors():
            vertex_colors_t = torch.tensor(vertex_colors[np.newaxis, :, :3], dtype=torch.float32, device=device).float().requires_grad_(False)  # Normalize to [0,1]
            textures = TexturesVertex(verts_features=vertex_colors_t)
            pytorch3d_mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
        elif mesh.has_triangle_uvs():
            p3d_verts, p3d_faces, aux = load_obj(mesh_path, device=device)
            
            verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
            faces_uvs = p3d_faces.textures_idx[None, ...]  # (1, F, 3)
            tex_maps = aux.texture_images

            # tex_maps is a dictionary of {material name: texture image}.
            # Take the first image:
            texture_image = list(tex_maps.values())[0]
            texture_image = texture_image[None, ..., [2,1,0]]  # (1, H, W, 3)

            # Create a textures object
            textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image.to(device))

            pytorch3d_mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
        return pytorch3d_mesh
    

def remesh(verts, faces):
    '''
        verts: NX3,
        faces: NX3
    '''
    import trimesh
    # remesh / increase the number of verticess
    new_vertices, new_faces, index_mapping = trimesh.remesh.subdivide_to_size(verts, faces, \
                                                                    return_index=True, max_edge=0.01)
    
    num_vertices = len(new_vertices)
    new_colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (num_vertices, 1)) # new_colors as gray color 
    
    return new_vertices, new_faces, new_colors


def parse_trimesh_objdict(mesh:trimesh.Trimesh, device='cuda:0'):


    mesh_dict = {'type':'vertex_color', 'verts':torch.tensor(mesh.vertices[None], device=device).float(),\
                'faces':torch.tensor(mesh.faces, dtype=torch.int32, device=device), \
                'vtx_col':torch.tensor(mesh.visual.vertex_colors, device=device).float()[:,:3], 
                'col_idx':torch.tensor(mesh.faces, dtype=torch.int32, device=device)}
    return mesh_dict

    

def parse_objectmesh_objdict(obj_name, min_vertex_num=None, remove_uv=False, renderer_type='pytorch3d', device='cuda', simplify=True):

    if renderer_type == 'pytorch3d':
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex

        pytorch3d_mesh, scaled = get_initial_mesh(obj_name, return_type='pytorch3d', post_processing=True, simplify=simplify, device=device)
        p3d_verts = pytorch3d_mesh.verts_padded().float().to(device).requires_grad_(False)
        p3d_faces = pytorch3d_mesh.faces_padded().to(device).requires_grad_(False)
        # _p3d_textures = pytorch3d_mesh.textures.verts_features_padded().float().to(device).requires_grad_(False)
        p3d_textures = pytorch3d_mesh.textures

        if min_vertex_num is not None and p3d_verts.shape[1]<min_vertex_num:
            new_vertices, new_faces, new_colors = remesh(p3d_verts.detach().cpu().numpy()[0], p3d_faces.detach().cpu().numpy()[0])
            p3d_verts = torch.tensor(new_vertices, dtype=torch.float32, device=device).float().requires_grad_(False)
            p3d_faces = torch.tensor(new_faces, dtype=torch.int64, device=device).float().requires_grad_(False)
            vertex_colors_t = torch.tensor(new_colors[np.newaxis, :, :3], dtype=torch.float32, device=device).float().requires_grad_(False) 
            p3d_textures = TexturesVertex(verts_features=vertex_colors_t)

        if remove_uv:
            if not isinstance(p3d_textures, TexturesVertex):
                num_vertices = p3d_verts.shape[1]
                gray_color = np.tile(np.array([[0.5, 0.5, 0.5]]), (num_vertices, 1))
                vertex_colors_t = torch.tensor(gray_color[np.newaxis, :, :3], dtype=torch.float32, device=device).float().requires_grad_(False) 
                p3d_textures = TexturesVertex(verts_features=vertex_colors_t)

        return {'verts':p3d_verts, 'faces':p3d_faces, 'textures':p3d_textures, 'scaled': scaled}
    else: # nvdiff 
        o3d_mesh, scaled = get_initial_mesh(obj_name, return_type='open3d', post_processing=True, simplify=simplify, device=device)
        vtx_pos = torch.tensor(np.array(o3d_mesh.vertices)).type(torch.float32).to(device).unsqueeze(0)
        pos_idx = torch.tensor(np.array(o3d_mesh.triangles)).type(torch.int32).to(device) # triangles
        
        if min_vertex_num is not None and vtx_pos.shape[1]<min_vertex_num:
            new_vertices, new_faces, new_colors = remesh(vtx_pos.detach().cpu().numpy()[0], pos_idx.detach().cpu().numpy())
            vtx_pos = torch.tensor(o3d_mesh.vertices).type(torch.float32).to(device).unsqueeze(0)
            pos_idx = torch.tensor(new_faces).type(torch.int32).to(device)
            vtx_col = torch.tensor(new_colors).type(torch.float32)[:,[2,1,0]].to(device)

            return {'type':'vertex_color', 'verts':vtx_pos, 'faces':pos_idx, 'vtx_col':vtx_col, 'col_idx':pos_idx, 'scaled': scaled}
        elif o3d_mesh.has_vertex_colors():
            vtx_col = torch.tensor(o3d_mesh.vertex_colors).type(torch.float32)[:,[2,1,0]].to(device)
            col_idx = pos_idx
            return {'type':'vertex_color', 'verts':vtx_pos, 'faces':pos_idx, 'vtx_col':vtx_col, 'col_idx':col_idx, 'scaled': scaled}
        elif o3d_mesh.has_triangle_uvs():
            uvs = torch.tensor(np.asarray(o3d_mesh.triangle_uvs), dtype=torch.float32).cuda().reshape(-1, 2)
            uv_idx = torch.arange(uvs.shape[0], dtype=torch.int32, device=uvs.device).reshape(-1, 3)
            texture_tensor = torch.tensor(np.array(o3d_mesh.textures[0],dtype=np.float32))[:,:,[2,1,0]].cuda()/255 # normalized to 0~1, BGR-> RGB

            if remove_uv:
                gray_color = np.tile(np.array([[0.5, 0.5, 0.5]]), (vtx_pos.shape[1], 1))
                vtx_col = torch.tensor(gray_color, dtype=torch.float32, device=device).float().requires_grad_(False)
                return {'type':'vertex_color', 'verts':vtx_pos, 'faces':pos_idx, 'vtx_col':vtx_col, 'col_idx':pos_idx, 'scaled': scaled}
            else:
                return {'type':'triangle_uvs', 'verts':vtx_pos, 'faces':pos_idx, 'uvs':uvs, 'uv_idx':uv_idx, \
                        'texture_tensor':texture_tensor, 'scaled': scaled}
        else:
            raise("Mesh doesn't have texture")
    


def load_ply_as_pytorch3d_mesh(ply_path, device="cuda", centered=True, simplify=False):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex

    import open3d as o3d
    import trimesh

    if simplify:
        if not os.path.exists(ply_path.replace('.ply','_simplified.ply')):
            original_mesh = o3d.io.read_triangle_mesh(ply_path)
            org_number_of_faces = np.asarray(original_mesh.triangles).shape[0]
            # o3d.visualization.draw_geometries([original_mesh.simplify_quadric_decimation(target_number_of_triangles=2000)])
            simplified_mesh = original_mesh.simplify_quadric_decimation(target_number_of_triangles=2000)
            o3d.io.write_triangle_mesh(ply_path.replace('.ply','_simplified.ply'), simplified_mesh)
        ply_path = ply_path.replace('.ply','_simplified.ply')

    # Load the PLY file with Trimesh
    mesh = trimesh.load(ply_path, process=True)
    
    # Extract vertices and faces
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    if centered:
        vertices-=torch.mean(vertices,dim=0)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

    # Extract vertex colors if available
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vertex_colors = torch.tensor(mesh.visual.vertex_colors[None, :, :3] / 255.0, dtype=torch.float32, device=device)  # Normalize to [0,1]
        textures = TexturesVertex(verts_features=vertex_colors)
        pytorch3d_mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    else:
        textures = None
        pytorch3d_mesh = Meshes(verts=[vertices], faces=[faces])
        

    return pytorch3d_mesh


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    import matplotlib.pyplot as plt
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                    facecolor=(0, 0, 0, 0), lw=2))

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
                marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
                marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            print(box)
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def get_axisLinesSet(length=1, rotmat=np.eye(3), trans=np.zeros(3), mesh=False, radius=0.003):
    import open3d as o3d
    boc = {'y': [1, 1, 0], 'g': [0, 1, 0], 'r': [0, 0, 1], 'b': [1, 0, 0], 'black': [0, 0, 0],
        'basic': [100/255, 100/255, 100/255], 'line': [205/255, 92/255, 92/255]}

    # zero centered coordinate axis
    axis_points = np.array([[0, 0, 0],
                            [length, 0, 0],
                            [0, length, 0],
                            [0, 0, length]])

    axis_points = (np.matmul(rotmat, axis_points.T).T+trans)

    axis_lines = [[0, 1], [0, 2], [0, 3]]
    axis_colors = [boc['b'], boc['g'], boc['r']] # x:blue, y:green, z: red

    axis_lines_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(axis_points),
            lines=o3d.utility.Vector2iVector(axis_lines),
        )
    axis_lines_set.colors = o3d.utility.Vector3dVector(axis_colors)

    return [axis_lines_set]


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """

    if images.ndim==3:
        images = np.repeat(images[...,np.newaxis], 3, axis=-1)

    assert images.ndim==4, 'Should check the image dimension'

    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()


def read_frame(video_path, frame_number=0):
    '''
        read or extract img as BGR format
    '''
    supposed_extracted_path = os.path.join(str(video_path).replace('video','rgb_extracted').split(".")[0],'%05d.jpeg'%(frame_number))
    if os.path.exists(supposed_extracted_path):
        return cv2.imread(supposed_extracted_path)
    else:

        import imageio.v3 as iio
        total_frames = iio.improps(video_path).n_images
        
        if frame_number >= total_frames:
            print(f"Error: Requested frame {frame_number} is beyond total frames.")
            return None
        
        frame = iio.imread(video_path, index=frame_number) # Default: RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(f"Total frames in video: {total_frames}")

        os.makedirs(str(video_path).replace('video','rgb_extracted').split(".")[0], exist_ok=True)
        cv2.imwrite(supposed_extracted_path, frame)

        return frame


def get_frame_number(video_path):
    cap = cv2.VideoCapture(str(video_path))
    cv2.VideoCapture


    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def read_video_to_numpy(path):
    import imageio.v3 as iio

    video_np = iio.imread(path, format_hint='.avi')
    return video_np


def pick_corners(img_info, numCorners, window_name = "Detected Window", ratio = 1): 
    colors = [
    (0, 0, 255),    # 빨강
    (0, 165, 255),  # 주황
    (0, 255, 255),  # 노랑
    (0, 255, 0),    # 초록
    (255, 0, 0),    # 파랑
    (128, 0, 0),    # 남색
    (128, 0, 128),   # 보라
    (0, 0, 255),    # 빨강
    (0, 165, 255),  # 주황
    (0, 255, 255),  # 노랑
    (0, 255, 0),    # 초록
    (255, 0, 0),    # 파랑
    (128, 0, 0),    # 남색
    (128, 0, 128),   # 보라
    ]
    if type(img_info)==Path:
        img = cv2.imread(str(img_info)) # read image
    elif type(img_info)==np.ndarray:
        img = np.copy(img_info)
    else:
        raise "Type of img is not defined"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    imgResized = cv2.resize(img, (W//ratio, H//ratio), interpolation=cv2.INTER_LINEAR)
    canvas = copy.deepcopy(imgResized)
    keypoint = dict()
    labels = dict()
    curIdx = 0
    
    def click_status(event, pix_x, pix_y, flags, params):
        """
        params 0 => curIdx, 1 => img
        """
        nonlocal img, canvas, imgResized, curIdx, keypoint

        if event == cv2.EVENT_LBUTTONDOWN:
            # change status and running images
            print("Left called ", curIdx)
            keypoint[curIdx]=[pix_x, pix_y]
            labels[curIdx] = 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Undo all events
            print("Right called ", curIdx)
            keypoint[curIdx]=[pix_x, pix_y]
            labels[curIdx] = 0

        canvas = copy.deepcopy(imgResized)
        for idx in keypoint:
            if keypoint[idx]:
                if labels[idx] == 0 :
                    color = (255, 255, 0)
                else:
                    color = colors[idx]
                cv2.circle(canvas, (keypoint[idx][0], keypoint[idx][1]), radius=1, color=color, thickness=4)
        cv2.imshow(window_name, canvas)
    
    def rescale_corners(corner, originalH, originalW):
        """
        corners : (70,1,2) detected corners at downscaled image
        img : original image
        """
        nonlocal ratio
        print(corner)
        rescaled_corner = np.zeros_like(corner)
        originCenW, originCenH = (originalW-1)/2, (originalH-1)/2
        scaledW, scaledH = originalW//ratio, originalH//ratio
        centerW, centerH = (scaledW-1) / 2, (scaledH-1) / 2 
        rescaled_corner[0] = (corner[0] - centerW)*ratio + originCenW # scale ratio
        rescaled_corner[1] = (corner[1] - centerH)*ratio + originCenH
        return rescaled_corner
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_status)

    for idx in range(0, numCorners):
        # overlay whitened
        print("Current corner number push ESC if you wanna quit or not choose any point", idx)
        while True:
            #image_list = np.concatenate(current_images, axis=1)
            cv2.imshow(window_name, imgResized)
            key = cv2.waitKey(0) & 0xFF
            if key == 27: # ESC
                if curIdx not in keypoint or keypoint[curIdx] is None:
                    keypoint[curIdx] = None
                    labels[curIdx] = None
                    print(f"skipped {curIdx}")
                else:
                    print(f'{keypoint[curIdx]} {labels[curIdx]}')
                curIdx += 1
                break

    cv2.destroyAllWindows()
    # Resize keypoints
    
    rescaled_keypoint = dict()
    result_labels = []
    for idx in range(0, numCorners):
        if keypoint[idx] is not None :
            rescaled_keypoint[idx] = rescale_corners(keypoint[idx], H, W) 
            result_labels.append(labels[idx])
            if labels[idx] == 0 :
                color = (255, 255, 0)
            else:
                color = colors[idx]
            cv2.circle(img, tuple(rescaled_keypoint[idx].astype(int)), radius=1, color=color, thickness=4)

    # cv2.imshow('debug', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(rescaled_keypoint)
    # while True:
    #     cv2.imshow("check if it is right", img)
    #     key = cv2.waitKey(0) & 0xFF
    #     if key == 27:
    #         break    
    return rescaled_keypoint, result_labels




def get_ray(point, T_cam2world, intrinsic_matrix, L=1):
    x,y = point
    fx, fy, cx, cy = intrinsic_matrix[0,0],intrinsic_matrix[1,1],intrinsic_matrix[0,2],intrinsic_matrix[1,2]

    ray_center_cam = np.zeros(3, dtype=np.float64)
    ray_point_cam = np.array([(x-cx)/fx*L, (y-cy)/fy*L, L])

    ray_center_world = transform_by_extrinsic(ray_center_cam, T_cam2world)
    ray_point_world = transform_by_extrinsic(ray_point_cam, T_cam2world)

    return ray_center_world, ray_point_world


def crop_and_resize_by_mask(image: np.ndarray,
                            mask: np.ndarray,
                            output_size: int) -> np.ndarray:
    """
    이미지와 이진 마스크를 받아,
    - 마스크 영역의 바운딩 박스를 구하고
    - 그 영역을 포함하는 정사각형으로 확장한 뒤
    - output_size x output_size 로 리사이즈

    Args:
        image (H x W x 3 np.uint8): 원본 RGB 이미지
        mask  (H x W      np.uint8 or bool): 0/1 또는 False/True 마스크
        output_size (int): 최종 출력 (h = w = output_size)

    Returns:
        cropped_resized (output_size x output_size x 3 np.uint8)
    """
    # 1) 마스크가 1인 좌표들 찾기
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        # print("no rendered point")
        if image.ndim == 2:
            return np.zeros((output_size, output_size))
        else:
            return np.zeros((output_size, output_size, 3))
        # raise "mask not found" 

    # 2) 최소/최대 좌표로 바운딩 박스 계산
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width  = x_max - x_min + 1
    height = y_max - y_min + 1

    # 3) 정사각형으로 확장: 짧은 쪽을 늘려줌
    if width > height:
        # 세로를 늘리기
        delta = width - height
        y_min = max(0, y_min - delta // 2)
        y_max = min(image.shape[0]-1, y_max + (delta - delta//2))
    else:
        # 가로를 늘리기
        delta = height - width
        x_min = max(0, x_min - delta // 2)
        x_max = min(image.shape[1]-1, x_max + (delta - delta//2))

    # 4) 잘라내기
    square_crop = image[y_min:y_max+1, x_min:x_max+1]

    # 5) 리사이즈
    cropped_resized = cv2.resize(square_crop,
                                 (output_size, output_size),
                                 interpolation=cv2.INTER_LINEAR)

    return cropped_resized



import copy
import matplotlib.pyplot as plt
cmap = plt.get_cmap('magma_r')  # or 'jet', 'plasma', etc.

def get_colored_mesh(o3d_mesh, vertex_values, save_path=None, cmap_nm='magma_r'):
    import open3d as o3d
    # trimesh?
    def normalize_arr(arr:np.ndarray): 
        norm_arr = (arr - np.min ( arr)) / (np.max ( arr) - np.min ( arr)) 
        return norm_arr
    
    normalized_vertex_values = normalize_arr(vertex_values)
    cmap = plt.get_cmap(cmap_nm)
    obj_contact_colors = cmap(normalized_vertex_values)[:, :3]  # Drop alpha channel -> shape (N, 3)
    colored_mesh = copy.deepcopy(o3d_mesh)
    colored_mesh.triangle_uvs = o3d.utility.Vector2dVector()  # Clear UVs
    colored_mesh.textures = []  # Clear texture images
    
    colors_uint8 = (obj_contact_colors * 255).astype(np.uint8)
    if isinstance(o3d_mesh, o3d.geometry.TriangleMesh):
        colored_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_contact_colors)
    else: # for trimesh
        colored_mesh.visual.vertex_colors = colors_uint8
    if save_path is not None:
        colored_mesh.export(save_path)
    
    return colored_mesh



def putText(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=4, thickness=2, color=(1,0,0)):
            
    # 텍스트 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 중간 상단 좌표 계산
    x = (img.shape[1] - text_width) // 2
    y = text_height + 10  # 상단에서 약간 아래로 내려오기 위해 +10

    canvas = np.copy(img)
    cv2.putText(canvas, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return canvas
