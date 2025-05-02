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

import torch



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





def pick_corners(impath, numCorners, window_name = "Detected Window"): 
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

    img = cv2.imread(impath) # read image
    H, W = img.shape[:2]

    ratio = 2
    imgResized = cv2.resize(img, (W//ratio, H//ratio), interpolation=cv2.INTER_LINEAR)
    canvas = copy.deepcopy(imgResized)
    keypoint = dict()
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
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Undo all events
            print("Right called ", curIdx)

            keypoint[curIdx]=None

        canvas = copy.deepcopy(imgResized)
        for idx in keypoint:
            if keypoint[idx]:
                cv2.circle(canvas, (keypoint[idx][0], keypoint[idx][1]), radius=1, color=colors[idx], thickness=4)
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
                    print(f"skipped {curIdx}")
                else:
                    print(keypoint[curIdx])
                curIdx += 1
                break

    cv2.destroyAllWindows()
    # Resize keypoints
    
    rescaled_keypoint = dict()
    for idx in range(0, numCorners):
        if keypoint[idx] is not None :
            rescaled_keypoint[idx] = rescale_corners(keypoint[idx], H, W) 
            cv2.circle(img, tuple(rescaled_keypoint[idx].astype(int)), radius=1, color=colors[idx], thickness=4)

    cv2.imshow('debug', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(rescaled_keypoint)
    # while True:
    #     cv2.imshow("check if it is right", img)
    #     key = cv2.waitKey(0) & 0xFF
    #     if key == 27:
    #         break    
    return rescaled_keypoint