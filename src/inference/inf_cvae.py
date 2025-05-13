import argparse
import json
import math
import os.path
import time
import sys
import shutil
from paradex.utils.file_io import shared_path

import trimesh.sample

import torch
import plotly.graph_objects as go
from paradex.visualization.visualize_plotly import plot_point_cloud, plot_point_cloud_cmap, plot_mesh
from paradex.model.set_seed import set_global_seed
import trimesh as tm
import torch.nn as nn
from paradex.model.PointNetCVAE import PointNetCVAE
import numpy as np
from paradex.model.get_models import get_handmodel

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pre_process', default='sharp_lift', type=str)
    parser.add_argument('--model', required=True, type=str)

    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--index', required=True, type=str)

    parser.add_argument('--num', default=10, type=int)

    args_ = parser.parse_args()
    tag = str(time.time())
    return args_, tag


def pre_process_sharp_clamp(contact_map):
    gap_th = 0.5  # delta_th = (1 - gap_th)
    gap_th = min(contact_map.max().item(), gap_th)
    delta_th = (1 - gap_th)
    contact_map[contact_map > 0.4] += delta_th
    # contact_map += delta_th
    contact_map = torch.clamp_max(contact_map, 1.)
    return contact_map

def identity_map(contact_map):
    return contact_map


if __name__ == '__main__':
    set_global_seed(seed=42)
    args, time_tag = get_parser()

    pre_process_map = {'sharp_lift': pre_process_sharp_clamp,
                       'identity': identity_map}
    pre_process_contact_map_goal = pre_process_map[args.pre_process]

    logs_basedir = os.path.join(shared_path, "inference", f'{args.name}', f'{args.index}', 'contact_map')
    os.makedirs(logs_basedir, exist_ok=True)

    device = "cuda"
    model_path = f"weight/{args.model}.pth"
    model: nn.Module
    model = PointNetCVAE(latent_size=128,
                            encoder_layers_size=[4, 64, 128, 512],
                            decoder_global_feat_size=512,
                            decoder_pointwise_layers_size=[3, 64, 64],
                            decoder_global_layers_size=[64, 128, 512],
                            decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    object_name = args.name
    object_index = args.index
    
    object_mesh: tm.Trimesh
    object_mesh = tm.load(os.path.join(shared_path, 'mesh', f'{args.name}.ply'))

    wrist_T = np.load(os.path.join(shared_path, 'contact_map', args.name, args.index, "wrist_T.npy"))
    object_T = np.load(os.path.join(shared_path, 'contact_map', args.name, args.index, "object_pose.npy"))

    gen_grasp_offset = np.zeros((4, 4))
    gen_grasp_offset[2, 0] = -1
    
    gen_grasp_offset[1, 2] = -1

    gen_grasp_offset[0, 1] = -1
    gen_grasp_offset[3, 3] = 1
    
    object_T = np.linalg.inv(wrist_T) @ object_T# @ gen_grasp_offset
    # object_T[:3,:3] = object_T[:3,:3] @ gen_grasp_offset[:3,:3]
    # object_T[:3, 3] = gen_grasp_offset[:3, :3] @ object_T[:3, 3]
    object_mesh.apply_transform(object_T)

    for i_sample in range(args.num):
        cmap_ood_sample = {'object_name': object_name,
                            'i_sample': i_sample,
                            'object_point_cloud': None,
                            'contact_map_value': None}
        
        object_point_cloud, faces_indices = trimesh.sample.sample_surface(mesh=object_mesh, count=2048)
        contact_points_normal = torch.tensor([object_mesh.face_normals[x] for x in faces_indices]).float()
        object_point_cloud = torch.Tensor(object_point_cloud).float()
        object_point_cloud = torch.cat([object_point_cloud, contact_points_normal], dim=1).to(device)
        z_latent_code = torch.randn(1, model.latent_size, device=device).float()
        contact_map_value = model.inference(object_point_cloud[:, :3].unsqueeze(0), z_latent_code).squeeze(0)
        
        contact_map_value = contact_map_value.detach().cpu().unsqueeze(1)
        contact_map_value = pre_process_contact_map_goal(contact_map_value).to(device)
        contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1)

        cmap_ood_sample['object_point_cloud'] = object_point_cloud
        cmap_ood_sample['contact_map_value'] = contact_map_value
        
        vis_data = []
        vis_data += [plot_point_cloud_cmap(contact_map_goal[:, :3].cpu().detach().numpy(),
                                            contact_map_goal[:, 6].cpu().detach().numpy())]
        vis_data += [plot_mesh(object_mesh)]
        fig = go.Figure(data=vis_data)

        fig.write_html(os.path.join( f'{logs_basedir}/{i_sample}.html'))
        torch.save(cmap_ood_sample, os.path.join(logs_basedir, f'{i_sample}.pt'))
