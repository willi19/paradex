import argparse
import json
import os.path
import time
import sys
import shutil
from paradex.model.AdamGrasp import AdamGrasp
import torch
import plotly.graph_objects as go
from paradex.utils.visualize_plotly import plot_point_cloud, plot_point_cloud_cmap, plot_mesh_from_name
from paradex.model.set_seed import set_global_seed
from torch.utils.tensorboard import SummaryWriter
from paradex.utils.file_io import shared_path
import numpy as np
from paradex.model.get_models import get_handmodel

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_name', type=str)
    parser.add_argument('--index', type=str)

    parser.add_argument('--max_iter', default=1000, type=int)
    parser.add_argument('--steps_per_iter', default=1, type=int)
    parser.add_argument('--num_particles', default=18, type=int)
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--init_rand_scale', default=0.5, type=float)

    parser.add_argument('--energy_func', default='align_dist', type=str)
    args_ = parser.parse_args()
    tag = str(time.time())
    return args_, tag

def visualize_results(object_point_clouds, cmap_values, hand_model, qpos, save_path):
    vis_data = [plot_point_cloud_cmap(object_point_clouds[:, :3].cpu().detach().numpy(),
                                        cmap_values.cpu().detach().numpy())]
    # q[i_vis][:9] = 0
    # q[i_vis][3] = 1
    # q[i_vis][7] = 1
    if qpos is not None:
        vis_data += hand_model.get_plotly_data(q=qpos.cuda().unsqueeze(0), color='pink')

    fig = go.Figure(data=vis_data)
    fig.write_html(save_path)

if __name__ == '__main__':
    set_global_seed(seed=42)
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=8)
    args, time_tag = get_parser()
    
    basedir = os.path.join(shared_path, 'inference', args.object_name, args.index)
    logs_basedir = os.path.join(basedir, 'grasp_pose')

    os.makedirs(logs_basedir, exist_ok=True)

    robot_name = "allegro"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    handmodel = get_handmodel(robot_name, 1, device, hand_scale=1.)

    # init model
    model = AdamGrasp(robot_name="allegro", writer=None, contact_map_goal=None,
                      num_particles=args.num_particles, init_rand_scale=args.init_rand_scale, max_iter=args.max_iter,
                      steps_per_iter=args.steps_per_iter, learning_rate=args.learning_rate, device=device,
                      energy_func_name=args.energy_func)
    
    for filename in os.listdir(os.path.join(basedir, 'contact_map')):
        if not filename.endswith('.pt'):
            continue
        cmap_ind = filename.split('.')[0]
        cmap_dataset = torch.load(f'{basedir}/contact_map/{filename}')
        
        
        object_name = cmap_dataset['object_name']
        object_point_cloud = cmap_dataset['object_point_cloud']
        i_sample = cmap_dataset['i_sample']
        contact_map_value = cmap_dataset['contact_map_value']

        running_name = f'{object_name}+{i_sample}'
        contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1).to(device)
        # record = model.run_adam(object_name=object_name, contact_map_goal=contact_map_goal, running_name=running_name)

        # with torch.no_grad():
        #     q_tra, energy, steps_per_iter = record

        # min_ind = energy.min(dim=0)[1]
        # np.save(os.path.join(logs_basedir, f'{i_sample}.npy'), q_tra[min_ind, -1, :].cpu().numpy())

        q_tra = np.load(os.path.join(shared_path, 'contact_map', f'{args.object_name}', f'{args.index}', 'robot_pose.npy' )).astype(np.float32)
        q_tra = torch.tensor(q_tra).to(device)
        # import pdb; pdb.set_trace()
        visualize_results(object_point_cloud, contact_map_value[:,0], handmodel, q_tra, os.path.join(logs_basedir, f'{i_sample}_ans.html'))