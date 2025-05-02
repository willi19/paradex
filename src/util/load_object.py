import os
import json
import pickle
import trimesh as tm

import argparse
from paradex.utils.file_io import shared_path

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='sharp_lift', type=str)
parser.add_argument('--index', required=True, type=str)
args = parser.parse_args()

loss_path = f"{shared_path}/processed/{args.name}/{args.index}/{args.name}_optim/final"
min_loss = 10000
min_loss_idx = -1

for i in range(-1, 20):
    with open(os.path.join(loss_path, f'obj_final_loss_0_{i}.json'), 'r') as f:
        data = json.load(f)
    if data['sil_loss'] < min_loss:
        min_loss = data['sil_loss']
        min_loss_idx = i


pickle_path = os.path.join(loss_path, f'obj_output_after_optim_total_0_{min_loss_idx}.pickle')
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

scale = data['scale'].item()
print(scale)

mesh_orig = tm.load(f'{shared_path}/mesh_before_processed/{args.name}.ply')

mesh_orig.apply_translation(-mesh_orig.center_mass)
mesh_orig.apply_scale(scale)

mesh_orig.export(f'{shared_path}/mesh/{args.name}.ply')
