import numpy as np
import trimesh
import os

from paradex.utils.file_io import shared_dir, rsc_path

ramen_offset = {
    "brown":np.array([[1, 0, 0, 0], 
                      [0, 0, 1, -0.055], 
                      [0, -1, 0, 0], 
                      [0, 0, 0, 1]]),
    "red":np.array([[1, 0, 0, 0], 
                      [0, 0, 1, -0.055], 
                      [0, -1, 0, 0], 
                      [0, 0, 0, 1]]),
    "yellow":np.array([[1, 0, 0, 0], 
                      [0, 0, 1, -0.055], 
                      [0, -1, 0, 0], 
                      [0, 0, 0, 1]]),
}

for color in ["brown", "red", "yellow"]:
    obj_name = f"{color}_ramen_von"
    mesh_path = os.path.join(shared_dir, "object_6d/data/mesh", obj_name, f"{obj_name}.obj")
    mesh = trimesh.load(mesh_path)
    
    if ramen_offset[color] is not None:
        mesh.apply_transform(np.linalg.inv(ramen_offset[color]))

    save_path = os.path.join(rsc_path, "object", obj_name, f"{obj_name}.obj")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mesh.export(save_path)

