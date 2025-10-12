import numpy as np
import pickle
import os
import curobo
# Pick Target 

from paradex.utils.file_io import find_latest_directory, shared_dir

place_origin = np.array([-0.55, -0.45, 0.251 + 0.1]) # 25cm : floor, 10cm: ramen height

Z_OFFSET = np.array([0.0, 0.0, 0.10])
Z_NUM = 2

X_OFFSET = np.array([0.13, 0.0, 0.0])
X_NUM = 2

Y_OFFSET = np.array([0.0, 0.13, 0.0])
Y_NUM = 5

def get_place_position(index):
    """ Get the place position based on the index 
        ORDER: Y -> X -> Z
    """
    z_index = index % Z_NUM
    x_index = (index // Z_NUM) % X_NUM
    y_index = (index // (Z_NUM * X_NUM)) % Y_NUM
    place_position = place_origin + z_index * Z_OFFSET + x_index * X_OFFSET + y_index * Y_OFFSET
    return place_position

def load_pick_position():
    obj_6d_path = os.path.join(shared_dir, 'object_6d', 'data', 'obj_output')
    latest_dir = find_latest_directory(obj_6d_path)

    with open(os.path.join(obj_6d_path, latest_dir, 'obj_T.pkl'), 'rb') as f:
        obj_T = pickle.load(f)

    return obj_T

print(load_pick_position())