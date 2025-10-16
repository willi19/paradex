import numpy as np
import os

from paradex.utils.file_io import shared_dir, find_latest_directory

OBSTACLE = {'cuboid': 
                { # xyz, quaternion
                 'table': {'dims': [2, 2, 0.2], 'pose': [0,0,-0.12, \
                                           0,0,0,1]}, 
                 # 'baseback': {'dims': [2.0, 0.5, 2.0], 'pose': [-1.0857807924568896, -0.011288158965621076, -0.015956839973832793, 0.7082079218969054, -0.00040869377511796283, -0.006448498134229638, 0.7059743544943244]}, 
                 'basetop': {'dims': [5.0, 5.0, 0.2], 'pose': [0, 0, 1.0, 0, 0, 0, 1]}, 
                 'shelf0': {'dims': [0.8, 0.33, 2.02], 'pose': [-0.68+0.33/2, -0.6+0.8/2, -0.76, 0.70710678, 0, 0, 0.70710678]}, 
                 'shelf1': {'dims': [0.8, 0.03, 2.06], 'pose': [-0.68+0.03/2+0.33, -0.6+0.8/2, -0.75, 0.70710678, 0, 0, 0.70710678]}, # + 0.1
                 'shelf2': {'dims': [0.8, 0.1, 1.0], 'pose': [-0.68-0.1/2, -0.6+0.8/2, 0.2541, 0.70710678, 0, 0, 0.70710678]},  # + 1.0141
                 'shelf3': {'dims': [0.8, 0.33, 0.05], 'pose': [-0.68+0.33/2, -0.6+0.8/2, 0.574, 0.70710678, 0, 0, 0.70710678]},  #+ 1.334
                 # 'table': {'dims': [5.0, 5.0, 5.0], 'pose': [-0.07808975157119691, -0.5062144110803565, -2.584682669305668, 0.6999402146008835, 0.004682160214565101, -0.0007793753508808123, -0.7141856662901159]}}
                 }
            }

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
                      [0, 0, 0, 1]])
}


# asdf I should have split the pick & pladce trajectory

def load_visualizer(pick_position):
    visualizer = ViserViewer(up_direction=np.array([0,0,1]))

    visualizer.add_floor()
    visualizer.add_robot("xarm", get_robot_urdf_path(arm_name="xarm", hand_name="inspire"))

    mesh_dict = {}
    for color in ["brown", "red", "yellow"]:
        mesh_path = os.path.join(rsc_path, "object", f"{color}_ramen_von", f"{color}_ramen_von.obj")
        mesh = trimesh.load(mesh_path)
        mesh_dict[color] = mesh

    for obj_name, obj_pose in pick_position.items():
        visualizer.add_object(obj_name, mesh_dict[obj_name.split('_')[0]], obj_pose)
    
    # make trimesh objects
    for obj_type, obstacles in OBSTACLE.items():
        if obj_type == 'cuboid':
            for obs_name, obs_data in obstacles.items():
                dims = obs_data['dims']
                pose = obs_data['pose']
                
                # Create box mesh
                box = trimesh.creation.box(extents=dims)
                
                # Create transformation matrix
                obs_T = np.eye(4)
                obs_T[:3, 3] = pose[:3]
                wxyz = [pose[6], pose[3], pose[4], pose[5]]  # Convert to wxyz format
                obs_T[:3, :3] = R.from_quat(wxyz).as_matrix()
                # 그냥 add_object 호출!
                visualizer.add_object(f"obstacle_{obs_name}", box, obs_T)

    return visualizer

def load_pick_position():
    obj_6d_path = os.path.join(shared_dir, 'object_6d', 'data', 'obj_output')
    latest_dir = find_latest_directory(obj_6d_path)
    obj_T = {}

    with open(os.path.join(obj_6d_path, latest_dir, 'obj_T.pkl'), 'rb') as f:
        obj_output = pickle.load(f)
    
    obj_idx = 0
    for obj_type, obj_list in obj_output.items():
        obj_type = obj_type.split('_')[0]  # brown_ramen_1 -> brown
        for obj_name, obj_se3 in obj_list.items():
            obj_se3 = np.linalg.inv(C2R) @ obj_se3 @ ramen_offset[obj_type]

            obj_T[f"{obj_type}_{obj_idx}"] = obj_se3
            obj_idx += 1

    return obj_T

pick_position = load_pick_position()
C2R = load_latest_C2R()