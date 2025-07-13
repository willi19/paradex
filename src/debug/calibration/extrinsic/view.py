import os
from paradex.utils.file_io import shared_dir
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from paradex.geometry.math import rigid_transform_3D
import json
import open3d as o3d

cam_param_dir = os.path.join(shared_dir, "cam_param")
dir = "config"

intrinsic_dict = {}
extrinsic_dict = {}

def draw_camera_pyramid(extrinsic_matrix, scale=0.1, color=(1, 0, 0)):
    """
    Draws a camera as a pyramid in 3D using Open3D.
    
    :param extrinsic_matrix: 3x4 or 4x4 numpy array defining the camera's extrinsic matrix.
    :param scale: Scale factor for the camera size.
    """
    # Convert 3x4 to 4x4 if necessary
    if extrinsic_matrix.shape == (3, 4):
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
    
    # Extract translation (camera center)
    camera_center = - np.linalg.inv(extrinsic_matrix[:3, :3]) @ extrinsic_matrix[:3, 3]
    
    # Define camera pyramid points in camera coordinate system
    pyramid_points = np.array([
        [-0.05, -0.05, 0.1],  # Bottom-left
        [0.05, -0.05, 0.1],   # Bottom-right
        [0.05, 0.05, 0.1],    # Top-right
        [-0.05, 0.05, 0.1],   # Top-left
        [0, 0, 0]         # Apex (camera position)
    ]) * scale * 10
    
    # Transform pyramid points to world coordinates
    pyramid_points = (np.linalg.inv(extrinsic_matrix[:3, :3]) @ pyramid_points.T).T + camera_center
    
    # Define pyramid edges
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Base edges
        [0, 4], [1, 4], [2, 4], [3, 4]   # Side edges to apex
    ]
    
    # Define colors for lines
    colors = [color, color, color, color,
              color, color, color, color]  # Side edges in green
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pyramid_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Create a sphere at the camera center
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.2)
    sphere.translate(camera_center)
    sphere.paint_uniform_color(color)  # Blue sphere
    
    return line_set, sphere

color_list = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 1, 0],
              [1, 0, 1],
              [1, 1, ]]

if __name__ == "__main__":
    name_list = ["20250713_205300", "20250713_192523", "20250713_195450", "20250713_200124", "20250713_205250"]
    point_list = {}
    camera_point = np.array([[0,0,0,1], [0.05,0,0.05,1], [0,0.05,0.05,1], [-0.05,0,0.05,1], [0,-0.05,0.05,1]])
    
    for name in name_list:
        root_dir = os.path.join(cam_param_dir, name)
        intrinsics = json.load(open(os.path.join(root_dir, "intrinsics.json")))
        extrinsics = json.load(open(os.path.join(root_dir, "extrinsics.json")))
        point_list[name] = {}
        extrinsic_dict[name] = {}
        for serial_num, intmat in intrinsics.items():
            if serial_num not in intrinsic_dict.keys():
                intrinsic_dict[serial_num] = {"orig":[], "dist":[]}
            intrinsic_dict[serial_num]["orig"].append(intmat['original_intrinsics'])
            intrinsic_dict[serial_num]["dist"].append(intmat['dist_params'])

        for serial_num, extmat in extrinsics.items():                
            extrinsic_serial = {}
            extrinsic_dict[name][serial_num] = np.array(extmat)
            point_list[name][serial_num] = (np.array(extmat) @ camera_point.T).T
            
    for serial_num, data in intrinsic_dict.items():
        mean = np.mean(data["dist"], axis=0)
        std = np.std(data["dist"], axis=0)
        # print(serial_num, std / mean, len(data["dist"]))
    
    root_name = name_list[0]
    
    for name in name_list[1:]:
        A = []
        B = []
        for serial_name in point_list[root_name].keys():
            if serial_name not in point_list[name]:
                continue
            A.append(point_list[name][serial_name].copy())
            B.append(point_list[root_name][serial_name].copy())
        
        A = np.concatenate(A,  axis=0)
        B = np.concatenate(B, axis=0)
        
        T = rigid_transform_3D(A, B)
        # print(A)
        # print(((T[:3,:3] @ A.T + T[:3,3:]).T - B))
        
        
        for serial_name, ext in extrinsic_dict[name].items():
            extmat = np.eye(4)
            extmat[:3,:] = ext
            extrinsic_dict[name][serial_name] = T @ extmat
            
            root_extmat = np.eye(4)
            root_extmat[:3,:] = extrinsic_dict[root_name][serial_name]
            
    extrinsic_serial = {}
    for name, data in extrinsic_dict.items():
        for serial_num, extmat in data.items():
            if serial_num not in extrinsic_serial.keys():
                extrinsic_serial[serial_num] = []
                
            extrinsic_serial[serial_num].append(extmat[:3,:])

    o3d_visuals = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d_visuals.append(coordinate_frame)  # Adding a reference frame

    for serial_num, data in extrinsic_serial.items():
        data = np.array(data)
        rots = R.from_matrix(data[:,:3,:3])  # (N, 4)
        log_rots = rots.as_rotvec()     # (N, 3) ← log-map: axis × angle
        
        mean_rotvec = np.mean(log_rots, axis=0)
        std_rotvec = np.std(log_rots, axis=0)
        
        mean_trans = np.mean(data[:,:3,3], axis=0)
        std_trans = np.mean(data[:,:3,3], axis=0)
        
        print(serial_num, std_rotvec, std_trans) 
        for i, ext in enumerate(data):
            extrinsic_matrix = np.array(ext)
            camera_pyramid, sphere = draw_camera_pyramid(extrinsic_matrix, scale=0.1, color=color_list[i])
            o3d_visuals.append(camera_pyramid)
            o3d_visuals.append(sphere)
        
    
    o3d.visualization.draw_geometries(o3d_visuals)
