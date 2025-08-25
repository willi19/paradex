import os
from paradex.utils.file_io import shared_dir, handeye_calib_path, load_current_camparam
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
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
    name_list = ["20250812_164341"]# , "20250714_031906", "20250714_031907", "20250714_031908"]
    
    for name in name_list:
        root_dir = os.path.join(handeye_calib_path, name, "0")
        intrinsics = json.load(open(os.path.join(root_dir, "cam_param", "intrinsics.json")))
        extrinsics = json.load(open(os.path.join(root_dir, "cam_param", "extrinsics.json")))
        c2r = np.load(os.path.join(root_dir, "C2R.npy"))
        
        extrinsic_dict[name] = {}
        
        for serial_num, extmat in extrinsics.items():                
            extrinsic_serial = {}
            ext_robot = np.eye(4)
            ext_robot[:3,:] = extmat
            
            extrinsic_dict[name][serial_num] = ext_robot @ c2r
    
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
        tmp_data = []
        for d in data:
            T = np.eye(4)
            T[:3,:] = np.array(d)
            
            tmp_data.append(np.linalg.inv(T))
        
        tmp_data = np.array(tmp_data)
        rots = R.from_matrix(tmp_data[:,:3,:3])  # (N, 4)
        log_rots = rots.as_rotvec()     # (N, 3) ← log-map: axis × angle
        
        mean_rotvec = np.mean(log_rots, axis=0)
        std_rotvec = np.std(log_rots, axis=0)
        
        mean_trans = np.mean(tmp_data[:,:3,3], axis=0)
        std_trans = np.std(tmp_data[:,:3,3], axis=0)
        
        print(serial_num, std_rotvec, std_trans) 
        
        for i, ext in enumerate(data):
            extrinsic_matrix = np.array(ext)
            camera_pyramid, sphere = draw_camera_pyramid(extrinsic_matrix, scale=0.1, color=color_list[i])
            o3d_visuals.append(camera_pyramid)
            o3d_visuals.append(sphere)
        
    
    o3d.visualization.draw_geometries(o3d_visuals)
