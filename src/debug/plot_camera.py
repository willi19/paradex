import open3d as o3d
import numpy as np
import json
import os

def draw_camera_pyramid(extrinsic_matrix, scale=0.1):
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
        [-0.5, -0.5, 1],  # Bottom-left
        [0.5, -0.5, 1],   # Bottom-right
        [0.5, 0.5, 1],    # Top-right
        [-0.5, 0.5, 1],   # Top-left
        [0, 0, 0]         # Apex (camera position)
    ]) * scale
    
    # Transform pyramid points to world coordinates
    pyramid_points = (np.linalg.inv(extrinsic_matrix[:3, :3]) @ pyramid_points.T).T + camera_center
    
    # Define pyramid edges
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Base edges
        [0, 4], [1, 4], [2, 4], [3, 4]   # Side edges to apex
    ]
    
    # Define colors for lines
    colors = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],  # Base edges in red
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]  # Side edges in green
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pyramid_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Create a sphere at the camera center
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.2)
    sphere.translate(camera_center)
    sphere.paint_uniform_color([0, 0, 1])  # Blue sphere
    
    return line_set, sphere

# Load extrinsic parameters
calib_dir = "/home/temp_id/shared_data/cam_param/20250313145857"
extrinsic = json.load(open(os.path.join(calib_dir, "extrinsics.json"), "r"))

o3d_visuals = []
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
o3d_visuals.append(coordinate_frame)  # Adding a reference frame

c2r = np.load("/home/temp_id/shared_data/handeye_calibration/20250313_193343/0/C2R.npy")
if c2r.shape == (3, 4):
    c2r = np.vstack((c2r, [0, 0, 0, 1]))
c2r_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
c2r_frame.transform(c2r[:4, :4])  # Ensure only 4x4 matrix is used

o3d_visuals.append(c2r_frame)

for serial_num, ext_mat in extrinsic.items():
    extrinsic_matrix = np.array(ext_mat)
    camera_pyramid, sphere = draw_camera_pyramid(extrinsic_matrix, scale=0.1)
    o3d_visuals.append(camera_pyramid)
    o3d_visuals.append(sphere)

# Display the 3D visualization
o3d.visualization.draw_geometries(o3d_visuals)
