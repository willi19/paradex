import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from dex_robot.utils.file_io_prev import load_obj_traj, load_mesh

# calcuate success rate, object distance, and robot distance


def compute_mesh_to_ground_distance(obj_pose, mesh):
    """
    Compute the minimum distance between the mesh's lowest point and the ground (z=0).

    Parameters:
        obj_traj (list or np.ndarray): Object trajectory where each element contains
                                       the position (x, y, z) of the object.
        mesh (open3d.geometry.TriangleMesh): Mesh representing the object.

    Returns:
        list: A list of distances (minimum z-values relative to the ground).
    """

    # Translate the mesh based on the object's pose
    # mesh_copy = mesh.clone()
    # mesh_copy = mesh_copy.transform(obj_pose)

    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices).copy()

    vertices = obj_pose[:3, :3] @ vertices.T + obj_pose[:3, 3][:, None]

    # Find the minimum z-coordinate (closest point to the ground)
    min_z = np.min(vertices[2, :]) + 0.0525
    distance_to_ground = min_z  # Ensure no negative distances

    distances = distance_to_ground

    return distances


def compute_distance(traj1, traj2):
    """
    Compute the average Euclidean distance between two trajectories.

    Parameters:
        traj1 (np.ndarray): First trajectory of shape (T, 4, 4).
        traj2 (np.ndarray): Second trajectory of shape (T, 4, 4).

    Returns:
        float: Average Euclidean distance between the two trajectories.
    """
    distances = []
    for pose1, pose2 in zip(traj1, traj2):
        if pose1.shape == (4, 4):
            pose1 = pose1[:3, 3]
            pose2 = pose2[:3, 3]
        dist = np.linalg.norm(pose1 - pose2)
        distances.append(dist)
    return np.max(distances)


def is_pick_success(obj_traj, mesh):
    """
    Check if the pick is successful.
    Returns:
        bool: True if the pick is successful, False otherwise.
    """
    heights = []
    T = len(obj_traj)
    for step in range(T):
        # Compute object heights
        h = compute_mesh_to_ground_distance(obj_traj[step], mesh)
        heights.append(h)
    max_height = np.max(heights)
    if max_height > 0.05:
        return True
    return False


# def get_pickplace_timing(height_traj):
#     pick = -1
#     place = -1
#     for i, h in enumerate(height_traj):
#         if h > 0.01 and pick == -1:
#             pick = i
#         if h < 0.01 and pick != -1:
#             place = i
#             break
#     return pick, place

def get_pickplace_timing(obj_traj, mesh):
    height_traj = []
    T = len(obj_traj)
    for step in range(T):
        # Compute object heights
        h = compute_mesh_to_ground_distance(obj_traj[step], mesh)
        height_traj.append(h)

    pick = -1
    place = -1
    for i, h in enumerate(height_traj):
        if h > 0.01 and pick == -1:
            pick = i
        if h < 0.01 and pick != -1:
            place = i
            break
    return pick, place

def get_grasp_timing(obj_traj, obj_name, robot_wrist_traj, robot_traj):
    # obj_name = list(obj_traj.keys())[0]
    T = len(obj_traj)

    mesh = load_mesh(obj_name)
    pick, place = get_pickplace_timing(obj_traj, mesh)

    wrist_pose_obj = [np.linalg.inv(obj_traj[t]) @ robot_wrist_traj[t] for t in range(T)]
    wrist_pose_obj = np.array(wrist_pose_obj)

    grasping_position = np.mean(wrist_pose_obj[pick:place+1, :3, 3], axis=0)

    grasping_pose = robot_traj[pick]
    grasp_end = -1
    grasp_start = -1

    max_dist = 0
    for t in range(pick, place+1):
        dist = np.linalg.norm(robot_traj[t] - grasping_pose)
        max_dist = max(max_dist, dist)
    
    for t in range(pick):
        dist = np.linalg.norm(robot_traj[t] - grasping_pose)
        if dist < max_dist:
            grasp_end = t+1
            break
    for t in range(grasp_end-1, -1, -1):
        dist = np.linalg.norm(wrist_pose_obj[t, :3, 3] - grasping_position)
        if dist > 0.2:
            grasp_start = t
            break
    return grasp_start, grasp_end

if __name__ == "__main__":
    from dex_robot.utils.file_io_prev import load_obj_traj, load_robot_traj
    obj_name = "smallbowl1"
    sim_root_path = f"data/simulation/{obj_name}"  # Replace with your actual path
    sim_demo_path_list = os.listdir(sim_root_path)

    teleop_root_path = f"data/teleoperation/{obj_name}"  # Replace with your actual path
    # teleop_demo_path_list = os.listdir(teleop_root_path)

    mesh = o3d.io.read_triangle_mesh(f"rsc/{obj_name}/{obj_name}.obj")
    success = 0
    obj_distances = []
    robot_distances = []
    for demo_name in sim_demo_path_list:
        sim_demo_path = os.path.join(sim_root_path, demo_name)

        # Load trajectories
        sim_obj_traj = load_obj_traj(sim_demo_path)[obj_name]
        sim_target_traj = load_robot_traj(sim_demo_path)

        teleop_demo_path = os.path.join(teleop_root_path, demo_name)
        teleop_obj_traj = load_obj_traj(teleop_demo_path)[obj_name]
        teleop_target_traj = load_robot_traj(teleop_demo_path)

        if is_pick_success(teleop_obj_traj, mesh):
            success += 1

        # Compute distances to measure sim2real gap
        obj_distance = compute_distance(sim_obj_traj, teleop_obj_traj)
        robot_distance = compute_distance(sim_target_traj, teleop_target_traj)
        obj_distances.append(obj_distance)
        robot_distances.append(robot_distance)

    print(success / len(sim_demo_path_list))
    print(np.mean(obj_distances))
    print(np.mean(robot_distances))
