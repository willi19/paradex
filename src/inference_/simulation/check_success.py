import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from paradex.utils.file_io import rsc_path
from paradex.utils.metric import compute_mesh_to_ground_distance
import open3d as o3d

demo_list = os.listdir("grasp/state")
obj_mesh_path = os.path.join(rsc_path, "bottle", "bottle.obj")
obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)

arrows = []

for demo_name in demo_list:
    print(f"Processing {demo_name}...")
    traj_list = pickle.load(open(os.path.join("grasp", "state", demo_name), "rb"))
    traj_info = pickle.load(open(os.path.join("grasp", "policy", demo_name), "rb"))

    for traj, ti in zip(traj_list, traj_info):
        obj_traj = traj['object']

        h_last = compute_mesh_to_ground_distance(obj_traj[-1], obj_mesh)
        h_init = compute_mesh_to_ground_distance(obj_traj[0], obj_mesh)

        tx = ti['tx'] / 100
        ty = ti['ty'] / 100
        theta = ti['theta'] * np.pi / 180  # Convert to radians

        success = (h_last - h_init > 0.05)
        color = 'green' if success else 'red'

        # 시계 방향으로 90도 회전: x ↔ y, dx ↔ dy
        dx = -np.sin(theta) * 0.025
        dy = np.cos(theta) * 0.025

        start_x = tx - dx * 2 
        start_y = ty - dy * 2 

        arrows.append((start_y, start_x, dy, dx, color))  # x <-> y로 바꿈

    # 시각화
    plt.figure(figsize=(6, 6))
    for x, y, dx, dy, color in arrows:
        plt.arrow(x, y, dx, dy, head_width=0.005, head_length=0.01, fc=color, ec=color)

    plt.gca().set_aspect('equal')
    plt.xlabel("ty (horizontal)")
    plt.ylabel("tx (vertical)")
    plt.title("Pick Success (green) / Fail (red)")
    plt.grid(True)
    plt.savefig(f"grasp/success_{demo_name}.png")
    plt.close()
    arrows.clear()  # Clear arrows for the next demo
