import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

from paradex.utils.file_io import rsc_path

color_list = ["red", "yellow", "brown"]
mesh_offset = {
    "red": np.eye(4),
    "yellow": np.eye(4),
    "brown": np.eye(4)
}
# 모든 mesh를 담을 리스트
meshes = []

rotation = R.from_euler('zyx', [-125, -6, 1], degrees=True)
mesh_offset["red"][:3, :3] = rotation.as_matrix()
mesh_offset["red"][:3, 3] = np.array([0.014, 0.0, -0.053])

rotation = R.from_euler('zyx', [0, 0, -0.5], degrees=True)
mesh_offset["yellow"][:3, :3] = rotation.as_matrix()
mesh_offset["yellow"][:3, 3] = np.array([0.002, -0.002, -0.055])

rotation = R.from_euler('zyx', [-30, 0, 1.1], degrees=True)
mesh_offset["brown"][:3, :3] = rotation.as_matrix()
mesh_offset["brown"][:3, 3] = np.array([0.0, 0.003, -0.055])

for color in color_list:
    mesh_path = os.path.join(rsc_path, "object", f"{color}_ramen_von", f"{color}_ramen_von.obj")
    mesh_dir = os.path.join(rsc_path, "object", f"{color}_ramen_von")
    mesh_path = os.path.join(mesh_dir, f"{color}_ramen_von.obj")
    
    # 현재 디렉토리를 mesh가 있는 곳으로 변경 (상대 경로 해결)
    original_dir = os.getcwd()
    os.chdir(mesh_dir)
    
    # mesh 로드 (MTL과 텍스처가 자동으로 로드됨)
    mesh = o3d.io.read_triangle_mesh(f"{color}_ramen_von.obj", enable_post_processing=True)

    #mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh_orig = deepcopy(mesh)

    mesh.transform(mesh_offset[color])
    # if color == "red":
    #     mesh.paint_uniform_color(np.array([1.0, 0.0, 0.0]))
    # elif color == "yellow":
    #     mesh.paint_uniform_color(np.array([1.0, 1.0, 0.0]))
    # elif color == "brown":
    #     mesh.paint_uniform_color(np.array([0.6, 0.3, 0.0]))
    aabb = mesh.get_axis_aligned_bounding_box()
    print(f"\n=== Transformed {color}_ramen_von.obj (x-axis 7°) ===")
    print(f"AABB min: {aabb.min_bound}")
    print(f"AABB max: {aabb.max_bound}")
    print(f"AABB center: {aabb.get_center()}")
    print(f"AABB extent (size): {aabb.get_extent()}")
          
    # mesh에 노멀 계산 (조명 효과를 위해)
    mesh.compute_vertex_normals()
    # mesh_orig.compute_vertex_normals()

    meshes.append(mesh)
    # meshes.append(mesh_orig)
    print(f"Loaded: {color}_ramen_von.obj")

# 좌표축 생성 (크기를 크게 설정)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2,  # 축의 크기 (필요에 따라 조절)
    origin=[0, 0, 0],  # 원점
)

# 시각화 (mesh들 + 좌표축)
print("\nVisualizing meshes with coordinate frame...")
o3d.visualization.draw_geometries(
    meshes + [coordinate_frame],
    window_name="Ramen Meshes",
    width=1024,
    height=768,
    left=50,
    top=50
)

for color, offset in mesh_offset.items():
    print(color, offset)