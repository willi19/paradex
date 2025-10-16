import os
import open3d as o3d
import numpy as np

from paradex.utils.file_io import rsc_path

color_list = ["red"]#, "yellow", "brown"]
mesh_offset = {
    "red": np.eye(4),
    "yellow": np.eye(4),
    "brown": np.eye(4)
}
# 모든 mesh를 담을 리스트
meshes = []

rotation = R.from_euler('x', 10, degrees=True)
mesh_offset["red"][:3, :3] = rotation.as_matrix()

for color in color_list:
    mesh_path = os.path.join(rsc_path, "object", f"{color}_ramen_von", f"{color}_ramen_von.obj")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # mesh에 노멀 계산 (조명 효과를 위해)
    mesh.compute_vertex_normals()
    
    meshes.append(mesh)
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