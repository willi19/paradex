import xml.dom.minidom
import trimesh
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import yaml

from paradex.utils.file_io import rsc_path, get_robot_urdf_path
from paradex.robot.curobo import to_quat

from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.geom.sphere_fit import SphereFitType

def visualize_meshlist(mesh_list):
    """Open3D를 사용해서 메시들을 시각화"""
    
    # Open3D 시각화 객체 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Robot Meshes", width=1200, height=800)
    
    # 색상 팔레트 생성 (링크별로 다른 색상)
    colors = np.random.rand(len(mesh_list), 3)
    color_idx = 0
    
    all_meshes = []

    for i, tmesh in enumerate(mesh_list):
        try:
            # Trimesh를 Open3D 메시로 변환
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(tmesh.faces)
            
            # 법선 벡터 계산
            o3d_mesh.compute_vertex_normals()
            
            # 색상 설정
            color = colors[i % len(colors)]
            o3d_mesh.paint_uniform_color(color)
            
            # 메시 위치 조정 (겹치지 않게)
            offset = np.array([color_idx * 0.5, 0, 0])  # X축으로 간격두고 배치
            o3d_mesh.translate(offset)
            
            # 시각화에 추가
            vis.add_geometry(o3d_mesh)
            all_meshes.append(o3d_mesh)
            
        except Exception as e:
            print(f"Error processing mesh {i} from link {link_name}: {e}")
    
    color_idx += 1
    
    # 카메라 설정
    vis.get_view_control().set_zoom(0.3)
    vis.get_view_control().set_front([1, 0, 0])
    vis.get_view_control().set_up([0, 0, 1])
    
    print(f"\n총 {len(all_meshes)}개의 메시를 로드했습니다.")
    print("시각화 창에서:")
    print("- 마우스로 회전/줌")
    print("- ESC 또는 창 닫기로 종료")
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    vis.add_geometry(coordinate_frame)
    # 시각화 실행
    vis.run()
    vis.destroy_window()
    
def visualize_meshes(mesh_dict):
    """Open3D를 사용해서 메시들을 시각화"""
    
    # Open3D 시각화 객체 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Robot Meshes", width=1200, height=800)
    
    # 색상 팔레트 생성 (링크별로 다른 색상)
    colors = np.random.rand(len(mesh_dict), 3)
    color_idx = 0
    
    all_meshes = []
    
    for link_name, trimesh_list in mesh_dict.items():
        print(f"Processing link: {link_name} with {len(trimesh_list)} meshes")
        
        for i, tmesh in enumerate(trimesh_list):
            try:
                # Trimesh를 Open3D 메시로 변환
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(tmesh.faces)
                
                # 법선 벡터 계산
                o3d_mesh.compute_vertex_normals()
                
                # 색상 설정
                color = colors[color_idx % len(colors)]
                o3d_mesh.paint_uniform_color(color)
                
                # 메시 위치 조정 (겹치지 않게)
                offset = np.array([color_idx * 0.5, 0, 0])  # X축으로 간격두고 배치
                o3d_mesh.translate(offset)
                
                # 시각화에 추가
                vis.add_geometry(o3d_mesh)
                all_meshes.append(o3d_mesh)
                
            except Exception as e:
                print(f"Error processing mesh {i} from link {link_name}: {e}")
        
        color_idx += 1
    
    # 카메라 설정
    vis.get_view_control().set_zoom(0.3)
    vis.get_view_control().set_front([1, 0, 0])
    vis.get_view_control().set_up([0, 0, 1])
    
    print(f"\n총 {len(all_meshes)}개의 메시를 로드했습니다.")
    print("시각화 창에서:")
    print("- 마우스로 회전/줌")
    print("- ESC 또는 창 닫기로 종료")
    
    # 시각화 실행
    vis.run()
    vis.destroy_window()
    
def visualize_single_robot(mesh_dict):
    """모든 메시를 원래 위치에서 하나의 로봇으로 시각화"""
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Complete Robot", width=1200, height=800)
    
    colors = np.random.rand(len(mesh_dict), 3)
    color_idx = 0
    
    for link_name, trimesh_list in mesh_dict.items():
        for tmesh in trimesh_list:
            try:
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(tmesh.faces)
                o3d_mesh.compute_vertex_normals()
                
                # 링크별로 다른 색상
                color = colors[color_idx % len(colors)]
                o3d_mesh.paint_uniform_color(color)
                
                vis.add_geometry(o3d_mesh)
                
            except Exception as e:
                print(f"Error processing mesh from link {link_name}: {e}")
        
        color_idx += 1
    
    # 좌표축 추가
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coordinate_frame)
    
    vis.run()
    vis.destroy_window()
    
mesh_dict = {}
sph_dict = {}
# Extract links
arm_name = "xarm"
hand_name = "allegro"
xml_path = get_robot_urdf_path(arm_name, hand_name)

root_path = Path(xml_path).parent
spehre_path = root_path / "spheres" / f"{arm_name}_{hand_name}.yml"

sphere_dict = yaml.safe_load(open(spehre_path, 'r'))['collision_spheres']

dom = xml.dom.minidom.parse(xml_path)
links = dom.getElementsByTagName("link")

for link in links:
    link_name = link.getAttribute("name")
    print(link_name)
    if link_name not in ["thumb_distal"]:
        continue
    visuals = link.getElementsByTagName("visual")
    mesh_dict[link_name] = []
    sph_dict[link_name] = []
    
    for i, visual in enumerate(visuals):
        visual_name = visual.getAttribute("name")

        mesh_subpath = visual.getElementsByTagName("geometry")[0].getElementsByTagName("mesh")[0].getAttribute("filename")
        mesh_path = str(root_path/mesh_subpath)
        mesh = trimesh.load(mesh_path)
        mesh_dict[link_name].append(mesh)        
    for sph in sphere_dict.get(link_name, []):
        m = trimesh.creation.icosphere(radius=sph['radius'])
        se3 = np.eye(4)
        se3[:3, 3] = sph['center']
        m.apply_transform(se3)
        mesh_dict[link_name].append(m)
        # mesh_cu = Mesh(name = f"{visual_name}_{i}", vertices=mesh.vertices, faces=mesh.faces, pose=to_quat(np.eye(4)))
        
        # sph = mesh_cu.get_bounding_spheres(
        #     20, 0.05, SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
        # )
        # trimesh_sph = WorldConfig.get_scene_graph(WorldConfig(sphere=sph))
        # for s in sph:
        #     m = trimesh.creation.icosphere(radius=s.radius)
        #     se3 = np.eye(4)
        #     se3[:3, 3] = s.position
        #     m.apply_transform(se3)
        #     # mesh_dict[link_name].append(m)
        
    if len(visuals) > 0:
        visualize_meshlist(mesh_dict[link_name])
# visualize_single_robot(mesh_dict)