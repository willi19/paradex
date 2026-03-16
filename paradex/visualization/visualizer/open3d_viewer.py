import numpy as np
import copy
import cv2
import trimesh
import time
from pathlib import Path
from tqdm.auto import tqdm
import open3d as o3d
import yourdfpy

from paradex.robot.robot_wrapper import RobotWrapper

class Open3DVideoRenderer:
    def __init__(
        self,
        obj_mesh,
        obj_T,
        urdf_path,
        qpos,
        c2r=None,
        object_nm="object",
        width=1280,
        height=720,
        fps=30
    ):
        self.obj_T = obj_T
        self.mesh = copy.deepcopy(obj_mesh)
        self.object_nm = object_nm
        self.width = width
        self.height = height
        self.fps = fps
        self.urdf_path = urdf_path

        if c2r is None:
            self.c2r = np.eye(4)
        else:
            self.c2r = c2r

        self.robot = RobotWrapper(urdf_path)
        self.qpos = qpos
        
        self.robot_mesh = self.load_mesh()
        
        self.num_frames = min(self.qpos.shape[0], self.obj_T.shape[0])
        
        # 지오메트리 관리를 위한 변수
        self.current_robot_parts = 0
        
        # Open3D 렌더러 설정
        self.setup_renderer()
        
    def setup_renderer(self):
        """Open3D 오프스크린 렌더러 설정"""
        # 오프스크린 렌더러 생성
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        
        # 재질 설정
        self.setup_materials()
        
        # 조명 설정
        self.setup_lighting()
        
    def setup_materials(self):
        """재질 설정"""
        # 기본 재질
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.base_color = [0.8, 0.8, 0.8, 1.0]
        self.mat.shader = "defaultLit"
        
        # 로봇 재질
        self.robot_mat = o3d.visualization.rendering.MaterialRecord()
        self.robot_mat.base_color = [0.7, 0.7, 0.9, 1.0]
        self.robot_mat.shader = "defaultLit"
        
        # 바닥 재질
        self.floor_mat = o3d.visualization.rendering.MaterialRecord()
        self.floor_mat.base_color = [0.5, 0.5, 0.5, 1.0]
        self.floor_mat.shader = "defaultLit"
        
    def setup_lighting(self):
        self.renderer.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.MED_SHADOWS,
            np.array([0.0, -1.0, -1.0])
        )
        
    def trimesh_to_o3d(self, trimesh_mesh):
        """Trimesh를 Open3D 메시로 변환"""
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
        
        # 노말 계산
        o3d_mesh.compute_vertex_normals()
        
        return o3d_mesh
    
    def load_mesh(self):
        urdf_ = yourdfpy.URDF.load(self.urdf_path)
        meshes = {}
        
        for link_name, link in urdf_.link_map.items():
            if link.visuals:
                for visual in link.visuals:
                    if visual.geometry.mesh:
                        # 메시 로드
                        try:
                            mesh_path = visual.geometry.mesh.filename
                            if not Path(mesh_path).is_absolute():
                                urdf_dir = Path(self.urdf_path).parent
                                mesh_path = urdf_dir / mesh_path
                            
                            mesh = trimesh.load(mesh_path)
                            
                            # 스케일 적용
                            if visual.geometry.mesh.scale is not None:
                                scale_matrix = np.diag(list(visual.geometry.mesh.scale) + [1])
                                mesh.apply_transform(scale_matrix)
                            
                            # 비주얼 변환 적용
                            if visual.origin is not None:
                                mesh.apply_transform(visual.origin)
                            
                            meshes[link_name] = mesh
                            
                        except Exception as e:
                            print(f"Warning: Could not load mesh for link {link_name}: {e}")
                            
                    elif visual.geometry.box:
                        # 박스 지오메트리 생성
                        size = visual.geometry.box.size
                        box_mesh = trimesh.creation.box(size)
                        
                        if visual.origin is not None:
                            box_mesh.apply_transform(visual.origin)
                        meshes[link_name] = box_mesh
                        
                    elif visual.geometry.cylinder:
                        # 실린더 지오메트리 생성
                        radius = visual.geometry.cylinder.radius
                        height = visual.geometry.cylinder.length
                        cylinder_mesh = trimesh.creation.cylinder(radius=radius, height=height)
                        
                        if visual.origin is not None:
                            cylinder_mesh.apply_transform(visual.origin)
                        meshes[link_name] = cylinder_mesh
        
        return meshes
        
    def add_floor(self):
        """바닥 추가"""
        floor_size = 2.0
        floor_mesh = trimesh.creation.box([floor_size, floor_size, 0.02])
        floor_mesh.apply_translation([0, 0, -0.041])
        
        o3d_floor = self.trimesh_to_o3d(floor_mesh)
        self.renderer.scene.add_geometry("floor", o3d_floor, self.floor_mat)
        
    def add_coordinate_frame(self):
        """좌표계 프레임 추가"""
        # 좌표축 생성 (짧게)
        axis_length = 0.05
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)
        
        # c2r 변환 적용
        coordinate_frame.transform(self.c2r)
        
        self.renderer.scene.add_geometry("coordinate_frame", coordinate_frame, self.mat)
        
    def render_frame(self, timestep):
        # Remove previous robot parts
        for i in range(self.current_robot_parts):
            if self.renderer.scene.has_geometry(f"robot_part_{i}"):
                self.renderer.scene.remove_geometry(f"robot_part_{i}")

        # Remove previous object
        if self.renderer.scene.has_geometry(self.object_nm):
            self.renderer.scene.remove_geometry(self.object_nm)
            
        self.robot.compute_forward_kinematics(self.qpos[timestep])
        robot_meshes = []
        for link_name, mesh in self.robot_mesh.items():
            mesh_ = copy.deepcopy(mesh)
            mesh_.apply_transform(self.robot.get_link_pose(self.robot.get_link_index(link_name)))
            robot_meshes.append(mesh_)
            
        self.current_robot_parts = len(robot_meshes)  # 현재 파트 수 업데이트
        
        for i, robot_mesh in enumerate(robot_meshes):
            o3d_robot_mesh = self.trimesh_to_o3d(robot_mesh)
            self.renderer.scene.add_geometry(f"robot_part_{i}", o3d_robot_mesh, self.robot_mat)
            
        # 오브젝트 메시 추가
        transformed_obj_mesh = copy.deepcopy(self.mesh).apply_transform(self.obj_T[timestep])
        o3d_obj_mesh = self.trimesh_to_o3d(transformed_obj_mesh)
        self.renderer.scene.add_geometry(self.object_nm, o3d_obj_mesh, self.mat)
        
        # 렌더링
        img = self.renderer.render_to_image()
        return np.asarray(img)
        
    def set_camera_view(self, eye=[1.5, 1.5, 1.5], center=[0, 0, 0.15], up=[0, 0, 1]):
        """카메라 뷰 설정"""
        self.renderer.setup_camera(90, center, eye, up)
        
    def render_video(self, output_path="rendered_video_o3d.mp4", camera_eye=[1.5, 1.5, 1.5], logger=None):
        if logger is None:
            logger = []

        def _log(msg, msg_type="process_msg"):
            logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": msg, "type": msg_type})

        _log(f"Starting Open3D video rendering...")
        _log(f"Frames: {self.num_frames}, Resolution: {self.width}x{self.height}, FPS: {self.fps}")

        self.set_camera_view(eye=np.array(camera_eye))
        self.add_floor()
        self.add_coordinate_frame()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))

        try:
            for timestep in tqdm(range(self.num_frames), desc="Rendering frames with Open3D"):
                img = self.render_frame(timestep)

                if img is None or img.size == 0:
                    _log(f"Warning: Frame {timestep} is empty!", "process_error")
                    img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                # Write directly to video (RGB -> BGR)
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

                if timestep % 50 == 0:
                    _log(f"Progress: {timestep / self.num_frames * 100:.1f}% ({timestep}/{self.num_frames})", "process_cnt")

            out.release()

            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                file_size = Path(output_path).stat().st_size
                _log(f"Video saved: {output_path} ({file_size / (1024*1024):.2f} MB, {self.num_frames} frames, {self.num_frames / self.fps:.2f}s)")
            else:
                _log("Video file is empty or doesn't exist", "process_error")

        except Exception as e:
            out.release()
            _log(f"Error during rendering: {e}", "process_error")
            import traceback
            traceback.print_exc()
        
            
