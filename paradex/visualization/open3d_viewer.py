import numpy as np
import copy
import cv2
import trimesh
import time
from pathlib import Path
from tqdm.auto import tqdm
import imageio.v3 as iio
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
        self.urdf_path = urdf_path  # URDF 경로 저장
        
        if c2r is None:
            self.c2r = np.eye(4)
        else:
            self.c2r = c2r
            
        # URDF 로드
        self.urdf_path = urdf_path
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
        """조명 설정"""
        # 환경광만 사용 (더 안정적)
        self.renderer.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.MED_SHADOWS, 
            np.array([0.0, -1.0, -1.0])
        )
        
        # 점광원은 문제가 있을 수 있으므로 주석처리
        # try:
        #     self.renderer.scene.scene.add_point_light(
        #         "point_light", 
        #         np.array([1.0, 1.0, 1.0], dtype=np.float32),  # color
        #         np.array([2.0, 2.0, 3.0], dtype=np.float32),  # position
        #         100.0,  # intensity
        #         10.0,   # falloff
        #         True    # cast_shadows
        #     )
        # except Exception as e:
        #     print(f"Warning: Could not add point light: {e}")
        #     pass
        
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
                                # 상대 경로인 경우 URDF 파일 위치 기준으로 해석
                                # if hasattr(urdf_, 'filename') and urdf_.filename:
                                urdf_dir = Path(self.urdf_path).parent# Path(urdf_.filename).parent
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
        """단일 프레임 렌더링"""
        # 이전 로봇 파트들 제거
        for i in range(self.current_robot_parts):
            try:
                self.renderer.scene.remove_geometry(f"robot_part_{i}")
            except:
                pass
        
        # 오브젝트 제거
        try:
            self.renderer.scene.remove_geometry(self.object_nm)
        except:
            pass
            
        # 로봇 메시들 추가
        qpos = np.zeros(22)
        qpos[:6] = self.qpos[timestep][:6]
        qpos[:6] = self.qpos[timestep][:6]
        qpos[:6] = self.qpos[timestep][:6]
        
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
        
    def render_video(self, output_path="rendered_video_o3d.mp4", camera_eye=[1.5, 1.5, 1.5], logger=[]):
        """전체 비디오 렌더링"""
        logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Starting Open3D video rendering...", "type": "process_msg"})
        logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Frames: {self.num_frames}, Resolution: {self.width}x{self.height}, FPS: {self.fps}", "type": "process_msg"})
        # 카메라 설정
        self.set_camera_view(eye=np.array(camera_eye))

        # 바닥과 좌표계 추가
        self.add_floor()
        self.add_coordinate_frame()
        
        frames = []
        
        try:
            for timestep in tqdm(range(self.num_frames), desc="Rendering frames with Open3D"):
                # 프레임 렌더링
                
                img = self.render_frame(timestep)
                
                # 디버깅: 이미지 크기와 내용 확인
                if timestep == 0:
                    logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"First frame shape: {img.shape}, dtype: {img.dtype}, min/max: {img.min()}/{img.max()}", "type": "process_msg"})
                
                # 이미지가 유효한지 확인
                if img is None or img.size == 0:
                    logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Warning: Frame {timestep} is empty!", "type": "process_error"})
                    # 빈 프레임 대신 검은 이미지 생성
                    img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                frames.append(img)
                
                
                if timestep % 50 == 0:
                    progress_percent = (timestep / self.num_frames) * 100
                    logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Progress: {progress_percent:.1f}% ({timestep}/{self.num_frames})", "type": "process_cnt"})
            
            if not frames:
                logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": "No frames were generated!", "type": "process_error"})
                return
                
            logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Generated {len(frames)} frames", "type": "process_msg"})
            
            # 비디오 저장 - OpenCV를 먼저 시도 (더 안정적)
            logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Saving video to {output_path}...", "type": "process_msg"})
            
            self.save_video_opencv(frames, output_path, logger, output_path)
                
            # 최종 상태 출력
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                file_size = Path(output_path).stat().st_size
                logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Final video: {output_path}", "type": "process_msg"})
                logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"File size: {file_size / (1024*1024):.2f} MB", "type": "process_msg"})
                logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Total frames: {len(frames)}", "type": "process_msg"})
                logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Video duration: {len(frames) / self.fps:.2f} seconds", "type": "process_msg"})
            else:
                logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": "Final video file is empty or doesn't exist", "type": "process_error"})
                
        except Exception as e:
            logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Error during rendering: {e}", "type": "process_error"})
            import traceback
            traceback.print_exc()
        
            
    def save_video_opencv(self, frames, output_path, logger=[]):
        """OpenCV를 사용한 비디오 저장 (대안)"""
        if not frames:
            logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": "No frames to save", "type": "process_error"})
            return
            
        height, width = frames[0].shape[:2]
        
        # OpenCV VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        for frame in frames:
            # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"Video saved with OpenCV: {output_path}", "type": "process_msg"})
            logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": f"File size: {file_size / (1024*1024):.2f} MB", "type": "process_msg"})
        else:
            logger.append({"root_dir": output_path, "time": time.time(), "state": "processing", "msg": "OpenCV video save also failed", "type": "process_error"})
            