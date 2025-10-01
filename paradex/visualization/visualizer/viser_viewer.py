"""
env: hoifield_310
play /stop / next frame/ previous frame
slider 
floor vis (on/off)
debugline / debugpoint / debug sphere (on/off)
rendering the current 3D scene into images
Visualizing the 3D scene changes during optimization
"""

import sys, os
import time
from pathlib import Path
from typing import List
import numpy as np
import copy
from scipy.spatial import cKDTree
import cv2
import pickle
import trimesh
import argparse
from tqdm.auto import tqdm
import imageio.v3 as iio
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf
import yourdfpy

import open3d as o3d
from paradex.utils.file_io import rsc_path
from paradex.visualization_.robot_module import Robot_Module
from paradex.robot.mimic_joint import parse_inspire

class ViserViewer:
    def __init__(
        self,
        obj_mesh,
        obj_T,
        urdf_path,
        qpos,
        c2r=None,
        object_nm="object"):
            
        self.obj_T = obj_T
        self.mesh = copy.deepcopy(obj_mesh)
        self.object_nm = object_nm
        
        if c2r is None:
            self.c2r = np.eye(4)
        else:
            self.c2r = c2r
            
        self.load_server()
        
        urdf = yourdfpy.URDF.load(urdf_path)
        self.urdf = urdf
        self.urdf_vis = ViserUrdf(self.server, urdf)
        self.qpos = qpos
        
        self.num_frames = min(self.qpos.shape[0], self.obj_T.shape[0])

        self.add_frames()
        self.add_player()
        self.prev_timestep = 0
        self.add_initial_meshes()
    
    def load_server(self):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(dark_mode=False)
        
        self.server.scene.set_up_direction(self.c2r[:3,2])
        self.server.scene.world_axes

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            near_slider = client.gui.add_slider(
                "Near", min=0.01, max=10.0, step=0.001, initial_value=client.camera.near
            )
            far_slider = client.gui.add_slider(
                "Far", min=1, max=20.0, step=0.001, initial_value=client.camera.far
            )

            @near_slider.on_update
            def _(_) -> None:
                client.camera.near = near_slider.value

            @far_slider.on_update
            def _(_) -> None:
                client.camera.far = far_slider.value
                
    def add_player(self):
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Timestep",
                min=0,
                max=self.num_frames - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            self.gui_next_frame = self.server.gui.add_button("Next Frame", disabled=True)
            self.gui_prev_frame = self.server.gui.add_button("Prev Frame", disabled=True)
            self.gui_playing = self.server.gui.add_checkbox("Playing", True)
            self.render_png = self.server.gui.add_checkbox("Render to PNG", False)
            self.gui_framerate = self.server.gui.add_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=10
            )
            gui_framerate_options = self.server.gui.add_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

            gui_up = self.server.gui.add_vector3(
                "Up Direction",
                initial_value=(0.0, -1.0, 1.0),
                step=0.01,
            )
            
        with self.server.gui.add_folder("Scene"):
            self.floor_visible = self.server.gui.add_checkbox("Show Floor", True)
            self.floor_size = self.server.gui.add_slider(
                "Floor Size", min=0.2, max=1.0, step=0.5, initial_value=0.5
            )
            self.grid_visible = self.server.gui.add_checkbox("Show Grid", True)
            
        with self.server.gui.add_folder("Video Rendering"):
            self.video_width = self.server.gui.add_number("Video Width", initial_value=1280, min=640, max=3840)
            self.video_height = self.server.gui.add_number("Video Height", initial_value=720, min=480, max=2160)
            self.video_fps = self.server.gui.add_slider("Video FPS", min=10, max=60, step=1, initial_value=30)
            self.render_video_btn = self.server.gui.add_button("Render Full Video")
            
        @gui_up.on_update
        def _(_) -> None:
            self.server.scene.set_up_direction(gui_up.value)
            
        @self.floor_visible.on_update
        def _(_) -> None:
            self.update_floor()
            
        @self.floor_size.on_update
        def _(_) -> None:
            self.update_floor()
            
        @self.grid_visible.on_update
        def _(_) -> None:
            self.update_floor()

        @self.gui_timestep.on_update
        def _(_) -> None:
            self.update_scene(self.gui_timestep.value)
            
        @gui_framerate_options.on_click
        def _(_) -> None:
            self.gui_framerate.value = int(gui_framerate_options.value)

        @self.gui_next_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames

        @self.gui_prev_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % self.num_frames

        @self.gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = self.gui_playing.value
            self.gui_next_frame.disabled = self.gui_playing.value
            self.gui_prev_frame.disabled = self.gui_playing.value
            
        @self.render_video_btn.on_click
        def _(_) -> None:
            self.render_full_video()

    def add_frames(self):
        self.server.scene.add_frame(
            "/frames",
            position=(0, 0, 0),
            show_axes=False,
        )

        self.server.scene.add_frame(
            "/system",
            wxyz=tf.SO3.from_matrix(self.c2r[:3,:3]).wxyz,
            position=self.c2r[:3,3],
            show_axes=True,
            axes_radius=0.001,  # 기존 0.002에서 0.001로 축 두께 감소
            axes_length=0.05    # 축 길이를 0.05로 명시적 설정 (기본값보다 짧게)
        )

        self.frame_nodes: list[viser.FrameHandle] = []
        
    def update_floor(self):
        try:
            self.server.scene.remove_by_name("floor")
        except:
            pass
        
        # for i in range(100):
        #     try:
        #         self.server.scene.remove_by_name(f"grid_x_{i}")
        #         self.server.scene.remove_by_name(f"grid_y_{i}")
        #     except:
        #         pass
        
        # if self.floor_visible.value:
        #     size = self.floor_size.value
            
        #     self.server.scene.add_box(
        #         name="floor",
        #         dimensions=(size * 2, size * 2, 0.02),
        #         position=(0.0, 0.0, -0.041),
        #         color=(0.7, 0.7, 0.7)
        #     )
            
        #     if self.grid_visible.value:
        #         self.add_grid_lines(size=size)

    def add_grid_lines(self, size=1.0):
        grid_spacing = 0.01
        lines_added = 0
        
        for y in np.arange(-size, size + grid_spacing, grid_spacing):
            self.server.scene.add_spline_catmull_rom(
                f"grid_x_{lines_added}",
                positions=np.array([[-size, y, 0.001], [size, y, 0.001]]),
                color=(0.4, 0.4, 0.4),
                line_width=1.0
            )
            lines_added += 1
            
        for x in np.arange(-size, size + grid_spacing, grid_spacing):
            self.server.scene.add_spline_catmull_rom(
                f"grid_y_{lines_added}",
                positions=np.array([[x, -size, 0.001], [x, size, 0.001]]),
                color=(0.4, 0.4, 0.4),
                line_width=1.0
            )
            lines_added += 1
        
    def add_initial_meshes(self):
        self.urdf_vis.update_cfg(self.qpos[0])
        
        initial_obj_mesh = copy.deepcopy(self.mesh).apply_transform(self.obj_T[0])
        self.server.scene.add_mesh_trimesh(
            name=self.object_nm,
            mesh=initial_obj_mesh,
            position=(0.0, 0.0, 0.0),
        )
        
        self.update_floor()
    
    def update_scene(self, timestep):
        with self.server.atomic():
            self.urdf_vis.update_cfg(self.qpos[timestep])
            
            transformed_obj_mesh = copy.deepcopy(self.mesh).apply_transform(self.obj_T[timestep])
            try:
                self.server.scene.remove_by_name(self.object_nm)
            except:
                pass
            self.server.scene.add_mesh_trimesh(
                name=self.object_nm,
                mesh=transformed_obj_mesh,
                position=(0.0, 0.0, 0.0),
            )
            
        self.prev_timestep = timestep
        self.server.flush()
        
        if self.render_png.value:
            self.render_current_frame(timestep)
    
    def render_current_frame(self, timestep):
        render_dir = Path("rendered")
        render_dir.mkdir(exist_ok=True)
        
        for _, client in self.server.get_clients().items():
            rendered_img = client.get_render(height=720*2, width=1280*2)
            output_path = render_dir / f'{timestep:05d}.jpeg'
            cv2.imwrite(str(output_path), rendered_img)
    
    def render_full_video(self):
        """모든 프레임을 렌더링하고 동영상으로 저장"""
        print("Starting full video rendering...")
        
        # 렌더링 디렉토리 생성
        render_dir = Path("video_render")
        render_dir.mkdir(exist_ok=True)
        
        # 기존 이미지 파일들 삭제
        for img_file in render_dir.glob("*.png"):
            img_file.unlink()
        
        # 현재 재생 상태 저장
        was_playing = self.gui_playing.value
        current_timestep = self.gui_timestep.value
        
        # 재생 중지
        self.gui_playing.value = False
        
        # 클라이언트 가져오기
        clients = list(self.server.get_clients().values())
        if not clients:
            print("No clients connected for rendering!")
            return
            
        client = clients[0]  # 첫 번째 클라이언트 사용
        
        frames = []
        try:
            # 모든 프레임 렌더링
            for timestep in tqdm(range(self.num_frames), desc="Rendering frames"):
                # 콘솔에 진행률 표시
                if timestep % 10 == 0:  # 10프레임마다 출력
                    progress_percent = (timestep / self.num_frames) * 100
                    print(f"Rendering progress: {progress_percent:.1f}% ({timestep}/{self.num_frames})")
                
                # 씬 업데이트
                self.update_scene(timestep)
                time.sleep(0.1)  # 씬 업데이트 대기
                
                # 프레임 렌더링
                rendered_img = client.get_render(
                    height=int(self.video_height.value), 
                    width=int(self.video_width.value)
                )
                
                # BGR을 RGB로 변환 (OpenCV는 BGR, imageio는 RGB 사용)
                rendered_img_rgb = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
                frames.append(rendered_img_rgb)
                
                # 개별 프레임도 저장 (디버깅용)
                output_path = render_dir / f'frame_{timestep:05d}.png'
                cv2.imwrite(str(output_path), rendered_img)
            
            # 동영상 생성
            video_path = Path("rendered_video.mp4")
            print(f"Creating video: {video_path}")
            
            iio.imwrite(
                video_path,
                frames,
                fps=self.video_fps.value,
                codec='libx264',
                quality=8,  # 높은 품질
                pixelformat='yuv420p'  # 호환성을 위해
            )
            
            print(f"Video saved successfully: {video_path}")
            print(f"Total frames: {len(frames)}")
            print(f"Video duration: {len(frames) / self.video_fps.value:.2f} seconds")
            
        except Exception as e:
            print(f"Error during video rendering: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 원래 상태 복원
            self.gui_playing.value = was_playing
            self.gui_timestep.value = current_timestep
            self.update_scene(current_timestep)
            print("Video rendering completed!")
        
    def update(self):
        if self.gui_playing.value:
            next_timestep = (self.gui_timestep.value + 1) % self.num_frames
            self.gui_timestep.value = next_timestep
            
        time.sleep(1.0 / self.gui_framerate.value)
    
    def start_viewer(self):
        try:
            while True:
                self.update()
        except KeyboardInterrupt:
            pass