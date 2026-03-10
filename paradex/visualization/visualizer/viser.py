
import json
import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R
from trimesh import Scene
import viser
import trimesh
from typing import List, Tuple, Dict
import time
import viser.transforms as tf
import threading
import os
from datetime import datetime
import shutil
import subprocess
from scipy.spatial.transform import Slerp
import tempfile 

from paradex.visualization.robot import RobotModule  

class ViserViewer():
    def __init__(self, up_direction=np.array([0,0,1])):
        self.frame_nodes: dict[str, viser.FrameHandle] = {}

        self.up_direction = up_direction
        self.robot_dict = {}
        self.obj_dict = {}

        self.traj_list = []
        self.num_frames = 0

        self.load_server()
        self.add_lights()
        self.add_player()
        # self.add_lights()

    def load_server(self):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(dark_mode=True)

        self.server.scene.set_up_direction(self.up_direction)
        self.server.scene.world_axes

        # @self.server.on_client_connect
        # def _(client: viser.ClientHandle) -> None:
        #     near_slider = client.gui.add_slider(
        #         "Near", min=0.01, max=10.0, step=0.001, initial_value=client.camera.near
        #     )
        #     far_slider = client.gui.add_slider(
        #         "Far", min=1, max=1000.0, step=0.001, initial_value=client.camera.far
        #     )

        #     @near_slider.on_update
        #     def _(_) -> None:
        #         client.camera.near = near_slider.value

        #     @far_slider.on_update
        #     def _(_) -> None:
        #         client.camera.far = far_slider.value

    def add_robot(self, name, urdf_path, pose=None):
        robot = ViserRobotModule(
            target=self.server,
            urdf_path=urdf_path,
            scale=1.0,
            root_node_name=f"/robot/{name}",
            load_meshes=True,
            load_collision_meshes=False,
        )
        if pose is not None:
            if hasattr(robot, '_visual_root_frame'):
                robot._visual_root_frame.position = pose[:3, 3]
                robot._visual_root_frame.wxyz = R.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        
        self.robot_dict[name] = robot

    def add_trimesh(self, name, mesh: trimesh.Trimesh, pose=None):
        obj_T = np.eye(4)
        if pose is not None:
            obj_T = pose
        
        frame_handle = self.server.scene.add_frame(
            f"/objects/{name}_frame",
            position=obj_T[:3, 3],
            wxyz=R.from_matrix(obj_T[:3, :3]).as_quat()[[3, 0, 1, 2]],
            show_axes=False,
            axes_length=0.05,
            axes_radius=0.002,
        )

        mesh_handle = self.server.scene.add_mesh_trimesh(
            name=f"/objects/{name}_frame/{name}",
            mesh=mesh,
            cast_shadow=True,
            receive_shadow=True
        )
        self.obj_dict[name] = {
            'mesh': mesh,
            'frame': frame_handle,
            'transform': obj_T,
            'handle': mesh_handle
        }

        self.frame_nodes[name] = frame_handle

    def add_object(self, name, obj: trimesh.Trimesh, obj_T):
        """
        Add an object mesh to the scene
        
        Args:
            name: Unique name for the object
            obj: trimesh.Trimesh object
            obj_T: 4x4 transformation matrix for the object pose
        """
        # Create a frame for the object
        frame_handle = self.server.scene.add_frame(
            f"/objects/{name}_frame",
            position=obj_T[:3, 3],
            wxyz=R.from_matrix(obj_T[:3, :3]).as_quat()[[3, 0, 1, 2]],
            show_axes=False,
            axes_length=0.05,
            axes_radius=0.002,
        )
        
        # Add mesh to the frame (at origin relative to frame)
        # mesh_handle = self.server.scene.add_mesh_trimesh(
        #         name=f"/objects/{name}_frame/{name}",
        #         mesh=obj
        #     )
        mesh_handle = self.server.scene.add_mesh_simple(
                name=f"/objects/{name}_frame/{name}",
                vertices=obj.vertices,
                faces=obj.faces,
                cast_shadow=True,
                receive_shadow=True
        )
        
        # Store in object dictionary
        self.obj_dict[name] = {
            'mesh': obj,
            'transform': obj_T,
            'frame': frame_handle,
            'handle': mesh_handle
        }
        
        self.frame_nodes[name] = frame_handle

    def add_traj(self, name, robot_traj: Dict, obj_traj: Dict = {}):
        # if len(robot_traj) == 0:
        #     return
        traj_len = max([traj.shape[0] for traj in list(robot_traj.values())] + [traj.shape[0] for traj in list(obj_traj.values())])
        new_traj_dict = {"robot":{}, "object":{}}
        for robot_name in list(self.robot_dict.keys()):
            if robot_name in robot_traj:
                new_traj_dict["robot"][robot_name] = robot_traj[robot_name]

        for obj_name in list(self.obj_dict.keys()):
            if obj_name in obj_traj:
                new_traj_dict["object"][obj_name] = obj_traj[obj_name]
                self.obj_dict[obj_name]['transform'] = obj_traj[obj_name][-1]
            else:
                obj_T = self.obj_dict[obj_name]['transform']
                new_traj_dict["object"][obj_name] = np.tile(obj_T[None, :, :], (traj_len, 1, 1))

        self.traj_list.append((name, new_traj_dict, traj_len))
        self.num_frames += traj_len
        self.gui_timestep.max = self.num_frames - 1
        # print(traj_len, self.num_frames)
    
    def clear_traj(self):
        self.gui_playing.value = False

        self.traj_list = []
        self.num_frames = 0
        self.gui_timestep.max = 1
        self.gui_timestep.value = 0
    
    def add_floor(self, height=0.0):
        self.floor_size = 15.0
        size = self.floor_size
                
        self.floor_box_handle = self.server.scene.add_box(
            name="floor/floor_box",
            dimensions=(size * 2, size * 2, 0.02),  # width, height, thickness
            position=(0.0, 0.0, height-0.01),  # Position slightly below z=0
            color=(0.7, 0.7, 0.7),
            cast_shadow=False,
            receive_shadow=True  # Floor should receive shadows
        )
        
        # Use viser's built-in grid
        self.grid_handles = self.server.scene.add_grid(
            name="floor/grid",
            width=size * 2,
            height=size * 2,
            plane="xy",  # Assuming XY plane at given height
            position=(0.0, 0.0, height),
            cell_size = 0.1
        )
    
    def add_grid(self, size=10.0, cell_size=0.1, height=0.0):
        self.grid_handles = self.server.scene.add_grid(
            name="grid",
            width=size,
            height=size,
            plane="xy",  # Assuming XY plane at given height
            position=(0.0, 0.0, height),
            cell_size = cell_size
        )

    def update_floor(self):
        # Toggle floor visibility
        if hasattr(self, 'floor_box_handle'):
            self.floor_box_handle.visible = not self.floor_visible.value
        
        # Toggle grid visibility
        if hasattr(self, 'grid_handles'):
            self.grid_handles.visible = not self.grid_visible.value
        

    def update_scene(self, timestep):
        # 현재 timestep이 속한 trajectory 찾기
        cumulative_frames = 0
        current_traj = None
        local_timestep = timestep
        
        for traj_name, traj_data, traj_len in self.traj_list:
            if timestep < cumulative_frames + traj_len:
                # 이 trajectory에 속함
                current_traj = traj_data
                local_timestep = timestep - cumulative_frames
                # print(f"Updating scene to timestep {timestep} (trajectory '{traj_name}', local frame {local_timestep})")
                break
            
            cumulative_frames += traj_len
        
        if current_traj is None:
            print(f"Warning: timestep {timestep} out of range")
            return
        
        # 해당 trajectory의 local timestep으로 로봇 업데이트
        with self.server.atomic():
            for robot_name, robot in self.robot_dict.items():
                if robot_name in current_traj["robot"]:
                    robot.update_cfg(current_traj["robot"][robot_name][local_timestep])
            for obj_name, obj in self.obj_dict.items():
                if obj_name in current_traj["object"]:
                    obj_transform = current_traj["object"][obj_name][local_timestep]
                    frame_handle = obj['frame']
                    
                    # Frame의 position과 rotation 업데이트
                    xyzw = R.from_matrix(obj_transform[:3, :3]).as_quat()
                    frame_handle.wxyz = xyzw[[3, 0, 1, 2]]
                    frame_handle.position = obj_transform[:3, 3]

        self.prev_timestep = timestep
        self.server.flush()
        
        if self.render_png.value:
            self.render_current_frame(timestep)

    def update(self):
        if self.gui_playing.value and len(self.traj_list) > 0 and self.num_frames > 0:
            next_timestep = (self.gui_timestep.value + 1) % self.num_frames
            self.gui_timestep.value = next_timestep
            
        time.sleep(1.0 / self.gui_framerate.value)
    
    def start_viewer(self, use_thread=False):
        if use_thread:
            thread = threading.Thread(target=self._viewer_loop, daemon=True)
            thread.start()
        else:
            self._viewer_loop()

    def _viewer_loop(self):
        while True:
            self.update()
    
    def add_player(self):
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Timestep",
                min=0,
                max=self.num_frames -1 if self.num_frames >0 else 1,
                step=1,
                initial_value=0
            )
            self.gui_next_frame = self.server.gui.add_button("Next Frame", disabled=True)
            self.gui_prev_frame = self.server.gui.add_button("Prev Frame", disabled=True)
            self.gui_playing = self.server.gui.add_checkbox("Playing", True)
            self.render_png = self.server.gui.add_checkbox("Render to PNG", False)
            self.gui_framerate = self.server.gui.add_slider(
                "FPS", min=1, max=120, step=0.1, initial_value=10
            )
            gui_framerate_options = self.server.gui.add_button_group(
                "FPS options", ("10", "20", "30", "60", "120")
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
        
        with self.server.gui.add_folder("Capture"):
            self.output_path = self.server.gui.add_text("Output Path", initial_value="")
            self.capture_width = self.server.gui.add_number("Capture Width", initial_value=720, min=256, max=4096)
            self.capture_height = self.server.gui.add_number("Capture Height", initial_value=720, min=256, max=4096)
            self.capture_png_btn = self.server.gui.add_button("Capture Current View (PNG)")

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
            if self.num_frames > 0:
                self.gui_timestep.value = (self.gui_timestep.value + 10) % self.num_frames

        @self.gui_prev_frame.on_click
        def _(_) -> None:
            if self.num_frames > 0:
                self.gui_timestep.value = (self.gui_timestep.value - 1) % self.num_frames

        @self.gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = self.gui_playing.value
            self.gui_next_frame.disabled = self.gui_playing.value
            self.gui_prev_frame.disabled = self.gui_playing.value
            
        @self.render_video_btn.on_click
        def _(_) -> None:
            self.render_full_video()

        @self.capture_png_btn.on_click
        def _(_) -> None:
            import os
            from datetime import datetime
            
            # output_path가 비어있으면 타임스탬프로 생성
            if self.output_path.value.strip() == "":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = f"capture_{timestamp}.png"
            else:
                out_path = self.output_path.value
            
            print(f"Capturing current view to {out_path}...")
            
            # GUI에서 설정한 해상도 사용
            width = int(self.capture_width.value)
            height = int(self.capture_height.value)
            
            # PNG 캡처
            self.capture_scene_png(out_path, height=height, width=width)
            print(f"Saved: {out_path}")

    def add_frame(self, name, T):
        self.frame_nodes[name] = self.server.scene.add_frame(
            name=f"/{name}/frame",
            show_axes=True,
            # axis_length=0.1,
            # axis_radius=0.002,
            axes_length=0.05,
            axes_radius=0.002,
            position=T[:3, 3],
            wxyz=R.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]],
        )

    def add_lights(self):
        # Main directional light (sun-like, from above-front)
        # Creates strong shadows for depth perception
        self.server.scene.add_light_directional(
            name="/system/directional_light",
            color=(255, 255, 255),
            intensity=1.5,
            wxyz=R.from_euler('xyz', [45, 0, 45], degrees=True).as_quat()[[3, 0, 1, 2]],
            cast_shadow=True,
        )

        # Fill light from opposite side (softer, no shadow)
        self.server.scene.add_light_point(
            name="/system/fill_light",
            color=(255, 255, 255),
            intensity=2.0,
            position=(-3.0, -3.0, 3.0),
            cast_shadow=False  # No shadow for fill light
        )

        # Key light (main shadow caster)
        self.server.scene.add_light_point(
            name="/system/key_light",
            color=(255, 255, 255),
            intensity=8.0,
            position=(3.0, 3.0, 4.0),
            cast_shadow=True
        )

        # Rim light (from back-top, for edge definition)
        self.server.scene.add_light_point(
            name="/system/rim_light",
            color=(255, 255, 255),
            intensity=3.0,
            position=(0.0, 0.0, 5.0),
            cast_shadow=False
        )

        # Disable default lights to only use our custom lights
        self.server.scene.enable_default_lights(False)
        
    def add_contact_module(self, robot_name, obj_name):
        from scipy.spatial import cKDTree
        
        self.contact_line_segments = [] 
        self.contact_point_segments = []
        
        with self.server.gui.add_folder("Contact", expand_by_default=False):
            compute_contact_button = self.server.gui.add_button("Compute Contact", disabled=False)
            clear_contact_button = self.server.gui.add_button("Clear Contact", disabled=False)
            drawing_arrow = self.server.gui.add_checkbox("Draw Arrow", True)

            contact_thres = self.server.gui.add_number(
                "Contact Thres",
                initial_value=0.02,
                step=0.005,
            )
        
        @compute_contact_button.on_click        
        def _(_) -> None:
            # Get NN at point
            current_timestep = self.gui_timestep.value
            
            link_T =  self.robot_dict[robot_name].get_link_vertices()
            obj = self.obj_dict[obj_name]['mesh']
            obj_vertices = obj.vertices
            robot_vertices = self.robot_dict[robot_name].get_link_vertices()
            
            obj_tree = cKDTree(obj_vertices)
            
            
        
        @clear_contact_button.on_click
        def _(_) -> None:
            for line in contact_line_segments:
                line.remove()
            contact_line_segments = []
            for point in contact_point_segments:
                point.remove()
            contact_point_segments = []
            
    def add_camera(self, name, extrinsic, intrinsic, color=(0, 255, 0), size=0.1):
        """
        Add a camera frustum visualization to the scene
        
        Args:
            name: Unique name for the camera
            extrinsic: 4x4 or 3x4 camera extrinsic matrix (cam_from_world or world_from_cam)
            intrinsic: Camera intrinsic parameters dict with keys 'fx', 'fy', 'cx', 'cy', 'width', 'height'
                    OR 3x3 intrinsic matrix
            color: RGB tuple (0-255)
            size: Size of the frustum (depth from camera center)
        """
        # Handle extrinsic matrix format
        if extrinsic.shape == (3, 4):
            extrinsic_4x4 = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
        else:
            extrinsic_4x4 = extrinsic
        
        # Convert to world_from_cam if it's cam_from_world
        # Assuming extrinsic is cam_from_world (COLMAP convention), invert it
        world_from_cam = extrinsic_4x4#3 np.linalg.inv(extrinsic_4x4)
        
        # Extract camera position and rotation in world frame
        cam_pos = world_from_cam[:3, 3]
        cam_rot = world_from_cam[:3, :3]
        
        # Parse intrinsic parameters
        if isinstance(intrinsic, dict):
            if 'intrinsics_undistort' in intrinsic:
                K = np.array(intrinsic['intrinsics_undistort'])
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
            else:
                fx = intrinsic.get('fx')
                fy = intrinsic.get('fy')
                cx = intrinsic.get('cx')
                cy = intrinsic.get('cy')
            width = intrinsic.get('width', 640)
            height = intrinsic.get('height', 480)
        else:  # Assume 3x3 matrix
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            width = cx * 2
            height = cy * 2
        
        # Create camera frame
        frame_handle = self.server.scene.add_frame(
            f"/cameras/{name}_frame",
            position=cam_pos,
            wxyz=R.from_matrix(cam_rot).as_quat()[[3, 0, 1, 2]],
            show_axes=True,
            axes_length=size * 0.5,
            axes_radius=size * 0.03,
        )
        
        # Calculate frustum corners in camera space
        # Calculate frustum corners in camera space
        frustum_depth = size
        corners_cam = np.array([
            [(0 - cx) / fx * frustum_depth, (0 - cy) / fy * frustum_depth, frustum_depth],  # top-left
            [(width - cx) / fx * frustum_depth, (0 - cy) / fy * frustum_depth, frustum_depth],  # top-right
            [(width - cx) / fx * frustum_depth, (height - cy) / fy * frustum_depth, frustum_depth],  # bottom-right
            [(0 - cx) / fx * frustum_depth, (height - cy) / fy * frustum_depth, frustum_depth],  # bottom-left
        ])

        # Draw frustum edges (from camera center to corners) - in CAMERA FRAME coordinates
        color_normalized = tuple(c / 255.0 for c in color)
        camera_origin = np.array([0, 0, 0])  # Origin in camera frame

        for i, corner in enumerate(corners_cam):
            self.server.scene.add_spline_catmull_rom(
                f"/cameras/{name}_frame/edge_{i}",
                positions=np.array([camera_origin, corner]),  # Use camera frame coordinates
                color=color_normalized,
                line_width=2.0
            )

        # Draw frustum rectangle (connecting corners) - in CAMERA FRAME coordinates
        for i in range(4):
            self.server.scene.add_spline_catmull_rom(
                f"/cameras/{name}_frame/rect_{i}",
                positions=np.array([corners_cam[i], corners_cam[(i + 1) % 4]]),
                color=color_normalized,
                line_width=2.0
            )

        # Optionally add a small sphere at camera center
        self.server.scene.add_icosphere(
            f"/cameras/{name}_frame/center",
            radius=size * 0.05,
            color=color_normalized,
            position=camera_origin  # At origin of camera frame
        )

        
        return frame_handle

    def change_color(self, name, color, name_list=[]):
        """Change the color of a robot's visualized URDF."""
        if name in self.robot_dict:
            self.robot_dict[name].change_color(name_list, color)
        elif name in self.obj_dict:
            mesh_handle = self.obj_dict[name]['handle']
            mesh_handle.color = tuple(int(c * 255) for c in color)
            if len(color) == 4:
                mesh_handle.opacity = color[3]
        else:
            print(f"Robot '{name}' not found.")    
    
    def add_sphere(self, name, position, radius=0.05, color=(1.0,0,0)):
        self.server.scene.add_icosphere(
            name=f"/spheres/{name}",
            radius=radius,
            color=tuple(int(c * 255) for c in color),
            position=position,
            cast_shadow=True,
            receive_shadow=True
        )

    def add_arrow(self, name, start, end, color=(0,255,0), shaft_radius=0.01, head_radius=0.02, head_length=0.03, opacity=1.0):
        """
        Add an arrow visualization with head
        
        Args:
            name: Unique name for the arrow
            start: Starting position [x, y, z]
            end: Ending position [x, y, z]
            color: RGB color (0-255 range)
            shaft_radius: Radius of arrow shaft
            head_radius: Radius of arrow head cone
            head_length: Length of arrow head cone
            opacity: Opacity value (0.0-1.0)
        """
        from scipy.spatial.transform import Rotation as R
        import trimesh
        
        start = np.array(start)
        end = np.array(end)
        direction = end - start
        total_length = np.linalg.norm(direction)
        direction_norm = direction / total_length
        
        # Shaft length
        shaft_length = total_length - head_length
        shaft_end = end - direction_norm * head_length
        
        # Create shaft cylinder
        shaft_mesh = trimesh.creation.cylinder(
            radius=shaft_radius,
            height=shaft_length
        )
        
        # Rotate and position shaft
        default_dir = np.array([0, 0, 1])
        rotation = R.align_vectors([direction_norm], [default_dir])[0]
        rotation_matrix = rotation.as_matrix()
        
        shaft_center = (start + shaft_end) / 2
        transform_shaft = np.eye(4)
        transform_shaft[:3, :3] = rotation_matrix
        transform_shaft[:3, 3] = shaft_center
        shaft_mesh.apply_transform(transform_shaft)
        
        # Add shaft mesh
        shaft_handle = self.server.scene.add_mesh_simple(
            name=f"/arrows/{name}_shaft",
            vertices=shaft_mesh.vertices,
            faces=shaft_mesh.faces,
            color=tuple(c / 255.0 for c in color),
            opacity=opacity,
            cast_shadow=True,
            receive_shadow=True
        )

        # Create cone mesh
        cone_mesh = trimesh.creation.cone(radius=head_radius, height=head_length)

        cone_center = shaft_end

        # Transform cone
        transform_cone = np.eye(4)
        transform_cone[:3, :3] = rotation_matrix
        transform_cone[:3, 3] = cone_center
        cone_mesh.apply_transform(transform_cone)

        # Add cone mesh
        head_handle = self.server.scene.add_mesh_simple(
            name=f"/arrows/{name}_head",
            vertices=cone_mesh.vertices,
            faces=cone_mesh.faces,
            color=tuple(c / 255.0 for c in color),
            opacity=opacity,
            cast_shadow=True,
            receive_shadow=True
        )
        
        return {'shaft': shaft_handle, 'head': head_handle}

    def capture_scene_png(self, out_path, height=720, width=720):
        import time
        import imageio.v2 as imageio

        # 1. client 연결될 때까지 대기
        while len(self.server.get_clients()) == 0:
            time.sleep(0.1)

        # 2. 첫 client 가져오기
        client = next(iter(self.server.get_clients().values()))
        print(f"Capturing PNG from client {client}...")

        # 3. 현재 카메라 상태로 렌더 요청
        img = client.get_render(
            height=height,
            width=width,
            transport_format="png",  # 논문용
        )
        print(f"Captured image of shape: {img.shape}")
        # 4. PNG 저장
        imageio.imwrite(out_path, img)

                
    def add_video_capture_gui(self):
        """Add video capture GUI controls"""
        with self.server.gui.add_folder("Video Capture"):
            # Resolution controls
            self.capture_width = self.server.gui.add_slider(
                "Width",
                min=480,
                max=1920,
                step=80,
                initial_value=1280
            )
            
            self.capture_height = self.server.gui.add_slider(
                "Height",
                min=480,
                max=1920,
                step=80,
                initial_value=720
            )
            
            # Video parameters
            self.video_fps = self.server.gui.add_slider(
                "Video FPS",
                min=10,
                max=60,
                step=1,
                initial_value=30
            )

            self.video_duration = self.server.gui.add_slider(
                "Duration (sec)",
                min=1.0,
                max=10.0,
                step=0.5,
                initial_value=3.0
            )
            
            # Output path
            self.output_path = self.server.gui.add_text(
                "Output Path",
                initial_value=""
            )
            
            # View controls
            self.set_start_view_btn = self.server.gui.add_button("Set Start View")
            self.set_end_view_btn = self.server.gui.add_button("Set End View")
            self.record_video_btn = self.server.gui.add_button("Record Video")
            
            # Screenshot
            self.capture_png_btn = self.server.gui.add_button("Capture PNG")

        # Callbacks
        @self.set_start_view_btn.on_click
        def _(_) -> None:
            if len(self.server.get_clients()) == 0:
                print("❌ No client connected!")
                return
            client = next(iter(self.server.get_clients().values()))
            self.start_view = {
                'position': client.camera.position,
                'wxyz': client.camera.wxyz
            }
            print("✓ Start view set:", self.start_view['position'], self.start_view['wxyz'])

        @self.set_end_view_btn.on_click
        def _(_) -> None:
            if len(self.server.get_clients()) == 0:
                print("❌ No client connected!")
                return
            client = next(iter(self.server.get_clients().values()))
            self.end_view = {
                'position': client.camera.position,
                'wxyz': client.camera.wxyz
            }
            print("✓ End view set:", self.end_view['position'])

        @self.record_video_btn.on_click
        def _(_) -> None:
            self._record_interpolated_video()
        
        @self.capture_png_btn.on_click
        def _(_) -> None:
            # output_path가 비어있으면 타임스탬프로 생성
            if self.output_path.value.strip() == "":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = f"capture_{timestamp}.png"
            else:
                out_path = self.output_path.value
            
            print(f"Capturing current view to {out_path}...")
            
            # GUI에서 설정한 해상도 사용
            width = int(self.capture_width.value)
            height = int(self.capture_height.value)
            
            # PNG 캡처
            self.capture_scene_png(out_path, height=height, width=width)
            print(f"✅ Saved: {out_path}")

    def _record_interpolated_video(self):
        """Record interpolated video between start and end views"""
        # Validation
        if not hasattr(self, 'start_view') or not hasattr(self, 'end_view'):
            print("❌ Please set both start and end views first!")
            return
        
        # Client 확인
        if len(self.server.get_clients()) == 0:
            print("❌ No client connected!")
            return
        
        client = next(iter(self.server.get_clients().values()))
        
        # Output path
        if self.output_path.value.strip() == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"video_{timestamp}.mp4"
        else:
            out_path = self.output_path.value
            if not out_path.endswith('.mp4'):
                out_path += '.mp4'
        
        print(f"\n🎬 Recording video to {out_path}...")
        
        # Parameters
        fps = int(self.video_fps.value)
        duration = float(self.video_duration.value)
        n_frames = int(fps * duration)
        width = int(self.capture_width.value)
        height = int(self.capture_height.value)
        
        # Position interpolation
        start_pos = np.array(self.start_view['position'])
        end_pos = np.array(self.end_view['position'])
        
        # Rotation interpolation (slerp)
        start_quat = np.array(self.start_view['wxyz'])  # [w, x, y, z]
        end_quat = np.array(self.end_view['wxyz'])
        
        # scipy는 [x, y, z, w] 순서
        start_rot = R.from_quat([start_quat[1], start_quat[2], start_quat[3], start_quat[0]])
        end_rot = R.from_quat([end_quat[1], end_quat[2], end_quat[3], end_quat[0]])
        
        slerp = Slerp([0, 1], R.concatenate([start_rot, end_rot]))
        
        # Temp directory for frames
        temp_dir = tempfile.mkdtemp()
        
        print(f"📸 Rendering {n_frames} frames ({width}x{height} @ {fps}fps)...")
        
        for i in range(n_frames):
            t = i / (n_frames - 1) if n_frames > 1 else 0
            
            # Linear position interpolation
            interp_pos = start_pos * (1 - t) + end_pos * t
            
            # Slerp rotation
            interp_rot = slerp(t)
            interp_quat_xyzw = interp_rot.as_quat()
            interp_quat_xyzw = interp_quat_xyzw / np.linalg.norm(interp_quat_xyzw)
            interp_quat_wxyz = [interp_quat_xyzw[3], interp_quat_xyzw[0], 
                            interp_quat_xyzw[1], interp_quat_xyzw[2]]
            
            # Update camera
            client.camera.position = tuple(interp_pos)
            client.camera.wxyz = tuple(interp_quat_wxyz)
            
            print(f"Frame {i+1}/{n_frames}: Pos {client.camera.position}, WXYZ {client.camera.wxyz}")
            # Capture
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            print(frame_path)
            self.capture_scene_png(frame_path, height=height, width=width)
            if True:#(i + 1) % 30 == 0 or i == n_frames - 1:
                print(f"  Progress: {i + 1}/{n_frames} frames")
        
        # Encode with ffmpeg
        print("🎞️  Encoding video...", temp_dir)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-loglevel', 'warning',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            out_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print(f"✅ Video saved: {out_path}\n")

    def add_view_save_gui(self):
        with self.server.gui.add_folder("Save/Load View"):
            self.view_file_path = self.server.gui.add_text(
                "File Path", initial_value="view.json"
            )
            self.save_view_btn = self.server.gui.add_button("Save View")
            self.load_view_btn = self.server.gui.add_button("Load View")
            self.level_view_btn = self.server.gui.add_button("Level View")

        @self.save_view_btn.on_click
        def _(_) -> None:
            if len(self.server.get_clients()) == 0:
                print("No client connected!")
                return
            client = next(iter(self.server.get_clients().values()))
            view = {
                "position": list(client.camera.position),
                "wxyz": list(client.camera.wxyz),
            }
            path = self.view_file_path.value.strip() or "view.json"
            with open(path, "w") as f:
                json.dump(view, f, indent=2)
            print(f"View saved: {path}")

        @self.load_view_btn.on_click
        def _(_) -> None:
            if len(self.server.get_clients()) == 0:
                print("No client connected!")
                return
            path = self.view_file_path.value.strip() or "view.json"
            if not os.path.exists(path):
                print(f"File not found: {path}")
                return
            with open(path, "r") as f:
                view = json.load(f)
            client = next(iter(self.server.get_clients().values()))
            client.camera.position = tuple(view["position"])
            client.camera.wxyz = tuple(view["wxyz"])
            print(f"View loaded: {path}")

        @self.level_view_btn.on_click
        def _(_) -> None:
            if len(self.server.get_clients()) == 0:
                print("No client connected!")
                return
            
            client = next(iter(self.server.get_clients().values()))
            pos = np.array(client.camera.position)
            w, x, y, z = client.camera.wxyz

            # 1. 현재 카메라가 바라보는 방향(Forward vector) 계산
            r_current = R.from_quat([x, y, z, w])
            # 기본적으로 카메라는 -z(또는 라이브러리에 따라 +x)를 바라봅니다. 
            # Viser/Three.js 기준으로는 보통 [0, 0, -1]이 정면입니다.
            forward = r_current.as_matrix() @ np.array([0, 0, -1])

            # 2. 새로운 Right, Up, Forward 축 계산 (Gram-Schmidt 유사 방식)
            # 월드의 Up은 [0, 0, 1]
            world_up = np.array([0, 0, 1])
            
            # Right = Forward x World_Up (수평 오른쪽 방향 추출)
            right = np.cross(forward, world_up)
            
            # 만약 카메라가 수직으로 하늘/땅을 보고 있어서 cross product가 0이 되는 경우 처리
            if np.linalg.norm(right) < 1e-6:
                print("Camera is looking straight up/down, cannot level Roll.")
                return

            right /= np.linalg.norm(right)
            
            # New Up = Right x Forward (이제 New Up은 월드 Z축 쪽으로 최대한 정렬됨)
            new_up = np.cross(-right, forward)
            new_up /= np.linalg.norm(new_up)

            # 3. 새로운 Rotation Matrix 생성
            # Matrix columns: [Right, Up, -Forward] (Viser/Three.js 좌표계 기준)
            new_R = np.stack([-right, new_up, -forward], axis=1)
            
            # 4. 쿼터니언 변환 및 적용
            new_q = R.from_matrix(new_R).as_quat() # x, y, z, w
            client.camera.wxyz = (new_q[3], new_q[0], new_q[1], new_q[2])
            
            print("View leveled: Up vector is now aligned with World +Z")

class ViserRobotModule():
    def __init__(self, target,#: viser.ViserServer | viser.ClientHandle,
                 urdf_path, 
                 scale: float = 1.0,
                 root_node_name: str = "/",
                 load_meshes=True, 
                 load_collision_meshes=False):
        self._urdf = RobotModule(urdf_path)
        self._target = target
        self._scale = scale
        self._load_meshes = load_meshes
        self._load_collision_meshes = load_collision_meshes
        self._joint_frames: List[viser.FrameHandle] = []
        self._meshes: Dict[str, viser.MeshHandle] = {}
        num_joints_to_repeat = 0

        if load_meshes:
            if self._urdf.scene is not None:
                num_joints_to_repeat += 1
                self._visual_root_frame = self._add_joint_frames_and_meshes(
                    self._urdf.scene,
                    root_node_name,
                    collision_geometry=False
                )

        if load_collision_meshes:
            if self._urdf.collision_scene is not None:
                num_joints_to_repeat += 1
                self._collision_root_frame = self._add_joint_frames_and_meshes(
                    self._urdf.collision_scene,
                    root_node_name,
                    collision_geometry=True
                )
        self._joint_map_values = [*self._urdf.joint_map.values()] * num_joints_to_repeat
        self.update_cfg(np.zeros(len(self._urdf.joint_map)))    

    def change_color(self, name_list, color: Tuple[float, float, float]) -> None:
        """Change the color of the visualized URDF."""
        name_list = list(self._meshes.keys()) if len(name_list) == 0 else name_list
        
        if len(color) == 4:
            opacity = color[3]
        else:
            opacity = 1.0

        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        for name in name_list:
            self._meshes[name].color = color
            self._meshes[name].opacity = opacity

    @property
    def show_visual(self) -> bool:
        """Returns whether the visual meshes are currently visible."""
        return self._visual_root_frame is not None and self._visual_root_frame.visible

    @show_visual.setter
    def show_visual(self, visible: bool) -> None:
        """Set whether the visual meshes are currently visible."""
        if self._visual_root_frame is not None:
            self._visual_root_frame.visible = visible

    @property
    def show_collision(self) -> bool:
        """Returns whether the collision meshes are currently visible."""
        return (
            self._collision_root_frame is not None
            and self._collision_root_frame.visible
        )

    @show_collision.setter
    def show_collision(self, visible: bool) -> None:
        """Set whether the collision meshes are currently visible."""
        if self._collision_root_frame is not None:
            self._collision_root_frame.visible = visible

    @property
    def urdf(self) -> RobotModule:
        """Get the underlying URDF object."""
        return self._urdf
    
    def remove(self) -> None:
        """Remove URDF from scene."""
        # Some of this will be redundant, since children are removed when
        # parents are removed.
        for frame in self._joint_frames:
            frame.remove()
        for mesh in self._meshes:
            mesh.remove()
               
    def update_cfg(self, configuration: np.ndarray) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        for joint, frame_handle in zip(self._joint_map_values, self._joint_frames):
            assert isinstance(joint, yourdfpy.Joint)
            T_parent_child = self._urdf.get_transform(
                joint.child, joint.parent, collision_geometry=not self._load_meshes
            )
            
            xyzw = R.from_matrix(T_parent_child[:3, :3].copy()).as_quat()
            frame_handle.wxyz = xyzw[[3, 0, 1, 2]]
            frame_handle.position = T_parent_child.copy()[:3, 3] * self._scale
            
    def _add_joint_frames_and_meshes(
        self,
        scene: Scene,
        root_node_name: str,
        collision_geometry: bool
    ) :
        """
        Helper function to add joint frames and meshes to the ViserUrdf object.
        """
        prefix = "collision" if collision_geometry else "visual"
        prefixed_root_node_name = (f"{root_node_name}/{prefix}").replace("//", "/")
        root_frame = self._target.scene.add_frame(
            prefixed_root_node_name, show_axes=False
        )
        
        # Add coordinate frame for each joint.
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            self._joint_frames.append(
                self._target.scene.add_frame(
                    _viser_name_from_frame(
                        scene,
                        joint.child,
                        prefixed_root_node_name,
                    ),
                    show_axes=False,
                )
            )

        # Add the URDF's meshes/geometry to viser.
        for link_name, mesh in scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = self._urdf.get_transform(
                link_name,
                scene.graph.transforms.parents[link_name],
                collision_geometry=collision_geometry,
            )
            name = _viser_name_from_frame(scene, link_name, prefixed_root_node_name)
            # Scale + transform the mesh. (these will mutate it!)
            #
            # It's important that we use apply_transform() instead of unpacking
            # the rotation/translation terms, since the scene graph transform
            # can also contain scale and reflection terms.
            mesh = mesh.copy()
            mesh.apply_scale(self._scale)
            mesh.apply_transform(T_parent_child)

            vertices = mesh.vertices
            faces = mesh.faces
            color = np.array(mesh.visual.vertex_colors[0, :3])
            self._meshes[name] = (self._target.scene.add_mesh_simple(
                name, vertices, faces, color=color,
                cast_shadow=True, receive_shadow=True
            ))
            # self._meshes[name] = self._target.scene.add_mesh_trimesh(
            #     name=name,
            #     mesh=mesh
            # )
        return root_frame
    
    def get_link_vertices(self, link_name: str = None) -> Dict[str, np.ndarray]:
        """
        Get vertices of link meshes in world coordinates
        
        Args:
            link_name: Specific link name to get vertices from. If None, returns all links.
            
        Returns:
            Dictionary mapping link names to their vertices (N, 3) in world coordinates
        """
        vertices_dict = {}
        
        # Iterate through all meshes
        for mesh_name, mesh_handle in self._meshes.items():
            # Extract actual link name from the viser mesh name
            # Example: "/robot/arm/visual/link1" -> "link1"
            actual_link_name = mesh_name.split("/")[-1]
            
            # If specific link requested, filter
            if link_name is not None and actual_link_name != link_name:
                continue
            
            # Get the mesh from the handle
            # Note: viser doesn't directly expose vertex data after adding,
            # so we need to get it from the original URDF scene
            if self._load_meshes and self._urdf.scene is not None:
                if actual_link_name in self._urdf.scene.geometry:
                    mesh = self._urdf.scene.geometry[actual_link_name]
                    
                    # Get current transform for this link
                    T = self._urdf.get_transform(
                        actual_link_name, 
                        self._urdf.scene.graph.base_frame,
                        collision_geometry=False
                    )
                    
                    # Transform vertices to world coordinates
                    vertices = mesh.vertices.copy()
                    vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
                    vertices_world = (T @ vertices_homogeneous.T).T[:, :3]
                    
                    # Apply scale
                    vertices_world = vertices_world * self._scale
                    
                    vertices_dict[actual_link_name.split(".")[0]] = vertices_world
        
        return vertices_dict

    def set_visibility(self, visible: bool) -> None:
        """Set visibility of all meshes."""
        for mesh in self._meshes.values():
            mesh.visible = visible

    def get_all_vertices(self) -> np.ndarray:
        """
        Get all vertices from all links concatenated together
        
        Returns:
            (N, 3) array of all vertices in world coordinates
        """
        all_vertices = []
        vertices_dict = self.get_link_vertices()
        
        for vertices in vertices_dict.values():
            all_vertices.append(vertices)
        
        if len(all_vertices) == 0:
            return np.array([]).reshape(0, 3)
        
        return np.vstack(all_vertices)
    # def apply_mesh_color_override(mesh_color_override):
        # elif len(mesh_color_override) == 3:
        #     self._meshes.append(
        #         self._target.scene.add_mesh_simple(
        #             name,
        #             mesh.vertices,
        #             mesh.faces,
        #             color=mesh_color_override,
        #         )
        #     )
        # elif len(mesh_color_override) == 4:
        #     self._meshes.append(
        #         self._target.scene.add_mesh_simple(
        #             name,
        #             mesh.vertices,
        #             mesh.faces,
        #             color=mesh_color_override[:3],
        #             opacity=mesh_color_override[3],
        #         )
        #     )
        # else:
        #     assert_never(mesh_color_override)
        
def _viser_name_from_frame(
    scene: Scene,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")
    frames = []
    while frame_name != scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = scene.graph.transforms.parents[frame_name]
    if root_node_name != "/":
        frames.append(root_node_name)
    return "/".join(frames[::-1])

