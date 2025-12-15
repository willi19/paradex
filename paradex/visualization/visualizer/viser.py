from distro import name
import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R
from trimesh import Scene
import viser
import trimesh
from typing import List, Tuple, Dict
import time
import viser.transforms as tf

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
        self.server.gui.configure_theme(dark_mode=False)

        self.server.scene.set_up_direction(self.up_direction)
        self.server.scene.world_axes

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            near_slider = client.gui.add_slider(
                "Near", min=0.01, max=10.0, step=0.001, initial_value=client.camera.near
            )
            far_slider = client.gui.add_slider(
                "Far", min=1, max=1000.0, step=0.001, initial_value=client.camera.far
            )

            @near_slider.on_update
            def _(_) -> None:
                client.camera.near = near_slider.value

            @far_slider.on_update
            def _(_) -> None:
                client.camera.far = far_slider.value

    def add_robot(self, name, urdf_path):
        robot = ViserRobotModule(
            target=self.server,
            urdf_path=urdf_path,
            scale=1.0,
            root_node_name=f"/robot/{name}",
            load_meshes=True,
            load_collision_meshes=False,
        )
        # if "0" in name:
        #     robot.change_color(
        #         [f"/robot/{name}/visual/base_link"], (0.8, 0.2, 0.2)
        #     )
        
        self.robot_dict[name] = robot

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
            show_axes=True,
            axes_length=0.05,
            axes_radius=0.002,
        )
        
        # Add mesh to the frame (at origin relative to frame)
        mesh_handle = self.server.scene.add_mesh_trimesh(
                name=f"/objects/{name}_frame/{name}",
                mesh=obj
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
        if len(robot_traj) == 0:
            return
        traj_len = robot_traj[list(robot_traj.keys())[0]].shape[0]
        new_traj_dict = {"robot":{}, "object":{}}
        for robot_name in list(self.robot_dict.keys()):
            if robot_name in robot_traj:
                new_traj_dict["robot"][robot_name] = robot_traj[robot_name]
            else:
                new_traj_dict["robot"][robot_name] = np.tile(self.robot_dict[robot_name].urdf.get_cfg(), (traj_len, 1))

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
        print(traj_len, self.num_frames)

    def add_grid_lines(self, height=0.0, size=5.0):
        """Add grid lines separately using line segments"""
        grid_spacing = 0.1  # 0.5m grid spacing
        lines_added = 0
        # X direction lines (parallel to X axis)
        for y in np.arange(-size, size + grid_spacing, grid_spacing):
            self.server.scene.add_spline_catmull_rom(
                f"grid_x_{lines_added}",
                positions=np.array([[-size, y, height], [size, y, height]]),
                color=(0.4, 0.4, 0.4),
                line_width=1.0
            )
            lines_added += 1
            
        # Y direction lines (parallel to Y axis) 
        for x in np.arange(-size, size + grid_spacing, grid_spacing):
            self.server.scene.add_spline_catmull_rom(
                f"grid_y_{lines_added}",
                positions=np.array([[x, -size, height], [x, size, height]]),
                color=(0.4, 0.4, 0.4),
                line_width=1.0
            )
            lines_added += 1
    
    def add_floor(self, height=0.0):
        """Update floor visibility and appearance"""
        self.floor_size = 1.0
        try:
            # Remove existing floor and grid elements
            try:
                self.server.scene.remove_by_name("floor")
            except:
                pass
            
            # Remove existing grid lines  
            for i in range(100):
                try:
                    self.server.scene.remove_by_name(f"grid_x_{i}")
                    self.server.scene.remove_by_name(f"grid_y_{i}")
                except:
                    pass

            if True:  # self.floor_visible.value:
                # size = self.floor_size.value
                size = self.floor_size
                
                # Create a simple box as floor (very thin)
                self.server.scene.add_box(
                    name="floor",
                    dimensions=(size * 2, size * 2, 0.02),  # width, height, thickness
                    position=(0.0, 0.0, height-0.01),  # Position slightly below z=0
                    color=(0.7, 0.7, 0.7)
                )
                
                # Add grid lines if enabled
                if True: #self.grid_visible.value:
                    try:
                        self.add_grid_lines(height, size=size)
                    except Exception as e:
                        print(f"❌ Grid lines failed: {e}")
            else:
                print("🚫 Floor hidden")
                
        except Exception as e:
            print(f"❌ Floor update failed: {e}")
            import traceback
            traceback.print_exc()

    def create_floor_mesh(self, size=5.0):
        """Create a simple floor mesh"""
        # Create floor plane - make sure vertices are in correct order for proper normals
        floor_vertices = np.array([
            [-size, -size, 0],
            [size, -size, 0], 
            [size, size, 0],
            [-size, size, 0]
        ], dtype=np.float32)
        
        # Make sure faces have correct winding order (counter-clockwise when viewed from above)
        floor_faces = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3]   # Second triangle
        ])
        
        # Create mesh and ensure normals point upward
        floor_mesh = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)
        floor_mesh.fix_normals()  # Fix normal directions
        
        return floor_mesh
    
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
        if self.gui_playing.value and len(self.traj_list) > 0:
            next_timestep = (self.gui_timestep.value + 1) % self.num_frames
            self.gui_timestep.value = next_timestep
            
        time.sleep(1.0 / self.gui_framerate.value)
    
    def start_viewer(self):
        try:
            while True:
                self.update()
        except KeyboardInterrupt:
            pass
    
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
            self.gui_timestep.value = (self.gui_timestep.value + 10) % self.num_frames

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

    def add_frame(self, name):
        self.frame_nodes[name] = self.server.scene.add_frame(
            name=f"/{name}/frame",
            show_axes=True,
            axis_length=0.1,
            axis_radius=0.002,
        )

    def add_lights(self):
        self.server.scene.add_transform_controls(
            "/system/control_light0", position=(5, 1.0, 0.5), scale=0.5
        )
        self.server.scene.add_label("/system/control_light0/label", "Point")
        point_light = self.server.scene.add_light_point(
            name="/system/control_light0/point_light",
            color=(255, 255, 255),
            intensity=100.0,
            visible=True
        )

        self.server.scene.add_transform_controls(
            "/system/control_light1", position=(5, -1.0, 0.5), scale=0.5
        )
        self.server.scene.add_label("/system/control_light1/label", "Point")
        point_light = self.server.scene.add_light_point(
            name="/system/control_light1/point_light",
            color=(255, 255, 255),
            intensity=100.0,
            visible=True
        )
        self.server.scene.enable_default_lights(True)
        
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
            axes_radius=size * 0.01,
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
        name_list = list(self._meshes.keys())
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        for name in name_list:
            self._meshes[name].color = color
    
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

            # vertices = mesh.vertices
            # faces = mesh.faces
            # color = np.array(mesh.visual.vertex_colors[0, :3])
            # self._meshes[name] = (self._target.scene.add_mesh_simple(name, vertices, faces, color=color))
            self._meshes[name] = self._target.scene.add_mesh_trimesh(
                name=name,
                mesh=mesh
            )
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
