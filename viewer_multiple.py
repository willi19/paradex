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

import open3d as o3d
import torch
import json
from glob import glob

from paradex.utils.file_io import load_current_camparam

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

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
from paradex.utils.file_io import rsc_path, load_latest_C2R, get_robot_urdf_path
from paradex.visualization_.robot_module import Robot_Module
from paradex.robot.mimic_joint import parse_inspire
from paradex.object_detection.obj_utils.vis_utils import read_mesh, get_initial_mesh



class ViserViewer:
    def __init__(
        self,
        obj_meshes,
        obj_Ts=None,
        urdf_path=None,
        qpos=None,
        c2r=None,
        object_nms="object",
    ):
            
            
        '''
            If all value is None including obj_T, urdf_path, qpos
        '''
        
        self.obj_Ts = obj_Ts
        self.meshes = copy.deepcopy(obj_meshes)
        self.object_nms = object_nms
        self.mesh_handles = []
        
        # Set default c2r if not provided
        if c2r is None:
            self.c2r = np.eye(4)
        else:
            self.c2r = c2r
            
        self.load_server()
        
        # Load URDF
        if urdf_path is not None and qpos is not None:
            urdf = yourdfpy.URDF.load(urdf_path)
            self.urdf = urdf
            self.robot_base = self.server.scene.add_frame("/robot", show_axes=False)
            self.robot_base.position = (-0.6, 0.6, 0.625)
            self.robot_base.wxyz = (0.0, 1.0, -0.3, 0.14)
            self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/robot")
            self.qpos = qpos
            self.num_frames  = self.qpos.shape[0]
            # Print joint information for debugging
            print("=== URDF Joint Information ===")
            print(f"Total joints: {len(urdf.joint_map)}")
            print(f"Actuated joints: {len(urdf.actuated_joint_names)}")
            print("Joint order:")
            for i, joint_name in enumerate(urdf.actuated_joint_names):
                print(f"  {i}: {joint_name}")
            print(f"Expected qpos shape: ({self.num_frames}, {len(urdf.actuated_joint_names)})")
            print(f"Actual qpos shape: {qpos.shape}")
            print("=" * 30)
            
            if obj_Ts is not None:
                self.num_frames = min(self.num_frames, self.obj_Ts[0].shape[0])
        else:
            self.urdf, self.urdf_vis, self.qpos = None, None, None
            if obj_Ts is not None:
                self.num_frames = self.obj_Ts[0].shape[0]
            else:
                self.num_frames = 1
                
        if obj_Ts is None:
            self.obj_loaded = False
            self.prev_objtimestamp = -1

        self.add_frames()
        self.add_player()
        self.prev_timestep = 0
        self.add_initial_meshes()
        
        self.last_obj_number = 0
    
    '''
    def load_server(self):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(dark_mode=False)
        
        self.server.scene.set_up_direction(-self.c2r[:3,2])
        self.server.scene.world_axes

        # z_near, z_far setting
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            return
            """For each client that connects, create GUI elements for adjusting the
            near/far clipping planes."""
            
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
    '''
    def load_server(self):
        self.server = viser.ViserServer(port=8081)
        self.server.gui.configure_theme(dark_mode=False)
        
        self.server.scene.set_up_direction(-self.c2r[:3,2])
        self.server.scene.world_axes

        # z_near, z_far setting
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.wxyz = [0.785, 0.366, 0.211,  0.453]
            client.camera.position = np.array([-2.0, 1.5, -0.5])
            return
            """For each client that connects, create GUI elements for adjusting the
            near/far clipping planes."""
            
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
        # add player 
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
            
        # Floor visibility controls
        with self.server.gui.add_folder("Scene"):
            self.floor_visible = self.server.gui.add_checkbox("Show Floor", False)
            self.floor_size = self.server.gui.add_slider(
                "Floor Size", min=1.0, max=10.0, step=0.5, initial_value=5.0
            )
            self.grid_visible = self.server.gui.add_checkbox("Show Grid", True)
            gui_framerate_options = self.server.gui.add_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

            gui_up = self.server.gui.add_vector3(
                "Up Direction",
                initial_value=(0.0, -1.0, 1.0),
                step=0.01,
            )
            
            self.manual_update = self.server.gui.add_button("Manual Update", disabled=True)
            
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
            self.update_scene(self.gui_timestep.value, visualize_texture=self.visualize_texture)
            
        @gui_framerate_options.on_click
        def _(_) -> None:
            self.gui_framerate.value = int(gui_framerate_options.value)

        # Frame step buttons.
        @self.gui_next_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames

        @self.gui_prev_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % self.num_frames

        # Disable frame controls when we're playing.
        @self.gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = self.gui_playing.value
            self.gui_next_frame.disabled = self.gui_playing.value
            self.gui_prev_frame.disabled = self.gui_playing.value
            
        @self.manual_update.on_click
        def _(_) -> None:
            self.update_scene(self.gui_timestep.value, visualize_texture=self.visualize_texture)
            
    def add_frames(self):
        # NOTE: scene setting start
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
            axes_radius=0.01
        )

        self.frame_nodes: list[viser.FrameHandle] = []
        
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
    

    def add_grid_lines(self, size=5.0, position=(0,0,0), wxyz=(1.0,0.0,0.0,0.0)):
        """Add grid lines that follow floor's position and rotation"""
        import scipy.spatial.transform

        grid_spacing = 0.5

        r = scipy.spatial.transform.Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        lines_added = 0

        for y in np.arange(-size, size + grid_spacing, grid_spacing):
            start = np.array([-size, y, 0.001])
            end   = np.array([ size, y, 0.001])
            
            start_rotated = r.apply(start) + np.array(position)
            end_rotated   = r.apply(end) + np.array(position)
            
            self.server.scene.add_spline_catmull_rom(
                f"grid_x_{lines_added}",
                positions=np.array([start_rotated, end_rotated]),
                color=(0.4, 0.4, 0.4),
                line_width=1.0
            )
            lines_added += 1

        for x in np.arange(-size, size + grid_spacing, grid_spacing):
            start = np.array([x, -size, 0.001])
            end   = np.array([x,  size, 0.001])
            
            start_rotated = r.apply(start) + np.array(position)
            end_rotated   = r.apply(end) + np.array(position)
            
            self.server.scene.add_spline_catmull_rom(
                f"grid_y_{lines_added}",
                positions=np.array([start_rotated, end_rotated]),
                color=(0.4, 0.4, 0.4),
                line_width=1.0
            )
            lines_added += 1

    def update_floor(self, position=(0.0, 0.0, -0.041), wxyz=(1.0, 0.0, 0.0, 0.0)):
        """Update floor visibility, position, and rotation"""
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
            
            if self.floor_visible.value:
                size = self.floor_size.value
                
                # Add floor box with position + rotation
                self.server.scene.add_box(
                    name="floor",
                    dimensions=(size * 2, size * 2, 0.02),  # width, height, thickness
                    position=position,                      # 원하는 위치
                    wxyz=wxyz,                              # 원하는 회전 (quaternion w,x,y,z)
                    color=(0.7, 0.7, 0.7)
                )
                
                print(f"✅ Floor box added: size={size*2}x{size*2}x0.02 at {position} with rotation {wxyz}")
                
                # Add grid lines if enabled
                if self.grid_visible.value:
                    try:
                        self.add_grid_lines(size=size)
                        print(f"✅ Grid lines added")
                    except Exception as e:
                        print(f"❌ Grid lines failed: {e}")
            else:
                print("🚫 Floor hidden")
                
        except Exception as e:
            print(f"❌ Floor update failed: {e}")
            import traceback
            traceback.print_exc()

    def add_initial_meshes(self):
        """Add initial meshes for timestep 0"""
        # Add initial robot configuration
        if self.urdf_vis is not None:
            self.urdf_vis.update_cfg(self.qpos[0])
            # self.server.scene.set_object_transform(self.urdf_vis.root_handle, position=(5.0, 5.0, 5.0), rotation=(1.0, 0.0, 0.0, 0.0))
            
        if self.obj_Ts is not None:    
            # Add initial object

            for obj_name, mesh, obj_T in zip(self.object_nms, self.meshes, self.obj_Ts):
                initial_obj_mesh = copy.deepcopy(mesh).apply_transform(obj_T[0])

                mesh_handle = self.server.scene.add_mesh_trimesh(
                    name=obj_name,
                    mesh=initial_obj_mesh,
                    position=(0.0, 0.0, 0.0),
                )
                self.mesh_handles.append(mesh_handle)
        
        # Add floor
        self.update_floor()
    
    def update_scene(self, timestep):
        """Update scene for given timestep"""
        with self.server.atomic():
            if self.urdf_vis is not None:
                # Update robot configuration
                self.urdf_vis.update_cfg(self.qpos[timestep])
                
            if self.obj_Ts is not None:
                # Update object mesh
                for obj_name, mesh, obj_T in zip(self.object_nms, self.meshes, self.obj_Ts):
                    transformed_obj_mesh = copy.deepcopy(mesh).apply_transform(obj_T[timestep])
                    self.server.scene.remove_by_name(obj_name)
                    self.server.scene.add_mesh_trimesh(
                        name=obj_name,
                        mesh=transformed_obj_mesh,
                        position=(0.0, 0.0, 0.0),
                    )
            else:
                obj_T_path = '/home/temp_id/paradex/objoutput/obj_T.pkl'
                if os.path.exists(obj_T_path):
                    try:
                        cur_obj_T = pickle.load(open(obj_T_path, 'rb'))
                    except:
                        return
                    if self.prev_objtimestamp == cur_obj_T['timestamp']:
                        # print("No update in object pose")
                        return
                    else:
                        self.prev_objtimestamp = cur_obj_T['timestamp']
                        cur_obj_T.pop('timestamp')
                    print(f"Updating objects")
                    if self.obj_loaded:
                        for oidx in range(self.last_obj_number):
                            self.server.scene.remove_by_name(f"{self.object_nm}_{oidx}")

                    for oidx, obj_T in cur_obj_T.items():
                        # Get Current Object Transformation 
                        transformed_obj_mesh = \
                            copy.deepcopy(self.mesh).apply_transform(np.linalg.inv(self.c2r)@obj_T)
                        # c2r = np.linalg.inv(self.c2r)
                        self.server.scene.add_mesh_trimesh(
                            name=f"{self.object_nm}_{oidx}",
                            mesh=transformed_obj_mesh,
                            position=(0.0, 0.0, 0.0),
                        )
                        self.obj_loaded = True
                        
                    self.last_obj_number = len(cur_obj_T)

                        
            
        self.prev_timestep = timestep
        self.server.flush()
        
        # Render to PNG if enabled
        if self.render_png.value:
            self.render_current_frame(timestep)
    
    def render_current_frame(self, timestep):
        """Render current frame to PNG"""
        render_dir = Path("rendered")
        render_dir.mkdir(exist_ok=True)
        
        for _, client in self.server.get_clients().items():
            try:
                rendered_img = client.get_render(height=720*2, width=1280*2)
                output_path = render_dir / f'{timestep:05d}.jpeg'
                cv2.imwrite(str(output_path), rendered_img)
            except Exception as e:
                print(f"Failed to render frame {timestep}: {e}")
        
    def update(self, data_path, cam2extr, cam2intr):
        """Main update loop - call this in a loop"""
        if self.gui_playing.value:
            if self.num_frames > 1:
                next_timestep = (self.gui_timestep.value + 1) % self.num_frames
                self.gui_timestep.value = next_timestep
            else:
                self.update_scene(0)
            
        
        if os.path.exists(f"{data_path}/obj_T.pkl"):
            with open(f"{data_path}/obj_T.pkl", "rb") as handle:
                obj_T_dict = pickle.load(handle)
        elif os.path.exists(f"{data_path}/obj_output/obj_T.pkl"):
            with open(f"{data_path}/obj_output/obj_T.pkl", "rb") as handle:
                obj_T_dict = pickle.load(handle)
        elif len(sorted(glob(f"{data_path}/obj_output/*/obj_T.pkl"))) > 0:
            with open(sorted(glob(f"{data_path}/obj_output/*/obj_T.pkl"))[-1], "rb") as handle:
                obj_T_dict = pickle.load(handle)
        elif len(sorted(glob(f"{data_path}/obj_output/*/*/obj_T.pkl"))) > 0:
            with open(sorted(glob(f"{data_path}/obj_output/*/*/obj_T.pkl"))[-1], "rb") as handle:
                obj_T_dict = pickle.load(handle)

        obj_meshes = []
        obj_Ts = []
        obj_names = []
        for key, val in obj_T_dict.items():
            if "brown" in key: obj_name = "brown_ramen_von"
            elif "red" in key: obj_name = "red_ramen_von"
            elif "yellow" in key: obj_name = "yellow_ramen_von"

            for obj_idx, obj_T in val.items():
                #if visualize_texture:
                #    obj_mesh = trimesh.load(f"template/{obj_name}/{obj_name}.obj")
                #else:
                #    obj_mesh, scaled = get_initial_mesh(obj_name, return_type='trimesh', simplify=True, device=DEFAULT_DEVICE)
                #obj_mesh, scaled = get_initial_mesh(obj_name, return_type='trimesh', simplify=False, device=DEFAULT_DEVICE)
                obj_mesh = trimesh.load(f"template_mesh/{obj_name}/{obj_name}.obj")
                obj_meshes.append(obj_mesh)
                obj_Ts.append(np.array([obj_T]))
                obj_names.append(obj_name+f"_{obj_idx}")

        for mesh_handle in self.mesh_handles:
            mesh_name = mesh_handle.name
            index = obj_names.index(mesh_name)
            mesh_handle.mesh = obj_meshes[index].apply_transform(obj_Ts[index][0])

        #key_list = [img.split(".")[0] for img in os.listdir("obj_output/image") if ".png" in img]
        #for key in key_list:
        #    image_cam_id = cv2.imread(f"obj_output/image/{cam_id}.png")
        #    print(cam2intr[image_cam_id])
        #    viewer.add_camera_with_image(
        #        cam2intr[image_cam_id],
        #        cam2extr[image_cam_id],
        #        image_cam_id,
        #        name=f"Camera: {image_cam_id}",
        #    )

        time.sleep(1.0 / self.gui_framerate.value)
    
    def start_viewer(self, data_path, cam2extr, cam2intr):
        """Start the viewer in a loop"""
        print(f"Starting viewer with {self.num_frames} frames")
        print("Visit the URL printed above to view the visualization")
        
        try:
            while True:
                self.update(data_path, cam2extr, cam2intr)
        except KeyboardInterrupt:
            print("Viewer stopped")

    def add_camera_with_image(self, intrinsic, extrinsic_3x4, image, name):
        H, W = image.shape[:2]

        T_camera2world = tf.SE3.from_matrix(
            extrinsic_3x4
        ).inverse()

        fy = intrinsic[1,1]
        image = image[::5, ::5]*255

        frame = self.server.scene.add_frame(
            name=f"{name}_frustum",
            wxyz=T_camera2world.rotation().wxyz,
            position=T_camera2world.translation(),
            axes_length=0.1,
            axes_radius=0.005,
        )
        
        frustum = self.server.scene.add_camera_frustum(
            name=f"{name}_frustum",
            fov=5 * np.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.05,
            image=image,
            position=T_camera2world.translation(),
            wxyz=T_camera2world.rotation().wxyz,
        )
        @frustum.on_click
        def _(_, frame=frame) -> None:
            print(f'Selected: {frame.name}')
            for client in self.server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', default="yellow_ramen_von/0")
    parser.add_argument('--visualize_texture', action="store_true")
    args = parser.parse_args()

    scene_path = args.scene_path
    visualize_texture = args.visualize_texture

    if os.path.exists(f"{scene_path}/obj_T.pkl"):
        with open(f"{scene_path}/obj_T.pkl", "rb") as handle:
            obj_T_dict = pickle.load(handle)
    elif os.path.exists(f"{scene_path}/obj_output/obj_T.pkl"):
        with open(f"{scene_path}/obj_output/obj_T.pkl", "rb") as handle:
            obj_T_dict = pickle.load(handle)
    elif len(sorted(glob(f"{scene_path}/obj_output/*/obj_T.pkl"))) > 0:
        with open(sorted(glob(f"{scene_path}/obj_output/*/obj_T.pkl"))[-1], "rb") as handle:
            obj_T_dict = pickle.load(handle)
    elif len(sorted(glob(f"{scene_path}/obj_output/*/*/obj_T.pkl"))) > 0:
        with open(sorted(glob(f"{scene_path}/obj_output/*/*/obj_T.pkl"))[-1], "rb") as handle:
            obj_T_dict = pickle.load(handle)

    print(obj_T_dict)
    obj_meshes = []
    obj_Ts = []
    obj_names = []
    for key, val in obj_T_dict.items():
        if "brown" in key: obj_name = "brown_ramen_von"
        elif "red" in key: obj_name = "red_ramen_von"
        elif "yellow" in key: obj_name = "yellow_ramen_von"

        for obj_idx, obj_T in val.items():
            #if visualize_texture:
            #    obj_mesh = trimesh.load(f"template/{obj_name}/{obj_name}.obj")
            #else:
            #    obj_mesh, scaled = get_initial_mesh(obj_name, return_type='trimesh', simplify=True, device=DEFAULT_DEVICE)
            #obj_mesh, scaled = get_initial_mesh(obj_name, return_type='trimesh', simplify=False, device=DEFAULT_DEVICE)
            obj_mesh = trimesh.load(f"template_mesh/{obj_name}/{obj_name}.obj")
            obj_meshes.append(obj_mesh)
            obj_Ts.append(np.array([obj_T]))
            obj_names.append(obj_name+f"_{obj_idx}")

    urdf_path = get_robot_urdf_path("xarm", "allegro")

    # obj_meshes = []
    # obj_Ts = []
    # obj_names = []
    # for key, val in obj_T_simple.items():
    #     obj_name = key
    #     obj_mesh, scaled = get_initial_mesh(obj_name, return_type='trimesh', simplify=True, device=DEFAULT_DEVICE)
    #     obj_meshes.append(obj_mesh)
    #     obj_Ts.append(val)
    #     obj_names.append(obj_name)

    c2r = load_latest_C2R()

    viewer = ViserViewer(obj_meshes, object_nms=obj_names, c2r=c2r, urdf_path=urdf_path, qpos=np.zeros((1, 29)), obj_Ts=obj_Ts)

    #with open(f"obj_output/cam_param/extrinsics.json", "r", encoding="utf-8") as f: extrinsics_dict = json.load(f)
    #with open(f"obj_output/cam_param/intrinsics.json", "r", encoding="utf-8") as f: intrinsics_dict = json.load(f)

    intrinsics_dict, extrinsics_dict = load_current_camparam()

    cam_params = dict(); cam2extr = dict(); cam2intr = dict()
    for cam_id in extrinsics_dict:
        extrinsic_np = np.array(extrinsics_dict[cam_id]) # 3X4
        intrinsic_key = 'intrinsics_undistort' if 'intrinsics_undistort' in intrinsics_dict[cam_id] else 'original_intrinsics'
        intrinsic_np = np.array(intrinsics_dict[cam_id][intrinsic_key]).reshape(3,3)
        dist = intrinsics_dict[cam_id]['dist_params'] if 'dist_params' in intrinsics_dict[cam_id] else  intrinsics_dict[cam_id]['dist_param'] 
        cam_params[cam_id] = {'extrinsic':extrinsic_np, 'intrinsic':intrinsic_np.reshape(3,3), 'dist':dist}
        cam2extr[cam_id] = extrinsic_np
        cam2intr[cam_id] = intrinsic_np

    key_list = [img.split(".")[0] for img in os.listdir("obj_output/image") if ".png" in img]
    for key in key_list:
        image_cam_id = cv2.imread(f"obj_output/image/{cam_id}.png")
        viewer.add_camera_with_image(
            cam2intr[image_cam_id],
            cam2extr[image_cam_id],
            image_cam_id,
            name=f"Camera: {image_cam_id}",
        )

    #for cam_id in cam2extr.keys():

        # if cam_id not in ["22684210", 22684210]: continue
    #    image_cam_id = cv2.imread(f"obj_output/image/{cam_id}.png")
    #    viewer.add_camera_with_image(
    #        cam2intr[cam_id],
    #        cam2extr[cam_id],
    #        image_cam_id,
    #        name=f"Camera: {cam_id}",
    #    )

    
    viewer.start_viewer(scene_path, cam2extr, cam2intr)