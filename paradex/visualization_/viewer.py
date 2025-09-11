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
        obj_mesh,
        obj_T=None,
        urdf_path=None,
        qpos=None,
        c2r=None,
        object_nm="object"):
            
            
        '''
            If all value is None including obj_T, urdf_path, qpos
        '''
        
        self.obj_T = obj_T
        self.mesh = copy.deepcopy(obj_mesh)
        self.object_nm = object_nm
        
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
            self.urdf_vis = ViserUrdf(self.server, urdf)
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
            
            if obj_T is not None:
                self.num_frames = min(self.num_frames, self.obj_T.shape[0])
        else:
            self.urdf, self.urdf_vis, self.qpos = None, None, None
            if obj_T is not None:
                self.num_frames = self.obj_T.shape[0]
            else:
                self.num_frames = 1
                
        if obj_T is None:
            self.obj_loaded = False

        self.add_frames()
        self.add_player()
        self.prev_timestep = 0
        self.add_initial_meshes()
        
        self.last_obj_number = 0
    
    def load_server(self):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(dark_mode=False)
        
        self.server.scene.set_up_direction(-self.c2r[:3,2])
        self.server.scene.world_axes

        # z_near, z_far setting
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
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
            self.floor_visible = self.server.gui.add_checkbox("Show Floor", True)
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
    
    def add_grid_lines(self, size=5.0):
        """Add grid lines separately using line segments"""
        grid_spacing = 0.5  # 0.5m grid spacing
        
        lines_added = 0
        # X direction lines (parallel to X axis)
        for y in np.arange(-size, size + grid_spacing, grid_spacing):
            self.server.scene.add_spline_catmull_rom(
                f"grid_x_{lines_added}",
                positions=np.array([[-size, y, 0.001], [size, y, 0.001]]),
                color=(0.4, 0.4, 0.4),
                line_width=1.0
            )
            lines_added += 1
            
        # Y direction lines (parallel to Y axis) 
        for x in np.arange(-size, size + grid_spacing, grid_spacing):
            self.server.scene.add_spline_catmull_rom(
                f"grid_y_{lines_added}",
                positions=np.array([[x, -size, 0.001], [x, size, 0.001]]),
                color=(0.4, 0.4, 0.4),
                line_width=1.0
            )
            lines_added += 1
    
    def update_floor(self):
        """Update floor visibility and appearance"""
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
                
                # Create a simple box as floor (very thin)
                self.server.scene.add_box(
                    name="floor",
                    dimensions=(size * 2, size * 2, 0.02),  # width, height, thickness
                    position=(0.0, 0.0, -0.01),  # Position slightly below z=0
                    color=(0.7, 0.7, 0.7)
                )
                
                print(f"✅ Floor box added: size={size*2}x{size*2}x0.02 at (0,0,-0.01)")
                
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
            
        if self.obj_T is not None:    
            # Add initial object
            initial_obj_mesh = copy.deepcopy(self.mesh).apply_transform(self.obj_T[0])
            self.server.scene.add_mesh_trimesh(
                name=self.object_nm,
                mesh=initial_obj_mesh,
                position=(0.0, 0.0, 0.0),
            )
        
        # Add floor
        self.update_floor()
    
    def update_scene(self, timestep):
        """Update scene for given timestep"""
        with self.server.atomic():
            if self.urdf_vis is not None:
                # Update robot configuration
                self.urdf_vis.update_cfg(self.qpos[timestep])
                
            if self.obj_T is not None:
                # Update object mesh
                transformed_obj_mesh = copy.deepcopy(self.mesh).apply_transform(self.obj_T[timestep])
                self.server.scene.remove_by_name(self.object_nm)
                self.server.scene.add_mesh_trimesh(
                    name=self.object_nm,
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
        
    def update(self):
        """Main update loop - call this in a loop"""
        if self.gui_playing.value:
            if self.num_frames > 1:
                next_timestep = (self.gui_timestep.value + 1) % self.num_frames
                self.gui_timestep.value = next_timestep
            else:
                self.update_scene(0)
            
            
        time.sleep(1.0 / self.gui_framerate.value)
    
    def start_viewer(self):
        """Start the viewer in a loop"""
        print(f"Starting viewer with {self.num_frames} frames")
        print("Visit the URL printed above to view the visualization")
        
        try:
            while True:
                self.update()
        except KeyboardInterrupt:
            print("Viewer stopped")


# Example usage:
if __name__ == "__main__":

    urdf_path = get_robot_urdf_path("xarm", "allegro")
    obj_name = 'pringles'
    obj_mesh, scaled = get_initial_mesh(obj_name, return_type='trimesh', simplify=True,\
                                        device=DEFAULT_DEVICE)
    qpos = []
    for obj_id in os.listdir("pickplace/traj"):
        qpos.append(np.load(f"pickplace/traj/{obj_id}/start_qpos.npy"))
        qpos.append(np.load(f"pickplace/traj/{obj_id}/pick_qpos.npy"))
        qpos.append(np.load(f"pickplace/traj/{obj_id}/end_qpos.npy"))
        qpos.append(np.load(f"pickplace/traj/{obj_id}/place_qpos.npy"))
    qpos = np.concatenate(qpos, axis=0)
    
    c2r = load_latest_C2R()
    
    viewer = ViserViewer(obj_mesh, object_nm='pringles', c2r=c2r, \
                        urdf_path=urdf_path, qpos=qpos)
    viewer.start_viewer()