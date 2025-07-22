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

from paradex.utils.file_io import rsc_path
from paradex.visualization_.robot_module import Robot_Module
from paradex.robot.mimic_joint import parse_inspire

class ViserViewer:
    def __init__(
        self,
        scene_path: Path = Path('/home/jisoo/teserract_nas/processed/spray/1'),
        object_nm: str='spray',
        arm_nm="xarm",
        hand_nm="inspire",
        draw_tg:list = ['mesh','pointcloud','camera']):
            # self.obj_status_path = obj_status_path
            self.object_nm = object_nm
            self.arm_nm = arm_nm
            self.hand_nm = hand_nm
            
            self.c2r = np.load(os.path.join(scene_path, 'C2R.npy'))
            self.obj_T = np.load(os.path.join(scene_path, 'obj_T.npy'))
            self.mesh = trimesh.load(open(f"{rsc_path}/object/{object_nm}/{object_nm}.obj"), file_type="obj")# o3d.io.read_triangle_mesh(f"{rsc_path}/{object_nm}/{object_nm}.obj")
            self.load_server()
            
            
            self.hand_state = np.load(os.path.join(scene_path, hand_nm, "qpos.npy"))
            self.arm_state = np.load(os.path.join(scene_path, arm_nm, "qpos.npy"))
            if self.hand_nm == "inspire":
                self.state = np.concatenate([self.arm_state, parse_inspire(self.hand_state)], axis=1)
            else:
                self.state = np.concatenate([self.arm_state, self.hand_state], axis=1)
                
            self.robot_module = Robot_Module(os.path.join(rsc_path, "robot", f"{arm_nm}_{hand_nm}.urdf"), mesh_tg='all', state = self.state)
            self.num_frames = min(self.robot_module.state.shape[0], self.obj_T.shape[0])
            # import pdb; pdb.set_trace()
            # self.num_frames = len(self.obj_T)
            self.add_frames()
            self.add_player()
            self.prev_timestep = 0
            self.draw_tg = draw_tg
            self.add_mesh()
    
    def load_server(self):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(
                dark_mode=False,
        )
        
        self.server.scene.set_up_direction(self.c2r[:3,2])
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
        add_player = True
        if add_player:
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
                render_png = self.server.gui.add_checkbox("Render to PNG", False)
                gui_framerate = self.server.gui.add_slider(
                    "FPS", min=1, max=60, step=0.1, initial_value=1
                ) # slider 자동 반응
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


            @self.gui_timestep.on_update
            def _(_) -> None:
                current_timestep = self.gui_timestep.value
                with self.server.atomic():
                    # combined_tri_mesh, org_vertices_arr = self.robot_module.get_combined_trimesh(current_timestep, self.c2r)
                    self.server.scene.remove_by_name(f"robot_mesh")

                    self.server.scene.add_mesh_trimesh(
                        name=f"robot_mesh",
                        mesh=self.mesh_list[current_timestep],
                    )  
                    
                    transformed_obj_mesh = copy.deepcopy(self.mesh).apply_transform(self.obj_T[current_timestep])
                    self.server.scene.remove_by_name(f"{self.object_nm}")
                    self.server.scene.add_mesh_trimesh(
                        name=f"{self.object_nm}",
                        mesh=self.obj_mesh_list[current_timestep],
                        # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                        position=(0.0, 0.0, 0.0),
                    )
                     
                    # Toggle visibility.
                    # self.frame_nodes[current_timestep].visible = True
                    # self.frame_nodes[self.prev_timestep].visible = False
                    
                    # self.mesh_nodes[current_timestep].visible = True
                    # self.mesh_nodes[self.prev_timestep].visible = False
                    
                    # self.obj_mesh_nodes[current_timestep].visible = True
                    # self.obj_mesh_nodes[self.prev_timestep].visible = False
                    
                self.prev_timestep = current_timestep
                self.server.flush()  # Optional!
                
            @gui_framerate_options.on_click
            def _(_) -> None:
                gui_framerate.value = int(gui_framerate_options.value)

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

            # @render_png.on_update
            # def _(event: viser.GuiEvent) -> None:
            #     client = event.client
            #     assert client is not None
            #     rendered_img = client.get_render(height=720*2, width=1280*2)
            #     cv2.imwrite(img_render_path/('%05d.jpeg'%(gui_timestep.value)),rendered_img)




    #def add_additional_component(self):    

    def add_frames(self):
        # NOTE: scene setting start
        self.server.scene.add_frame(
            "/frames",
            # wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
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
        
    def add_mesh(self,):
        self.mesh_nodes: list[viser.MeshHandle] = []
        self.obj_mesh_nodes: list[viser.MeshHandle] = []

        obj_frame_handle = None
        self.obj_mesh_handle = None

        contact_arrow_nodes = dict()
        original_vertices = dict()
        self.mesh_dictionary = {}
        
        self.mesh_list = []
        self.obj_mesh_list = []
        
        if 'mesh' in self.draw_tg:
            obj_frame_handle = self.server.scene.add_frame(f"/frames/initial_object", show_axes=False, visible = False)                      
            transformed_obj_mesh = copy.deepcopy(self.mesh).apply_transform(self.obj_T[0])
            
            self.obj_mesh_handle = self.server.scene.add_mesh_trimesh(
                    name=f"/frames/initial_object/{self.object_nm}",
                    mesh=transformed_obj_mesh,
                    # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                    position=(0.0, 0.0, 0.0),
                )
            # self.mesh_dictionary[self.object_nm] = transformed_obj_mesh 
                
            for i in tqdm(range(self.num_frames)):
                combined_tri_mesh, org_vertices_arr = self.robot_module.get_combined_trimesh(i, self.c2r)
                self.mesh_list.append(combined_tri_mesh)
                # Add base frame.
                self.frame_nodes.append(self.server.scene.add_frame(f"/frames/t{i}", show_axes=False, visible = False))
                # self.server.scene.add_mesh_trimesh(
                #     name=f"/frames/t{i}/robot_mesh",
                #     mesh=combined_tri_mesh,
                #     # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                #     # position=(0.0, 0.0, 0.0),
                # )
                # self.mesh_nodes.append(
                #     self.server.scene.add_mesh_trimesh(
                #     name=f"/frames/t{i}/robot_mesh",
                #     mesh=combined_tri_mesh,
                #     # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                #     # position=(0.0, 0.0, 0.0),
                # ))
                # Contact Arrow Visualization
                # self.mesh_dictionary[self.frame_nodes[-1].name] = combined_tri_mesh
                # original_vertices[self.frame_nodes[-1].name] = org_vertices_arr

                transformed_obj_mesh = copy.deepcopy(self.mesh).apply_transform(self.obj_T[i])
                self.obj_mesh_list.append(transformed_obj_mesh)
                # self.server.scene.add_mesh_trimesh(
                #         name=f"/frames/t{i}/{self.object_nm}",
                #         mesh=transformed_obj_mesh,
                #         # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                #         position=(0.0, 0.0, 0.0),
                #     )
                # self.obj_mesh_nodes.append(
                #     self.server.scene.add_mesh_trimesh(
                #         name=f"/frames/t{i}/{self.object_nm}",
                #         mesh=transformed_obj_mesh,
                #         # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                #         position=(0.0, 0.0, 0.0),
                #     )
                # )
                
                # self.mesh_dictionary[self.frame_nodes[-1].name+'_obj'] = transformed_obj_mesh


            # for i, frame_node in enumerate(frame_nodes):
            #     if i != gui_timestep.value:
            #         frame_node.visible = False

            # with self.server.gui.add_folder("Update", expand_by_default=False):
            #     # cam_id_list = [cam_id for cam_id in capture_scene.cam_params]
            #     object_update_button = self.server.gui.add_button("Update Object", disabled=False)
            #     object_file_path = self.server.gui.add_text("Object Status Path", initial_value='None', hint='Ray File in pickle file')

            #     @object_update_button.on_click
            #     def _(_) -> None:
            #         if os.path.exists(object_file_path.value):
            #             obj_mesh, scaled = get_initial_mesh(object_nm, return_type='trimesh', simplify=True, device=DEFAULT_DEVICE)
            #             if object_nm is not None:
            #                 obj_optim_output = pickle.load(open(object_file_path.value,'rb'))
            #                 if scaled:
            #                     obj_R = obj_optim_output['R'].detach().cpu().numpy()
            #                 else:
            #                     obj_R = obj_optim_output['R'].detach().cpu().numpy()*obj_optim_output['scale'].detach().cpu().numpy().item(0)
            #                 obj_t = obj_optim_output['t'].detach().cpu().numpy()
            #                 obj_T = np.eye(4)
            #                 obj_T[:3,:3] = obj_R
            #                 obj_T[:3,3] = obj_t
            #                 obj_mesh.apply_transform(obj_T)

            #             obj_mesh_handle.remove()
                        
            #             obj_mesh_handle = server.scene.add_mesh_trimesh(
            #                 name=f"/frames/initial_object/{object_nm}", 
            #                 mesh=obj_mesh,
            #                 # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            #                 position=(0.0, 0.0, 0.0),
            #             )
            #             mesh_dictionary[object_nm] = obj_mesh
            #         else:
            #             print(f'object path {object_file_path.value} not exist')

            #     robot_update_button = server.gui.add_button("Update Robot", disabled=False)
            #     robot_file_path = server.gui.add_text("Robot Status Path", initial_value='None', hint='Ray File in pickle file')

            #     @robot_update_button.on_click
            #     def _(_) -> None:
            #         if os.path.exists(robot_file_path.value):
            #             updated_state = pickle.load(open(robot_file_path.value,'rb'))
            #             tg_fidx = 0 
            #             robot_module.state[tg_fidx] =  updated_state.detach().cpu().numpy()

            #             mesh_nodes[tg_fidx].remove()
            #             combined_tri_mesh, org_vertices_arr = robot_module.get_combined_trimesh(tg_fidx, capture_scene.C2R)

            #             mesh_nodes[tg_fidx] = server.scene.add_mesh_trimesh(
            #                 name=f"/frames/t{tg_fidx}/robot_mesh",
            #                 mesh=combined_tri_mesh,
            #                 # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            #                 # position=(0.0, 0.0, 0.0),
            #             )
            #             mesh_dictionary[frame_nodes[tg_fidx].name] = combined_tri_mesh
            #             original_vertices[frame_nodes[-1].name] = org_vertices_arr  

            #             server.flush()

            #         else:
            #             print(f'robot path {robot_file_path.value} not exist')
            
    # def draw_camera(self):
    #     draw_camera = True
    #     if draw_camera and 'camera' in draw_tg:
    #         server.scene.add_frame(
    #             "/frames/cameras",
    #             # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
    #             # position=(0, 0, 0),
    #             show_axes=False,
    #         )

    #         camera_frames: list[viser.FrameHandle] = []
    #         for cam_id in tqdm(capture_scene.cam_params):
    #             img_path = scene_path/'video_extracted'/cam_id/'00000.jpeg'
    #             if not os.path.exists(img_path):
    #                 continue
    #             image = iio.imread(img_path)   
    #             image = image[::downsample_factor, ::downsample_factor]

    #             cam_param = capture_scene.cam_params[cam_id]

    #             # cam_param['extrinsic'] is extrinsic to camera
    #             T_camera2world = tf.SE3.from_matrix(
    #                 cam_param['extrinsic']
    #             ).inverse()

    #             frame = server.scene.add_frame(
    #                 f"/frames/cameras/{cam_id}",
    #                 wxyz=T_camera2world.rotation().wxyz,
    #                 position=T_camera2world.translation(),
    #                 axes_length=0.1,
    #                 axes_radius=0.005,
    #             )
    #             camera_frames.append(frame)
    #             # draw frustum
    #             H, W = capture_scene.height, capture_scene.width
    #             fy = cam_param['intrinsic'][1,1]
    #             image = iio.imread(img_path)
    #             image = image[::downsample_factor, ::downsample_factor]
    #             frustum = server.scene.add_camera_frustum(
    #                 f"/frames/cameras/{cam_id}/frustum",
    #                 fov=5 * np.arctan2(H / 2, fy),
    #                 aspect=W / H,
    #                 scale=0.05,
    #                 image=image,
    #                 )
    #             @frustum.on_click
    #             def _(_, frame=frame) -> None:
    #                 print(f'Selected: {frame.name}')
    #                 for client in server.get_clients().values():
    #                     client.camera.wxyz = frame.wxyz
    #                     client.camera.position = frame.position

    def draw_light(self):
        add_light = True
        if add_light:
            server.scene.add_transform_controls(
                "/control_light0", position=(0.0, 5.0, 5.0), scale=2.0
            )
            server.scene.add_label("/control_light0/label", "Point")
            point_light = server.scene.add_light_point(
                name="/control_light0/point_light",
                color=(255, 255, 255),
                intensity=100.0,
                visible=False
            )

            server.scene.add_transform_controls(
                "/control_light1", position=(0.0, -5.0, 5.0), scale=2.0
            )
            server.scene.add_label("/control_light1/label", "Point")
            point_light = server.scene.add_light_point(
                name="/control_light1/point_light",
                color=(255, 255, 255),
                intensity=100.0,
                visible=False
            )
            server.scene.enable_default_lights(True)


        server.scene.add_frame(
            "/global",
            # wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
            # position=(0, 0, 0),
            show_axes=False,
        )
        grid = server.scene.add_grid(
            "/global/grid",
            width=20.0,
            height=20.0,
            wxyz=tf.SO3.from_matrix(capture_scene.C2M[:3,:3]).wxyz,
            position=capture_scene.C2M[:3,3],
            # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            # position=np.array([0.0, 0.0, 0.0]),
        )
        
    def update(self):
        if self.gui_playing.value:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames
        
            # point_nodes[gui_timestep.value].point_size = gui_point_size.value
            # point_nodes[
            #     (gui_timestep.value + 1) % num_frames
            # ].point_size = gui_point_size.value
            
            # if render_png.value:
            #     cur_timestep = gui_timestep.value
            #     for _, client in server.get_clients().items():
            #         rendered_img = client.get_render(height=720*2, width=1280*2)
            #         cv2.imwrite(img_render_path/('%05d.jpeg'%(cur_timestep)),rendered_img)
            

        # time.sleep(1.0 / gui_framerate.value)
        
