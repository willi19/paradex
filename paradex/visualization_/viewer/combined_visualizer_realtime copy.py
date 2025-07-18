"""
env: hoifield_310
play /stop / next frame/ previous frame
slider 
floor vis (on/off)
debugline / debugpoint / debug sphere (on/off)
rendering the current 3D scene into images
Visualizing the 3D scene changes during optimization
"""
import __init__
from __init__ import PROJECT_PATH
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

from utils.vis_utils import read_mesh, get_initial_mesh
from utils.scene import Scene
# from utils.vis_utils_o3d import simpleViewer, get_camLinesSet, get_pointcloud
from utils.robot_module import robot_info, robot_asset_file, Robot_Module
from utils.vis_utils import pick_corners, get_ray, get_colored_mesh
from utils.renderer_utils import convert_meshtrimesh2py3d
# from visualize.viser_utils import MESH_DIR
from utils.io import get_obj_info, get_obj_realtime_t
from utils.robot_asset.contact_info import contact_tg, sensororder, get_categorized_mesh

import torch
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

nas_path = Path(os.environ['NAS_PATH'])
CACHE_PATH = Path(__file__).parent/'cache'

def main(
    scene_path: Path = Path('/home/jisoo/teserract_nas/processed/spray/1'),
    object_nm: str='spray',
    downsample_factor=4,
    draw_tg:list = ['mesh','pointcloud','camera'],
) -> None:
    
    mesh_dictionary = {}
    img_render_path = scene_path/'rendered_viser'
    os.makedirs(img_render_path, exist_ok=True)
    
    obj_initial_T, obj_mesh = None, None
    obj_trajectory = {}

    obj_status_path = None
    if 'mesh' in draw_tg:
        # Get Object Mesh Information
        if object_nm is not None:
            obj_mesh, obj_initial_T, obj_trajectory = get_obj_info(scene_path, object_nm, \
                                                        obj_status_path, device=DEFAULT_DEVICE)
            assert isinstance(obj_mesh, trimesh.Trimesh)
        # vertices = mesh.vertices
        # faces = mesh.faces

    # Load Scene
    capture_scene = Scene(scene_path, rescale_factor=0.5)
    capture_scene.get_renderer() # prepare pytorch3d renderer
    cam_params = capture_scene.cam_params
    robot_module = Robot_Module(mesh_tg='all', state = capture_scene.robot_traj, contact = capture_scene.contact) 
    # both state, contact are known in realtime
    robot_link2vertex_mapping = robot_module.link2vertex_mapping
    # num_frames = robot_module.state.shape[0]
    # num_frames = 200
    
    server = viser.ViserServer()
    server.gui.configure_theme(
            dark_mode=False,
    )
    
    if capture_scene.C2M is not None:
        server.scene.set_up_direction(capture_scene.C2M[:3,1])
    else:
        server.scene.set_up_direction("+y")
    server.scene.world_axes

    # z_near, z_far setting
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        """For each client that connects, create GUI elements for adjusting the
        near/far clipping planes."""
        
        client.camera.far = 20.0

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
            

    add_player = True
    if add_player:
        # add player 
        with server.gui.add_folder("Playback"):
            gui_playing = server.gui.add_checkbox("Playing", True)
            render_png = server.gui.add_checkbox("Render to PNG", False)
            gui_framerate = server.gui.add_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=1
            ) # slider 자동 반응
            gui_framerate_options = server.gui.add_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

            gui_up = server.gui.add_vector3(
                "Up Direction",
                initial_value=(0.0, -1.0, 1.0),
                step=0.01,
            )
            @gui_up.on_update
            def _(_) -> None:
                server.scene.set_up_direction(gui_up.value)

        # Disable frame controls when we're playing.
        @gui_playing.on_update
        def _(_) -> None:
            pass

        @render_png.on_update
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            rendered_img = client.get_render(height=720*2, width=1280*2)
            cv2.imwrite(img_render_path/('%05d.jpeg'%(0)), rendered_img)


        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)


        # @gui_timestep.on_update
        # def _(_) -> None:
        #     nonlocal prev_timestep
        #     current_timestep = 0
        #     with server.atomic():
        #         # Toggle visibility.
        #         frame_nodes[current_timestep].visible = True
        #         frame_nodes[prev_timestep].visible = False
        #         for contact_arrow_mesh in contact_arrow_nodes[current_timestep]:
        #            contact_arrow_mesh.visible=drawing_arrow.value
        #     prev_timestep = current_timestep
        #     server.flush()  # Optional!

    
    add_additional_component = True
    line_segments = []
    point_segments = []
    if add_additional_component:
        # Update Current status
        # Ray
        with server.gui.add_folder("Ray", expand_by_default=False):
            # cam_id_list = [cam_id for cam_id in capture_scene.cam_params]
            add_ray_button = server.gui.add_button("Add Ray", disabled=False)
            ray_file_path = server.gui.add_text("Ray File Path", initial_value='None', hint='Ray File in pickle file')
            remove_ray_button = server.gui.add_button("Remove All Rays", disabled=False)
            # camera_dropdown = server.gui.add_dropdown(
            #     "Target Camera", (cam_id_list), initial_value=cam_id_list[0]
            # ) # camera_dropdown.value

        @add_ray_button.on_click
        def _(_) -> None:
            if os.path.exists(ray_file_path.value):
                '''
                NOTE: if not exist: run utils/pick_point.py
                '''
                picked_points = pickle.load(open(ray_file_path.value,'rb'))
                for cam_id in picked_points:
                    point = picked_points[cam_id]
                    cam_param = capture_scene.cam_params[cam_id]
                    
                    T_camera2world = tf.SE3.from_matrix(
                        cam_param['extrinsic']
                    ).inverse()
                    
                    intrinsic_matrix = cam_param['intrinsic']
                    ray_center_world, ray_point_world = get_ray((point), T_camera2world.as_matrix(), intrinsic_matrix)
                    points = np.vstack([ray_center_world, ray_point_world])[np.newaxis,]
                    colors = np.array([[255,0,0],[255,0,0]])[np.newaxis,]
                    line = server.scene.add_line_segments(f"/line_segments/line{len(line_segments)}", points=points, colors=colors, line_width=3.0)
                    line_segments.append(line)


        @remove_ray_button.on_click
        def _(_) -> None:
            for line in line_segments:
                line.remove()
                line_segments = []


        # Add Point
        with server.gui.add_folder("Point", expand_by_default=False):
            # cam_id_list = [cam_id for cam_id in capture_scene.cam_params]
            add_point_button = server.gui.add_button("Add Point", disabled=False)
            point_vector3 = server.gui.add_vector3(
                    "Position(3D)",
                    initial_value=(0.0, 0.0, 0.0),
                    step=0.25,
                )
            remove_points_button = server.gui.add_button("Remove All Points", disabled=False)
            # camera_dropdown = server.gui.add_dropdown(
            #     "Target Camera", (cam_id_list), initial_value=cam_id_list[0]
            # ) # camera_dropdown.value

        @add_point_button.on_click
        def _(_) -> None:
            point_arr = np.array([point_vector3.value], dtype=np.float32)

            point = server.scene.add_point_cloud(
                f"/point_segments/point{len(point_segments)}",
                points=point_arr,
                colors=(
                    np.array([[0,0,255]])
                ).astype(np.uint8),
                point_shape=['circle'],
                point_size=0.01
            )
            point_segments.append(point)
        
        @remove_points_button.on_click
        def _(_) -> None:
            for point in point_segments:
                point.remove()
            point_segments = []

        
        # Render on Image
        render_directory = scene_path/'rendered_scene'
        os.makedirs(render_directory, exist_ok=True)
        with server.gui.add_folder("Renderer", expand_by_default=False):
            render_tg_list = ['all']+[cam_id for cam_id in capture_scene.cam_params]
            render_button = server.gui.add_button("Render", disabled=False)
            render_tg_dropdown = server.gui.add_dropdown(
                "Target Camera", (render_tg_list), initial_value=render_tg_list[0]
            ) # camera_dropdown.value

        # @render_button.on_click
        # def _(_) -> None:
        #     render_tg = [cam_id for cam_id in capture_scene.cam_params] if render_tg_dropdown.value=='all' else [render_tg_dropdown.value]
        #     # IF object trajectory not exist -> use first frame.
        #     mesh_tg_list = [[mesh_dictionary[frame_node.name], \
        #                      mesh_dictionary[frame_node.name+'_obj'] if (frame_node.name+'_obj') in mesh_dictionary else mesh_dictionary['frames/t0_obj'] ] \
        #                     for frame_node in frame_nodes if frame_node.visible][0]
        #     mesh_combined_pytorch3d = convert_meshtrimesh2py3d(mesh_tg_list, DEFAULT_DEVICE)

        #     for cam_id in render_tg:
        #         rendered_rgb, rendered_silhouette = capture_scene.get_projected_img(cam_id, 0, mesh_combined_pytorch3d[0])
        #         iio.imwrite(render_directory/('rgb_%s_%05d.jpeg'%(cam_id, 0)), rendered_rgb)
        #         iio.imwrite(render_directory/('sil_%s_%05d.jpeg'%(cam_id, 0)), rendered_silhouette)
        
        #         print(f'RGB and Silhouette saved in %s %05d'%(cam_id, 0))


    contact_line_segments = []
    contact_point_segments = []
    add_contact_module = True
    if add_contact_module:  
        with server.gui.add_folder("Contact", expand_by_default=False):
            compute_contact_button = server.gui.add_button("Compute Contact", disabled=False)
            clear_contact_button = server.gui.add_button("Clear Contact", disabled=False)
            drawing_arrow = server.gui.add_checkbox("Draw Arrow", True)

            contact_thres = server.gui.add_number(
                "Contact Thres",
                initial_value=0.02,
                step=0.005,
            )
            # @contact_thres.on_update
            # def _(_) -> None:
            #     server.scene.set_up_direction(gui_up.value)


        @drawing_arrow.on_update
        def _(_) -> None:
            # drawing_arrow.value
            current_timestep = 0
            with server.atomic():
                for contact_arrow_mesh in contact_arrow_nodes[current_timestep]:
                   contact_arrow_mesh.visible = drawing_arrow.value
            server.flush()  # Optional!

        @compute_contact_button.on_click
        def _(_) -> None:
            # Get NN at point
            current_timestep = 0
            link_T =  robot_module.get_T_dict(current_timestep)
            
            cur_frame_node_name = f"/frames/t{current_timestep}"
            obj_mesh_nm = cur_frame_node_name+'_obj' if cur_frame_node_name+'_obj' in mesh_dictionary else object_nm
            cur_object_mesh = mesh_dictionary[obj_mesh_nm] 
            obj_vertices = cur_object_mesh.vertices
            
            combined_robot_vertices = original_vertices[cur_frame_node_name]

            points = []
            colors = []

            object_min_dist = np.ones(obj_vertices.shape[0])
            object_min_dist_tg = np.zeros(obj_vertices.shape[0])
            object_min_dist_tg.fill(-1) # Set to -1 (not contacted)
            
            #  [mesh_dictionary[obj_frame.name]]+[mesh_dictionary[frame_node.name] for frame_node in frame_nodes if frame_node.visible]object_mesh_vertices = 
            obj_tree = cKDTree(obj_vertices)
            for tidx, hand_tg in enumerate(contact_tg):
                vertex_range = robot_link2vertex_mapping[hand_tg]
                robot_part_vertices = combined_robot_vertices[vertex_range[0]:vertex_range[1]]
                distances, indices = obj_tree.query(robot_part_vertices, k=1)
                robot_index = np.argmin(distances)
                obj_index = indices[robot_index]
                robot_point = robot_part_vertices[robot_index]
                obj_point = obj_vertices[obj_index]
                points.append([robot_point, obj_point])
                colors.append([[0,0,255],[0,0,255]])

                hand_tree = cKDTree(robot_part_vertices)
                distance_obj2hand = hand_tree.query(obj_vertices, k=1)[0]
                update_filter = distance_obj2hand<object_min_dist
                object_min_dist[update_filter] = distance_obj2hand[update_filter]
                object_min_dist_tg[update_filter] = tidx

            points = np.stack(points)
            colors = np.stack(colors)

            line = server.scene.add_line_segments(f"/line_segments/contact_line{len(line_segments)}", points=points, colors=colors, line_width=3.0)
            contact_line_segments.append(line)

            # add point
            for point in np.vstack(points):
                point = server.scene.add_point_cloud(
                    f"/point_segments/point{len(contact_point_segments)}",
                    points=point[np.newaxis,...],
                    colors=(
                        np.array([[255,0,0]])
                    ).astype(np.uint8),
                    point_size=0.001,
                    point_shape='circle'
                )
                contact_point_segments.append(point)

            # Transfer Contact to Object Mesh
            threshold_filter = object_min_dist>contact_thres.value
            object_min_dist_tg[threshold_filter] = -1
            object_min_dist_sensor = np.zeros_like(object_min_dist)

            current_contact = robot_module.contact[current_timestep]

            for sidx, sensor_nm in enumerate(sensororder):
                link_nm = sensor_nm[:-2]
                contact_value = current_contact[sidx]
                tg_filter = np.logical_and(object_min_dist_tg==contact_tg.index(link_nm),object_min_dist_sensor<contact_value)
                object_min_dist_sensor[tg_filter] = contact_value

            save_directory = scene_path/'contact'
            os.makedirs(save_directory, exist_ok=True)

            _ = get_colored_mesh(mesh_dictionary[obj_mesh_nm], object_min_dist, save_path=str(save_directory/'debug_distance.obj'), cmap_nm='viridis_r')
            _ = get_categorized_mesh(mesh_dictionary[obj_mesh_nm], object_min_dist_tg, save_path=str(save_directory/'debug_part.obj'))
            _ = get_colored_mesh(mesh_dictionary[obj_mesh_nm], object_min_dist_sensor, save_path=str(save_directory/'debug_sensor.obj'), cmap_nm='viridis')

            pickle.dump(object_min_dist, open(save_directory/'object_min_dist.pickle','wb'))
            pickle.dump(object_min_dist_tg, open(save_directory/'object_min_dist_tg.pickle','wb'))
            pickle.dump(object_min_dist_sensor, open(save_directory/'object_min_dist_sensor.pickle','wb'))

            robot_state = capture_scene.robot_traj[current_timestep] # 6 for arm, and 16 for hand
            pickle.dump({'current_timestep':current_timestep,'robot_traj':robot_state, 'link_T':link_T}, \
                        open(save_directory/'robot_pose_in_contact.pickle','wb'))
            print("Transfer ended")


        @clear_contact_button.on_click
        def _(_) -> None:
            nonlocal contact_line_segments, contact_point_segments
            for line in contact_line_segments:
                line.remove()
            contact_line_segments = []
            for point in contact_point_segments:
                point.remove()
            contact_point_segments = []


    # NOTE: scene setting start
    server.scene.add_frame(
        "/frames",
        # wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    if capture_scene.C2M is not None:
        server.scene.add_frame(
            "/system",
            wxyz=tf.SO3.from_matrix(capture_scene.C2M[:3,:3]).wxyz,
            position=capture_scene.C2M[:3,3],
            show_axes=True,
            axes_radius=0.01
        )

    frame_nodes: list[viser.FrameHandle] = []
    mesh_nodes: list[viser.MeshHandle] = []
    obj_mesh_nodes: list[viser.MeshHandle] = []

    obj_frame_handle = None
    obj_mesh_handle = None
    robot_mesh_handle = None

    contact_arrow_nodes = dict()
    original_vertices = dict()

    obj_frame_handle = server.scene.add_frame(f"/frames/object", show_axes=False, visible = False)   
    robot_frame_handle = server.scene.add_frame(f"/frames/robot", show_axes=False, visible = False)   
    
    def update_scene():

        transl = get_obj_realtime_t(scene_path=CACHE_PATH, object_nm=object_nm)
        
        
        if obj_mesh is not None and transl is not None:
            obj_initial_T = np.eye(4)
            obj_initial_T[:3,3] = transl
            obj_initial_T = capture_scene.C2R@obj_initial_T

            transformed_obj_mesh = copy.deepcopy(obj_mesh).apply_transform(obj_initial_T)
            obj_mesh_handle = server.scene.add_mesh_trimesh(
                    name=f"/frames/initial_object/{object_nm}",
                    mesh=transformed_obj_mesh,
                    # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                    position=(0.0, 0.0, 0.0),
                )
            mesh_dictionary[object_nm] = transformed_obj_mesh 

        # Root STatus
        if os.path.exists(CACHE_PATH/'state.npy'):

            # hand_action = np.load(scene_path/'hand'/'action.npy') # FX16
            # hand_state = np.load(scene_path/'hand_demo'/'state.npy') # FX16
            # arm_action = np.load(scene_path/'arm'/'action.npy') # FX6
            arm_state = np.load(str(CACHE_PATH/'state.npy')) # FX6
            robot_state = np.zeros(22)
            robot_state[:6] = arm_state

            # robot_state = np.concatenate([arm_state, hand_state], axis=1)[0] # FX22
            

            # mesh_nodes[tg_fidx].remove()
            combined_tri_mesh, org_vertices_arr = robot_module.get_combined_trimesh(0, capture_scene.C2R, state_realtime=robot_state)

            robot_mesh_handle = server.scene.add_mesh_trimesh(
                name=f"/frames/robot",
                mesh=combined_tri_mesh,
                # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                # position=(0.0, 0.0, 0.0),
            )
            mesh_dictionary['robot'] = combined_tri_mesh
            # original_vertices[frame_nodes[-1].name] = org_vertices_arr  

            server.flush()

        
        if os.path.exists(CACHE_PATH/'hand_detections.npy'):
        
            # for line in line_segments:
            #     line.remove()
            line_segments = []
    
            hand_detections = np.load(str(CACHE_PATH/'hand_detections.npy'), allow_pickle=True).item()


            media_hand_index = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]]
            for side in hand_detections:
                for line_pair in media_hand_index:
                    st_idx, ed_idx = line_pair
                    if st_idx in hand_detections[side] and ed_idx in hand_detections[side]:
                        st_pts = capture_scene.C2R[:3,:3]@hand_detections[side][st_idx]+capture_scene.C2R[:3,3]
                        ed_pts = capture_scene.C2R[:3,:3]@hand_detections[side][ed_idx]+capture_scene.C2R[:3,3]
                        points = np.vstack([st_pts, ed_pts])[np.newaxis,]
                        if side.lower()=='left':
                            colors = np.array([[0,0,255],[0,0,255]])[np.newaxis,]
                        else:
                            colors = np.array([[255,0,0],[255,0,0]])[np.newaxis,]
                        line = server.scene.add_line_segments(f"/line_segments/line_{side}_{st_idx}_{ed_idx}", points=points, colors=colors, line_width=3.0)
                        line_segments.append(line)

    

    draw_camera = True
    if draw_camera and 'camera' in draw_tg:
        server.scene.add_frame(
            "/frames/cameras",
            # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            # position=(0, 0, 0),
            show_axes=False,
        )

        camera_frames: list[viser.FrameHandle] = []
        for cam_id in tqdm(capture_scene.cam_params):
            # /home/jisoo/teserract_nas/demo_250618/pringles/0/images_undistorted
            img_path = scene_path/'images_undistorted'/f'{cam_id}.jpg'
            if not os.path.exists(img_path):
                continue
            image = iio.imread(img_path)   
            image = image[::downsample_factor, ::downsample_factor]*255

            cam_param = capture_scene.cam_params[cam_id]

            # cam_param['extrinsic'] is extrinsic to camera
            T_camera2world = tf.SE3.from_matrix(
                cam_param['extrinsic']
            ).inverse()

            frame = server.scene.add_frame(
                f"/frames/cameras/{cam_id}",
                wxyz=T_camera2world.rotation().wxyz,
                position=T_camera2world.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            camera_frames.append(frame)
            # draw frustum
            H, W = capture_scene.height, capture_scene.width
            fy = cam_param['intrinsic'][1,1]
            image = iio.imread(img_path)
            image = image[::downsample_factor, ::downsample_factor]*255
            frustum = server.scene.add_camera_frustum(
                f"/frames/cameras/{cam_id}/frustum",
                fov=5 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.05,
                image=image,
                )
            @frustum.on_click
            def _(_, frame=frame) -> None:
                print(f'Selected: {frame.name}')
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

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
        wxyz=tf.SO3.from_matrix(capture_scene.C2M[:3,:3]).wxyz if capture_scene.C2M is not None else np.array((1.0, 0.0, 0.0, 0.0)),
        position=capture_scene.C2M[:3,3] if capture_scene.C2M is not None else np.array((0.0, 0.0, 0.0)),
        # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
        # position=np.array([0.0, 0.0, 0.0]),
    )

    
    while True:
        if gui_playing.value:
            # update 
            update_scene()

        time.sleep(0.5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
        scene_path: Path = Path('/home/jisoo/teserract_nas/demo_250618/pringles/0'),
        object_nm: str='spray',
        downsample_factor=4,
    '''
    parser.add_argument('--scene_path', type=str, required=True)
    # parser.add_argument('--object_nm', type=str, required=True)
    parser.add_argument('--downsample_factor', type=int, default=4) # image resolution down sample 
    args = parser.parse_args()

    object_nm = args.scene_path.split("/")[-2]

    main(Path(args.scene_path), object_nm, args.downsample_factor, draw_tg=['mesh', 'camera'])
