import os
import numpy as np
from isaacgym import gymapi
import pickle
from scipy.spatial.transform import Rotation as R
import cv2
from paradex.utils.file_io import rsc_path


class simulator:
    def __init__(
        self,
        obj_name,
        view_physics=True,
        view_replay=True,
        headless=False,
        save_video=True,
        save_state=False,
        fixed=False,
        add_plane=True,
        camera_param=None,
        contact=None
    ):
        self.fixed = fixed
        self.obj_name = obj_name

        self.view_physics = view_physics
        self.view_replay = view_replay

        self.gym = gymapi.acquire_gym()
        self.sim = self.generate_sim()

        if add_plane:
            self.add_plane()

        self.add_assets(obj_name)

        (self.env, self.actor_handle) = self.load_env()

        self.headless = headless
        self.save_video = save_video
        self.save_state = save_state

        if not headless:
            self.set_viewer()
        self.step_idx = 0
        # self.camera_param = camera_param

    def set_savepath(self, video_path, state_path):
        if self.save_video:
            self.out = {}
            for name, camera_handle in self.camera_handle.items():
                if len(self.camera_handle) == 1:
                    os.makedirs(os.path.dirname(video_path), exist_ok=True)
                    output_filename = video_path
                else:
                    os.makedirs(video_path, exist_ok=True)
                    output_filename = os.path.join(video_path, f"{name}.mp4")
    
                print("save to", output_filename)

                self.frame_width = 2048
                self.frame_height = 1536
                self.fps = 30
                self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out[name] = cv2.VideoWriter(
                    output_filename,
                    self.fourcc,
                    self.fps,
                    (self.frame_width, self.frame_height),
                )

        if self.save_state:
            self.state_path = state_path
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)

            self.history = {"robot": [], "object": []}

    def set_camera(self, name, camera_position, camera_lookat):
        self.gym.set_camera_location(
            self.camera_handle[name],
            self.env,
            gymapi.Vec3(*camera_position),
            gymapi.Vec3(*camera_lookat),
        )

    def save_stateinfo(self):

        robot_dof_state = self.gym.get_actor_dof_states(
            self.env, self.actor_handle["robot"], gymapi.STATE_POS
        )
        self.history["robot"].append(robot_dof_state["pos"])

        object_rb_state = self.gym.get_actor_rigid_body_states(
            self.env, self.actor_handle["object"], gymapi.STATE_POS
        )

        obj_quat = np.array(
            [
                object_rb_state["pose"]["r"]["x"][0],
                object_rb_state["pose"]["r"]["y"][0],
                object_rb_state["pose"]["r"]["z"][0],
                object_rb_state["pose"]["r"]["w"][0],
            ]
        )
        obj_rotmat = R.from_quat(obj_quat).as_matrix()

        obj_pos = np.array(
            [
                object_rb_state["pose"]["p"]["x"][0],
                object_rb_state["pose"]["p"]["y"][0],
                object_rb_state["pose"]["p"]["z"][0],
            ]
        )

        obj_T = np.eye(4)
        obj_T[:3, :3] = obj_rotmat
        obj_T[:3, 3] = obj_pos

        self.history["object"].append(obj_T)

    def step(self, action, viz_action, obj_pose, sphere_pos=None):
        if self.obj_name is not None:
            assert obj_pose.shape == (4, 4)

            obj_quat = R.from_matrix(obj_pose[:3, :3]).as_quat()
            obj_pos = obj_pose[:3, 3]

        action = action.astype(np.float32)
        viz_action = viz_action.astype(np.float32)

        if self.view_physics:
            if self.step_idx == 0:
                robot_dof_state = self.gym.get_actor_dof_states(
                    self.env, self.actor_handle["robot"], gymapi.STATE_POS
                )

                robot_dof_state["pos"] = action

                self.gym.set_actor_dof_states(
                    self.env,
                    self.actor_handle["robot"],
                    robot_dof_state,
                    gymapi.STATE_POS,
                )
                if self.obj_name is not None:
                    object_rb_state = self.gym.get_actor_rigid_body_states(
                        self.env, self.actor_handle["object"], gymapi.STATE_POS
                    )
                    object_rb_state["pose"]["r"].fill(
                        (obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3])
                    )
                    object_rb_state["pose"]["p"].fill((obj_pos[0], obj_pos[1], obj_pos[2]))

                    self.gym.set_actor_rigid_body_states(
                        self.env,
                        self.actor_handle["object"],
                        object_rb_state,
                        gymapi.STATE_POS,
                    )

            self.save_stateinfo()
            self.gym.set_actor_dof_position_targets(
                self.env, self.actor_handle["robot"], action
            )

        if self.view_replay:
            robot_dof_state = self.gym.get_actor_dof_states(
                self.env, self.actor_handle["robot_replay"], gymapi.STATE_POS
            )
            robot_dof_state["pos"] = viz_action

            self.gym.set_actor_dof_states(
                self.env,
                self.actor_handle["robot_replay"],
                robot_dof_state,
                gymapi.STATE_POS,
            )
            if self.obj_name is not None:
                object_rb_state = self.gym.get_actor_rigid_body_states(
                    self.env, self.actor_handle["object_replay"], gymapi.STATE_POS
                )
                object_rb_state["pose"]["r"].fill(
                    (obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3])
                )
                object_rb_state["pose"]["p"].fill((obj_pos[0], obj_pos[1], obj_pos[2]))

                self.gym.set_actor_rigid_body_states(
                    self.env,
                    self.actor_handle["object_replay"],
                    object_rb_state,
                    gymapi.STATE_POS,
                )
            robot_rb_state = self.gym.get_actor_rigid_body_states(
                self.env, self.actor_handle["robot_replay"], gymapi.STATE_POS
            )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if self.save_video or not self.headless:
            self.gym.step_graphics(self.sim)
        
        if not self.headless:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        if self.save_video:
            self.gym.render_all_camera_sensors(self.sim)

            for name, camera_handle in self.camera_handle.items():
                frame = self.gym.get_camera_image(
                    self.sim, self.env, camera_handle, gymapi.IMAGE_COLOR
                ).astype(np.uint8)

                frame = frame.reshape((self.frame_height, self.frame_width, 4))[:, :, :3]
                frame = frame[:, :, ::-1]
                self.out[name].write(frame)
        self.step_idx += 1

    def save(self):
        self.step_idx = 0
        if self.save_video:
            for name, out in self.out.items():
                out.release()

        if self.save_state:
            pickle.dump(self.history, open(self.state_path, "wb"))
            self.history = {}

    def load_env(self):
        actor_handle = {}
        env = self.gym.create_env(
            self.sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1
        )

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0, 0, 0)

        # 객체 추가
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(0.5, 0.0, 0.0)

        if self.view_physics:
            actor_handle["robot"] = self.gym.create_actor(
                env, self.assets["robot"], robot_pose, "robot", 1, 0
            )

            props = self.gym.get_actor_dof_properties(env, actor_handle["robot"])
            props["driveMode"].fill(gymapi.DOF_MODE_POS)

            props["stiffness"][:6] = 1000.0  # pgain for arm
            props["damping"][:6] = 10.0  # dgain for arm

            props["stiffness"][6:] = 500.0  # pgain for hand
            props["damping"][6:] = 10.0  # dgain for hand

            self.gym.set_actor_dof_properties(env, actor_handle["robot"], props)

            if self.obj_name is not None:
                actor_handle["object"] = self.gym.create_actor(
                    env, self.assets["object"], object_pose, "object", 1, 0
                )

                obj_props = self.gym.get_actor_rigid_shape_properties(
                    env, actor_handle["object"]
                )

                obj_props[0].restitution = 0.1
                obj_props[0].friction = 0.0

                self.gym.set_actor_rigid_shape_properties(
                    env, actor_handle["object"], obj_props
                )


        if self.view_replay:

            actor_handle["robot_replay"] = self.gym.create_actor(
                env, self.assets["vis_robot"], robot_pose, "robot_replay", 2, 1
            )

            if self.obj_name is not None:
                actor_handle["object_replay"] = self.gym.create_actor(
                    env, self.assets["vis_object"], object_pose, "object_replay", 3, 0
                )

            for obj_name in ["robot_replay", "object_replay"]:
                if self.obj_name is None and obj_name == "object_replay":
                    continue
                rigid_body_props = self.gym.get_actor_rigid_body_properties(
                    env, actor_handle[obj_name]
                )

                for prop in rigid_body_props:
                    prop.flags = (
                        gymapi.RIGID_BODY_DISABLE_GRAVITY
                    )  # Disable gravity flag

                self.gym.set_actor_rigid_body_properties(
                    env,
                    actor_handle[obj_name],
                    rigid_body_props,
                    recomputeInertia=False,
                )

                props = self.gym.get_actor_dof_properties(env, actor_handle[obj_name])
                props["driveMode"].fill(gymapi.DOF_MODE_NONE)
                self.gym.set_actor_dof_properties(env, actor_handle[obj_name], props)

            for i in range(32):
                self.gym.set_rigid_body_color(
                    env,
                    actor_handle["robot_replay"],
                    i,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(0.4, 0.4, 0.6),
                )

            if self.obj_name is not None:
                self.gym.set_rigid_body_color(
                    env,
                    actor_handle["object_replay"],
                    0,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(0.4, 0.4, 0.6),
                )

        return env, actor_handle

    def set_color(self, color_dict):
        
        for idx, color in color_dict.items():
            self.gym.set_rigid_body_color(
                self.env,
                self.actor_handle["robot_replay"],
                idx,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(*color),
            )
    def add_plane(self):
        # configure the ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0.0525
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 0.8
        plane_params.restitution = 0

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

    def set_viewer(self, cam_params=None):
        if cam_params is None:
            cam_props = gymapi.CameraProperties()

            viewer = self.gym.create_viewer(self.sim, cam_props)
            # Define camera position (e.g., behind and slightly above the actor)
            camera_position = np.array([2, 0, 1])

            # Calculate direction vector (camera looks at the actor)
            camera_direction = -camera_position
            camera_direction = camera_direction / np.linalg.norm(
                camera_direction
            )  # Normalize

            # Set viewer camera transform
            self.gym.viewer_camera_look_at(
                viewer, None, gymapi.Vec3(*camera_position), gymapi.Vec3(0, 0, 0)
            )
            self.viewer = viewer
        else:
            # Add a camera to capture frames
            int_mat, ext_mat = cam_params
            # camera_props = gymapi.CameraProperties()
            
            # fx = int_mat[0,0]
            # fy = int_mat[1,1]
            # camera_props.horizontal_fov = np.arctan(2048/(2*fx)) * 2 * 180 / np.pi
            # # camera_props.vertical_fov = np.arctan(1536/(2*fy)) * 2 * 180 / np.pi

            # camera_props.width = 2048
            # camera_props.height = 1536
            # # viewer = self.gym.create_viewer(self.sim, camera_props)

            position = ext_mat[:3, 3].tolist()
            lookat = (position + ext_mat[:3, 2] ).tolist()
            
            self.gym.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(*position),
                gymapi.Vec3(*lookat),
            )
            # self.viewer = viewer

    def generate_sim(self):
        # 시뮬레이션 설정
        sim_params = gymapi.SimParams()
        # set common parameters
        sim_params.dt = 1 / 30
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        sim_params.flex.solver_type = 5

        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        return sim

    def add_assets(self, obj_name):
        self.assets = {}

        asset_root = rsc_path
        if self.fixed:
            robot_asset_file = "xarm6/xarm6_allegro_wrist_mounted_rotate.urdf"
        else:
            robot_asset_file = "allegro/allegro_contact.urdf"

        if obj_name is not None:
            object_asset_file = f"{obj_name}/{obj_name}.urdf"

        if self.view_physics:
            robot_asset_options = gymapi.AssetOptions()
            robot_asset_options.fix_base_link = True
            robot_asset_options.armature = 0.001
            robot_asset_options.thickness = 0.002
            
            if self.obj_name is not None:
                object_asset_options = gymapi.AssetOptions()
                object_asset_options.override_inertia = True
                object_asset_options.mesh_normal_mode = (
                    gymapi.COMPUTE_PER_VERTEX
                )  # Use per-vertex normals
                object_asset_options.vhacd_enabled = True
                object_asset_options.vhacd_params = gymapi.VhacdParams()
                object_asset_options.vhacd_params.resolution = 300000

                robot_asset = self.gym.load_asset(
                    self.sim, asset_root, robot_asset_file, robot_asset_options
                )
                object_asset = self.gym.load_asset(
                    self.sim, asset_root, object_asset_file, object_asset_options
                )
                self.assets["object"] = object_asset

            self.assets["robot"] = robot_asset

        if self.view_replay:
            if self.obj_name != None:
                vis_object_asset_options = gymapi.AssetOptions()
                vis_object_asset_options.disable_gravity = True
                vis_object_asset = self.gym.load_asset(
                    self.sim, asset_root, object_asset_file, vis_object_asset_options
                )
                self.assets["vis_object"] = vis_object_asset

            vis_robot_asset_options = gymapi.AssetOptions()
            vis_robot_asset_options.disable_gravity = True
            vis_robot_asset_options.fix_base_link = True
            vis_robot_asset = self.gym.load_asset(
                self.sim, asset_root, robot_asset_file, vis_robot_asset_options
            )

            self.assets["vis_robot"] = vis_robot_asset


    def visualize_camera(self, cam_param_dict):
        """
        Loads and visualizes a camera URDF model in Isaac Gym.
        
        :param gym: Isaac Gym API instance
        :param sim: Isaac Gym simulation instance
        :param env: Simulation environment
        :param urdf_path: Path to the URDF file defining the camera model
        :param position: Tuple (x, y, z) for the camera position
        :param orientation: Quaternion (x, y, z, w) for camera orientation
        """
        cam_urdf_path = os.path.join(rsc_path, "camera", "camera.urdf")
        if "camera" not in self.assets:
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True  # Ensure the camera is fixed in place
        
            camera_asset = self.gym.load_asset(self.sim, os.path.dirname(cam_urdf_path), os.path.basename(cam_urdf_path), asset_options)
            self.assets["camera"] = camera_asset

        for serial_num, cam_param in cam_param_dict.items():
            _, ext_mat = cam_param
            if serial_num in self.actor_handle:
                continue
            camera_pose = gymapi.Transform()    
            camera_pose.p = gymapi.Vec3(*ext_mat[:3, 3])
            camera_pose.r = gymapi.Quat(*R.from_matrix(ext_mat[:3, :3]).as_quat())

            self.actor_handle[serial_num] = self.gym.create_actor(self.env, camera_asset, camera_pose, serial_num, 0, 0)

            if serial_num == "23029839":
                self.gym.set_rigid_body_color(self.env, self.actor_handle[serial_num], 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.4, 0.4, 0.6))
        
        return
    def load_camera(self, camera_param_dict=None):
        self.camera_handle = {}
        if camera_param_dict is None:
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75.0
            camera_props.width = 2048
            camera_props.height = 1536
            camera_handle = self.gym.create_camera_sensor(self.env, camera_props)

            self.gym.set_camera_location(
                camera_handle,
                self.env,
                gymapi.Vec3(0.3, 0, 0),
                gymapi.Vec3(0, 0, 0),
            )
            self.camera_handle[0] = camera_handle
            return
                                                                  
        for serial_num, camera_param in camera_param_dict.items():
            # Add a camera to capture frames
            int_mat, ext_mat = camera_param
            camera_props = gymapi.CameraProperties()
            
            fx = int_mat[0,0]
            fy = int_mat[1,1]
            camera_props.horizontal_fov = np.arctan(2048/(2*fx)) * 2 * 180 / np.pi
            # camera_props.vertical_fov = np.arctan(1536/(2*fy)) * 2 * 180 / np.pi

            camera_props.width = 2048
            camera_props.height = 1536
            camera_handle = self.gym.create_camera_sensor(self.env, camera_props)

            position = ext_mat[:3, 3].tolist()
            lookat = (position + ext_mat[:3, 2] ).tolist()
            
            self.gym.set_camera_location(
                camera_handle,
                self.env,
                gymapi.Vec3(*position),
                gymapi.Vec3(*lookat),
            )
            self.camera_handle[serial_num] = camera_handle
            
        return 

    def terminate(self):
        if self.save_video:
            self.out.release()
        if self.save_state:
            pickle.dump(self.history, open(self.state_path, "wb"))

        self.gym.destroy_env(self.env)
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

        print("Simulation terminated")

    def get_dof_names(self):
        if self.view_physics:
            return self.gym.get_asset_dof_names(self.assets["robot"])
        elif self.view_replay:
            return self.gym.get_asset_dof_names(self.assets["vis_robot"])
        else:
            raise ValueError("No robot loaded")
