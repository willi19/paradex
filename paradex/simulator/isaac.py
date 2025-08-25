import os
import numpy as np
from isaacgym import gymapi
import pickle
from scipy.spatial.transform import Rotation as R
import cv2
from paradex.utils.file_io import rsc_path


class Simulator:
    def __init__(
        self,
        headless,
        plane=True
    ):
        self.gym = gymapi.acquire_gym()
        self.sim = self.generate_sim()

        if plane:
          self.add_plane()
        
        self.headless = headless
        if not headless:
            self.set_viewer()
        
        self.asset_root = rsc_path
        self.assets = {"robot":{}, "robot_vis":{}, "object":{}, "object_vis":{}}
        
        self.env_dict = {}
        self.actor_handle_dict = {}
        
        spacing = 1.5
        self.env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.num_envs = 0
        
        self.save_state = False
        self.save_video = False
        
        self.camera_handle = {}
        self.out = {}
        
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

    def set_viewer(self, ext_mat = np.array([[1, 0, 0, 2],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 1],
                                             [0, 0, 0, 1]])):
        position = ext_mat[:3, 3].tolist()
        lookat = [0,0,0]#(position + ext_mat[:3, 2] ).tolist()
        
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
               
        self.gym.viewer_camera_look_at(
            self.viewer,
            None,
            gymapi.Vec3(*position),
            gymapi.Vec3(*lookat),
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

    def load_robot_name(self, arm_name, hand_name):
        robot_name = None
        if arm_name == None:
            robot_name = hand_name
        elif hand_name == None:
            robot_name = arm_name            
        else:
            robot_name = f"{arm_name}_{hand_name}"
        
        return robot_name
    
    def load_robot_asset(self, arm_name, hand_name):
        robot_name = self.load_robot_name(arm_name, hand_name)    
        if arm_name is not None:
            robot_asset_file = f"robot/{robot_name}.urdf"
        else:
            robot_asset_file = f"robot/{hand_name}_float.urdf"
            
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = True
        robot_asset_options.armature = 0.001
        robot_asset_options.thickness = 0.002

        vis_robot_asset_options = gymapi.AssetOptions()
        vis_robot_asset_options.fix_base_link = True
        
        robot_asset = self.gym.load_asset(
            self.sim, self.asset_root, robot_asset_file, robot_asset_options
        )
        
        vis_robot_asset = self.gym.load_asset(
            self.sim, self.asset_root, robot_asset_file, vis_robot_asset_options
        )
        
        self.assets["robot"][robot_name] = robot_asset
        self.assets["robot_vis"][robot_name] = vis_robot_asset
            
    def load_object_asset(self, obj_name):
        object_asset_file = f"object/{obj_name}/{obj_name}.urdf"

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.override_inertia = True
        object_asset_options.mesh_normal_mode = (
            gymapi.COMPUTE_PER_VERTEX
        )  # Use per-vertex normals
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 300000
        
        vis_object_asset_options = gymapi.AssetOptions()
        vis_object_asset_options.disable_gravity = True
        
        object_asset = self.gym.load_asset(
            self.sim, self.asset_root, object_asset_file, object_asset_options
        )
        
        vis_object_asset = self.gym.load_asset(
            self.sim, self.asset_root, object_asset_file, vis_object_asset_options
        )
        
        self.assets["object"][obj_name] = object_asset
        self.assets["object_vis"][obj_name] = vis_object_asset
        
    def visualize_camera(self, cam_param_dict): # deprecated
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

        return
    
    def load_camera(self, name, camera_param_dict=None):
        self.camera_handle[name] = {}
        if camera_param_dict is None: # Single default camera
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75.0
            camera_props.width = 2048
            camera_props.height = 1536

            env = self.env_dict[name]
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            self.gym.set_camera_location(
                camera_handle,
                env,
                gymapi.Vec3(1.5, 0, 1.5),
                gymapi.Vec3(0, 0, 0.3),
            )
            self.camera_handle[name][f"default"] = camera_handle
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
    
    def set_videopath(self, env_name, video_path):
        self.save_video = True
        self.out[env_name] = {}
        
        camera_handle_dict = self.camera_handle[env_name]
        os.makedirs(os.path.join(video_path, str(env_name)), exist_ok=True)
        
        for name, camera_handle in camera_handle_dict.items():    
            output_filename = os.path.join(video_path, str(env_name), f"{name}.mp4")
            
            self.frame_width = 2048
            self.frame_height = 1536
            self.fps = 30
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out[env_name][name] = cv2.VideoWriter(
                output_filename,
                self.fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )

    # def set_savepath(self, state_path):
    #     self.save_state = True
    #     self.state_path = state_path
    #     os.makedirs(os.path.dirname(self.state_path), exist_ok=True)

    #     self.history = {"robot": [], "object": []}

    def save_stateinfo(self, env_idx):
        if env_idx not in self.history:
            self.history[env_idx] = {"robot":{}, "object":{}}
        
        env = self.env_dict[env_idx]
        actor_handle = self.actor_handle_dict[env_idx]
        
        for robot_name, actor in actor_handle["robot"].items():
            if robot_name not in self.history[env_idx]["robot"]:
                self.history[env_idx]["robot"][robot_name] = []
            
            robot_dof_state = self.gym.get_actor_dof_states(
                env, actor_handle["robot"], gymapi.STATE_POS
            )
            
            self.history[env_idx]["robot"][robot_name].append(robot_dof_state["pos"])
                
        for obj_name, actor in actor_handle["object"].items():            
            object_rb_state = self.gym.get_actor_rigid_body_states(
                env, actor, gymapi.STATE_POS
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

            self.history[env_idx]["object"][obj_name].append(obj_T)

    def load_robot_actor(self, env, actor_name, arm_name, hand_name):
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0, 0, 0)
        
        robot_name = self.load_robot_name(arm_name, hand_name)
        
        actor = self.gym.create_actor(
            env, self.assets["robot"][robot_name], robot_pose, actor_name, self.num_envs, 0
        )
        
        props = self.gym.get_actor_dof_properties(env, actor)
        
        if arm_name == "xarm":
            props["driveMode"].fill(gymapi.DOF_MODE_POS)

            props["stiffness"][:6] = 1000.0  # pgain for arm
            props["damping"][:6] = 10.0  # dgain for arm

            props["stiffness"][6:] = 500.0  # pgain for hand
            props["damping"][6:] = 10.0  # dgain for hand
        
        self.gym.set_actor_dof_properties(env, actor, props)

        # rigid_prop = self.gym.get_actor_rigid_shape_properties(
        #         env, actor
        #     )

        # rigid_prop[0].restitution = 0.01
        # rigid_prop[0].friction = 0.8    
        return actor

    def load_vis_robot_actor(self, env, actor_name, arm_name, hand_name):
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0, 0, 0)
        
        robot_name = self.load_robot_name(arm_name, hand_name)
        actor = self.gym.create_actor(
                    env, self.assets["robot_vis"][robot_name], robot_pose, actor_name, 1000, 1
                )
        
        rigid_body_props = self.gym.get_actor_rigid_body_properties(
            env, actor
        )

        for prop in rigid_body_props:
            prop.flags = (
                gymapi.RIGID_BODY_DISABLE_GRAVITY
            )  # Disable gravity flag

        self.gym.set_actor_rigid_body_properties(
            env,
            actor,
            rigid_body_props,
            recomputeInertia=False,
        )
        
        props = self.gym.get_actor_dof_properties(env, actor)
        props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        self.gym.set_actor_dof_properties(env, actor, props)
                    
        # Set color to distinguish
        num_rigid_bodies = self.gym.get_actor_rigid_body_count(env, actor)
        for i in range(num_rigid_bodies):
            self.gym.set_rigid_body_color(
                env,
                actor,
                i,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.4, 0.4, 0.6),
            )
            
        return actor
    
    def load_object_actor(self, env, actor_name, obj_name):
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(0.5, 0.0, 0.0) # Temporary position for initialization
        actor = self.gym.create_actor(
                env, self.assets["object"][obj_name], object_pose, actor_name, self.num_envs, 0
            )

        obj_props = self.gym.get_actor_rigid_shape_properties(
            env, actor
        )

        obj_props[0].restitution = 0.01
        obj_props[0].friction = 0.8

        self.gym.set_actor_rigid_shape_properties(
            env, actor, obj_props
        )
        
        return actor
    
    def load_vis_object_actor(self, env, actor_name, obj_name):
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(0.5, 0.0, 0.0) # Temporary position for initialization

        actor = self.gym.create_actor(
                        env, self.assets["object_vis"][obj_name], object_pose, actor_name, 1001, 0
                    )
        
        rigid_body_props = self.gym.get_actor_rigid_body_properties(
            env, actor
        )

        for prop in rigid_body_props:
            prop.flags = (
                gymapi.RIGID_BODY_DISABLE_GRAVITY
            )  # Disable gravity flag

        self.gym.set_actor_rigid_body_properties(
            env,
            actor,
            rigid_body_props,
            recomputeInertia=False,
        )

        props = self.gym.get_actor_dof_properties(env, actor)
        props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        self.gym.set_actor_dof_properties(env, actor, props)
               
        self.gym.set_rigid_body_color(
            env,
            actor,
            0,
            gymapi.MESH_VISUAL_AND_COLLISION,
            gymapi.Vec3(0.4, 0.4, 0.6),
        )
        
        return actor
    
    def add_env(self, name, env_info):
        self.num_envs += 1
        
        actor_handle = {"robot":{}, "robot_vis":{}, "object":{}, "object_vis":{}}
        env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, 5)
            
        actor_handle["robot"] = {actor_name : self.load_robot_actor(env, actor_name, arm_name, hand_name) for actor_name, (arm_name, hand_name) in env_info["robot"].items()} 
        actor_handle["robot_vis"] = {actor_name : self.load_vis_robot_actor(env, actor_name, arm_name, hand_name) for actor_name, (arm_name, hand_name) in env_info["robot_vis"].items()}
        actor_handle["object"] = {actor_name : self.load_object_actor(env, actor_name, obj_name) for actor_name, obj_name in env_info["object"].items()}
        actor_handle["object_vis"] = {actor_name : self.load_vis_object_actor(env, actor_name, obj_name) for actor_name, obj_name in env_info["object_vis"].items()}
        
        
        self.env_dict[name] = env
        self.actor_handle_dict[name] = actor_handle

    def step(self, name, action_dict):
        env = self.env_dict[name]
        actor_handle = self.actor_handle_dict[name]
        
        if self.save_state:
            self.save_stateinfo(name)
        
        for robot_name, action in action_dict["robot"].items():
            actor = actor_handle["robot"][robot_name]
            self.gym.set_actor_dof_position_targets(
                env, actor, action
            )
        
        for robot_name, state in action_dict["robot_vis"].items():
            actor = actor_handle["robot_vis"][robot_name]
            # print(action)
            robot_dof_state = self.gym.get_actor_dof_states(
                env, actor, gymapi.STATE_POS
            )
            robot_dof_state["pos"] = state
            
            self.gym.set_actor_dof_states(
                env,
                actor,
                robot_dof_state,
                gymapi.STATE_POS,
            )
            
        for obj_name, obj_T in action_dict["object_vis"].items():
            actor = actor_handle["object_vis"][obj_name]
            
            obj_quat = R.from_matrix(obj_T[:3, :3]).as_quat()
            obj_pos = obj_T[:3, 3]
            
            object_rb_state = self.gym.get_actor_rigid_body_states(
                env, actor, gymapi.STATE_POS
            )
            object_rb_state["pose"]["r"].fill(
                (obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3])
            )
            object_rb_state["pose"]["p"].fill((obj_pos[0], obj_pos[1], obj_pos[2]))

            self.gym.set_actor_rigid_body_states(
                env,
                actor,
                object_rb_state,
                gymapi.STATE_POS,
            )


    def tick(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)
        
        if not self.headless:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        if self.save_video:
            self.gym.render_all_camera_sensors(self.sim)
            for env_name, camera_handle_dict in self.camera_handle.items():
                for name, camera_handle in camera_handle_dict.items():
                    frame = self.gym.get_camera_image(
                        self.sim, self.env_dict[env_name], camera_handle, gymapi.IMAGE_COLOR
                    ).astype(np.uint8)

                    frame = frame.reshape((self.frame_height, self.frame_width, 4))[:, :, :3]
                    frame = frame[:, :, ::-1]
                    self.out[env_name][name].write(frame)
        
    def reset(self, name, action_dict):
        env = self.env_dict[name]
        actor_handle = self.actor_handle_dict[name]
        
        for robot_name, action in action_dict["robot"].items():
            actor = actor_handle["robot"][robot_name]
            robot_dof_state = self.gym.get_actor_dof_states(
                env, actor, gymapi.STATE_POS
            )

            robot_dof_state["pos"] = action
            self.gym.set_actor_dof_states(
                env,
                actor,
                robot_dof_state,
                gymapi.STATE_POS,
            )
        
        for robot_name, action in action_dict["robot_vis"].items():
            actor = actor_handle["robot_vis"][robot_name]
            robot_dof_state = self.gym.get_actor_dof_states(
                env, actor, gymapi.STATE_POS
            )

            robot_dof_state["pos"] = action
            self.gym.set_actor_dof_states(
                env,
                actor,
                robot_dof_state,
                gymapi.STATE_POS,
            )
            
        for obj_name, obj_T in action_dict["object"].items():
            obj_quat = R.from_matrix(obj_T[:3, :3]).as_quat()
            obj_pos = obj_T[:3, 3]
            
            actor = actor_handle["object"][obj_name]
            object_rb_state = self.gym.get_actor_rigid_body_states(
                        env, actor, gymapi.STATE_POS
            )
            object_rb_state["pose"]["r"].fill(
                (obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3])
            )
            object_rb_state["pose"]["p"].fill((obj_pos[0], obj_pos[1], obj_pos[2]))

            self.gym.set_actor_rigid_body_states(
                env,
                actor,
                object_rb_state,
                gymapi.STATE_POS,
            )
            
    def save(self):
        if self.save_video:
            for env_id, out_dict in self.out.items():
                for name, out in out_dict.items():
                    out.release()
            self.save_video = False
            self.out = {}
        
        if self.save_state:
            pickle.dump(self.history, open(self.state_path, "wb"))

            self.save_state = False
            self.history = {}
            self.state_path = None
            
    def terminate(self):
        self.save()
        
        for env in self.env_dict.values():
            self.gym.destroy_env(env)
            
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        
        self.gym.destroy_sim(self.sim)
        print("Simulation terminated")

    def destroy_env(self, name):
        for out in self.out[name].values():
            out.release()
        del self.out[name]
        
        self.gym.destroy_env(self.env_dict[name])
        del self.env_dict[name]
        del self.camera_handle[name]
        