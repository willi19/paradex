import open3d as o3d
import numpy as np

class HandVisualizer(object):
    def __init__(self, skeleton_info):
        self.width = 2048
        self.height = 1536

        self.main_vis = o3d.visualization.Visualizer()
        self.main_vis.create_window(window_name="hand", width=self.width, height=self.height)

        # skeleton_info : List of parent idx, asuume 0 is root(wrist)
        self.num_sphere = len(skeleton_info)*2
        self.sphere = {}
        
        self.axes = {}
        self.prev_axis_inv_transforms = {}
        
        self.skeleton_info = skeleton_info

        self.global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.add_geometry({"name":"global_axis", "geometry":self.global_axis})

        # self.cur_rot = np.array([np.eye(3) for i in range(self.num_sphere)])
        for side in ['Left', 'Right']:
            for child, parent in skeleton_info.items():
                self.sphere[(side, child)] = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                self.sphere[(side, child)].paint_uniform_color([0.8, 0.1, 0.1])
                self.sphere[(side, child)].compute_vertex_normals()
            
                self.add_geometry({"name":"sphere"+ side + str(child), "geometry":self.sphere[(side, child)]})

                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)  # Small axes
                self.prev_axis_inv_transforms[(side, child)] = np.eye(4)
                
                self.axes[(side, child)] = axis
                self.add_geometry({"name":"axis"+ side + str(child), "geometry":self.axes[(side, child)]})
        
        self.bones = {}
        self.prev_bone_inv_transforms = {}
        
        # Create cylinders (bones) between connected joints
        for child, parent in skeleton_info.items():
            if parent == None:
                continue
            cylinder_l = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=1)
            cylinder_l.compute_vertex_normals()
            self.bones[(0, child)] = cylinder_l
            self.prev_bone_inv_transforms[(0, child)] = np.eye(4)
            cylinder_r = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=1)
            cylinder_r.compute_vertex_normals()
            self.bones[(1, child)] = cylinder_r
            self.prev_bone_inv_transforms[(1, child)] = np.eye(4)
            self.add_geometry({"name":f"bone_l_{parent}_{child}", "geometry":cylinder_l})
            self.add_geometry({"name":f"bone_r_{parent}_{child}", "geometry":cylinder_r})

    def update_bone_positions(self, hand_data):
        """
        Update cylinder positions (bones) to connect joints.
        """
        for child, parent in self.skeleton_info.items():
            if parent == None:
                continue
            self.update_cylinder(child, hand_data['Left'], 0) # Left
            self.update_cylinder(child, hand_data['Right'], 1) # Right

            
    def update_cylinder(self, child, joint_pos, hand_idx):
        """
        Move and align a cylinder between two points.
        """
        end = joint_pos[child]
        start = joint_pos[self.skeleton_info[child]]
        cylinder = self.bones[(hand_idx, child)]
        T_prev_inv = self.prev_bone_inv_transforms[(hand_idx, child)]

        orientation = end[:3, 3] - start[:3, 3]
        height = np.linalg.norm(orientation)  # Length of the bone
        orientation /= height

        #rotation_matrix = start[:3, :3]  # Rotation matrix of the bone
        
        orientation2 = np.array([0, 0, 1]) if np.allclose(orientation, np.array([0, 0, 1])) else np.array([1, 0, 0])
        orientation3 = np.cross(orientation, orientation2)
        orientation3 /= np.linalg.norm(orientation3)
        orientation2 = np.cross(orientation3, orientation)
        orientation2 /= np.linalg.norm(orientation2)

        S = np.eye(4)
        S[2, 2] = height

        SE = np.eye(4)
        SE[:3, 3] = (start[:3,3] + end[:3,3]) / 2  # Midpoint of the bone
        R = np.stack((orientation2, orientation3, orientation), axis=1)
        SE[:3, :3] = R

        transform_matrix = SE @ S @ T_prev_inv

        S_inv = np.eye(4)
        S_inv[2, 2] = 1 / height
        SE_inv = np.eye(4)
        SE_inv[:3, :3] = R.T
        SE_inv[:3, 3] = -R.T @ SE[:3, 3]
        self.prev_bone_inv_transforms[(hand_idx, child)] = S_inv @ SE_inv

        cylinder.transform(transform_matrix)
        self.main_vis.update_geometry(cylinder)

        # cylinder.scale(height, center=cylinder.get_center())  # Adjust size
        # # Transform cylinder
        # cylinder.rotate(rotation_matrix, center=cylinder.get_center())  # Rotate
        # cylinder.translate(mid_point - np.asarray(cylinder.get_center()), relative=False)  # Move

    def update_axis(self, joint_name, transform_matrix):
        """
        Move and orient the axis frame to match the given transformation matrix exactly.
        """
        # 1. 현재 회전 제거 (기존 회전 기준)
        # self.axes[i].rotate(self.cur_rot[i][:3, :3].T, center=self.axes[i].get_center())

        # 2. 새로운 회전 적용 (축 기준 회전)
        # new_rot = transform_matrix[i, :3, :3]
        # self.axes[i].rotate(new_rot, center=self.axes[i].get_center())

        # 3. 현재 중심점에서 목표 위치까지의 translation 계산
        # current_center = np.asarray(self.axes[i].get_center())
        # target_translation = transform_matrix[i, :3, 3] - current_center
        # self.axes[i].translate(target_translation, relative=True)

        # 4. 현재 회전 저장
        # self.cur_rot[i][:3, :3] = new_rot
        R_t = (transform_matrix[:3, :3]).T
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_t
        T_inv[:3, 3] = -R_t @ (transform_matrix[:3, 3])

        self.axes[joint_name].transform(transform_matrix @ self.prev_axis_inv_transforms[joint_name])
        self.main_vis.update_geometry(self.axes[joint_name])
        self.prev_axis_inv_transforms[joint_name] = T_inv

    def center_camera_on_objects(self):
        """
        Manually set the camera position.
        """
        view_control = self.main_vis.get_view_control()

        # 🔹 Define Camera Parameters
        camera_position = np.array([-1, -1, 1])  # Move behind & slightly above
        look_at = np.array([0, 0, 0])  # Center of the scene
        up_vector = np.array([0, 0, 1])  # Z-axis is "up"

        # 🔹 Apply Camera Settings
        view_control.set_lookat(look_at)
        view_control.set_up(up_vector)
        view_control.set_front(camera_position - look_at)  # Direction from eye to center
        view_control.set_zoom(0.9)  # Adjust zoom level
        view_control.set_constant_z_far(1000)


    def update_sphere_positions(self, hand_data):
        """
        Update sphere positions in Open3D.
        joint_positions: (J, 4, 4) NumPy array containing (x, y, z) positions of joints.
        """
        for side, joint_dict in hand_data.items():
            if side not in ['Left', 'Right']:
                continue
            for joint_name, joint_T in joint_dict.items():
                center = np.asarray(self.sphere[(side, joint_name)].get_center())
                translation = joint_T[:3, 3] - center
                self.sphere[(side,joint_name)].translate(translation, relative=True)
                
                self.main_vis.update_geometry(self.sphere[(side, joint_name)])
                self.update_axis((side,joint_name), joint_T)

        # self.update_bone_positions(hand_data)
        # self.center_camera_on_objects()
        self.main_vis.poll_events()
        self.main_vis.update_renderer()  # Request screen refresh
        return

    def tick(self):
        app = o3d.visualization.gui.Application.instance
        tick_return = app.run_one_tick()
        if tick_return:
            self.main_vis.post_redraw()
        return tick_return

    def add_plane(self, resolution=128, bound=100, up_vec='z'):
        def makeGridPlane(bound=100., resolution=128, color = np.array([0.5,0.5,0.5]), up='z'):
            min_bound = np.array([-bound, -bound])
            max_bound = np.array([bound, bound])
            xy_range = np.linspace(min_bound, max_bound, num=resolution)
            grid_points = np.stack(np.meshgrid(*xy_range.T), axis=-1).astype(np.float32) # asd
            if up == 'z':
                grid3d = np.concatenate([grid_points, np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1)], axis=2)
            elif up == 'y':
                grid3d = np.concatenate([grid_points[:,:,0][:,:,None], np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1), grid_points[:,:,1][:,:,None]], axis=2)
            elif up == 'x':
                grid3d = np.concatenate([np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1), grid_points], axis=2)
            else:
                print("Up vector not specified")
                return None
            grid3d = grid3d.reshape((resolution**2,3))
            indices = []
            for y in range(resolution):
                for x in range(resolution):  
                    corner_idx = resolution*y + x 
                    if x + 1 < resolution:
                        indices.append((corner_idx, corner_idx + 1))
                    if y + 1 < resolution:
                        indices.append((corner_idx, corner_idx + resolution))

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(grid3d),
                lines=o3d.utility.Vector2iVector(indices),
            )
            # line_set.colors = o3d.utility.Vector3dVector(colors)  
            line_set.paint_uniform_color(color)
            
            return line_set
        plane = makeGridPlane(bound, resolution, up=up_vec)
        self.main_vis.add_geometry({"name":"floor", "geometry":plane})
        return

    def remove_plane(self):
        self.main_vis.remove_geometry({"name":"floor"})
        return

    def add_geometry(self, geometry:dict):
        self.main_vis.add_geometry(geometry["geometry"])
        
    def remove_geometry(self, geom_name):
        self.main_vis.remove_geometry(geom_name)
    
    def run(self):
        self.main_vis.run()

    def stop(self):
        self.main_vis.destroy_window()
        self.main_vis.close()
        return