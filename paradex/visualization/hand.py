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
        self.sphere = []
        self.axes = []
        self.prev_axis_inv_transforms = []
        self.skeleton_info = skeleton_info

        self.global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.add_geometry({"name":"global_axis", "geometry":self.global_axis})

        # self.cur_rot = np.array([np.eye(3) for i in range(self.num_sphere)])
        
        for i in range(self.num_sphere):
            self.sphere.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.01))
            self.sphere[i].paint_uniform_color([0.8, 0.1, 0.1])
            self.sphere[i].compute_vertex_normals()
        
            self.add_geometry({"name":"sphere"+str(i), "geometry":self.sphere[i]})

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)  # Small axes
            self.prev_axis_inv_transforms.append(np.eye(4))
            
            self.axes.append(axis)
            self.add_geometry({"name":"axis"+str(i), "geometry":self.axes[i]})
        
        self.bones = {}
        self.prev_bone_inv_transforms = {}
        
        # Create cylinders (bones) between connected joints
        for child, parent in enumerate(skeleton_info):
            if parent == -1:
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

    def update_bone_positions(self, joint_pos):
        """
        Update cylinder positions (bones) to connect joints.
        """
        for child, parent in enumerate(self.skeleton_info):
            if parent == -1:
                continue
            self.update_cylinder(child, joint_pos[0], 0) # Left
            self.update_cylinder(child, joint_pos[1], 1) # Right

            
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

    def update_axis(self, i, transform_matrix):
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

        self.axes[i].transform(transform_matrix @ self.prev_axis_inv_transforms[i])
        self.main_vis.update_geometry(self.axes[i])
        self.prev_axis_inv_transforms[i] = T_inv

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


    def update_sphere_positions(self, joint_pos):
        """
        Update sphere positions in Open3D.
        joint_positions: (20, 4, 4) NumPy array containing (x, y, z) positions of joints.
        """
        joint_positions = joint_pos[:,:,:3,3].reshape(-1, 3)
        #print(joint_positions)
        for i in range(self.num_sphere):
        # Remove old sphere
            # self.main_vis.remove_geometry(f"sphere{i}")

            # Move the sphere to new position
            # self.sphere[i].translate(joint_positions[i] - np.asarray(self.sphere[i].get_center()), relative=False)
            center = np.asarray(self.sphere[i].get_center())
            translation = joint_positions[i] - center
            # print(self.sphere[i].get_center())
            self.sphere[i].translate(translation, relative=True)
            #self.sphere[i].compute_vertex_normals()
            # Re-add sphere with updated position
            # self.main_vis.add_geometry(f"sphere{i}", self.sphere[i])
            self.main_vis.update_geometry(self.sphere[i])
            self.update_axis(i, joint_pos[i // len(self.skeleton_info), i % len(self.skeleton_info)])

        self.update_bone_positions(joint_pos)
        self.center_camera_on_objects()
        self.main_vis.poll_events()
        self.main_vis.update_renderer()  # Request screen refresh
        return

    def add_geometry(self, geometry:dict):
        self.main_vis.add_geometry(geometry["geometry"])

    def stop(self):
        self.main_vis.destroy_window()
        self.main_vis.close()
        return