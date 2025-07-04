import open3d as o3d
import numpy as np

class BodyVisualizer(object):
    #def __init__(self, skeleton_info, head_idx=7, hand_range=(18, 69)):
    def __init__(self, skeleton_info, head_idx=1, hand_range=(2, 53)):
        self.width = 2048
        self.height = 1536
        self.hand_range = hand_range
        self.head_idx = head_idx

        self.main_vis = o3d.visualization.Visualizer()
        self.main_vis.create_window(window_name="body", width=self.width, height=self.height)

        # skeleton_info : List of parent idx, asuume 0 is root(wrist)
        self.num_sphere = len(skeleton_info)
        self.sphere = []
        self.axes = []
        self.prev_axis_inv_transforms = np.zeros((self.num_sphere, 4, 4))
        self.skeleton_info = skeleton_info

        #self.global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        #self.add_geometry({"name":"global_axis", "geometry":self.global_axis})

        # self.cur_rot = np.array([np.eye(3) for i in range(self.num_sphere)])
        
        for i in range(self.num_sphere):
            if hand_range[0] <= i <= hand_range[1]:
                self.sphere.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.01))
                self.sphere[i].paint_uniform_color([0.8, 0.1, 0.1])
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)  # Small axes
            elif i == head_idx:
                self.sphere.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.03))
                self.sphere[i].paint_uniform_color([0.1, 0.8, 0.1])
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # big axis
            else:
                self.sphere.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.03))
                self.sphere[i].paint_uniform_color([0.1, 0.1, 0.8])
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)  # middle axes
            
            self.sphere[i].compute_vertex_normals()
            self.add_geometry({"name":"sphere"+str(i), "geometry":self.sphere[i]})

            self.prev_axis_inv_transforms[i] = np.eye(4)
            
            self.axes.append(axis)
            self.add_geometry({"name":"axis"+str(i), "geometry":self.axes[i]})

        self.bones = []
        self.child_indices = []
        self.parent_indices = []
        # Create cylinders (bones) between connected joints
        for child, parent in enumerate(skeleton_info):
            if parent == -1:
                continue
            self.child_indices.append(child)
            self.parent_indices.append(parent)

            if hand_range[0] <= parent <= hand_range[1]:
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=1)
            else:
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=1)

            cylinder.compute_vertex_normals()
            self.bones.append(cylinder)
            self.add_geometry({"name":f"bone_{parent}_{child}", "geometry":cylinder})

        self.prev_bone_inv_transforms = self.prev_axis_inv_transforms[:len(self.child_indices)].copy()

    def update_bone_positions(self, joint_pos):
        """
        Update cylinder positions (bones) to connect joints.
        """

        end_points = joint_pos[self.child_indices, :3, 3]
        start_points = joint_pos[self.parent_indices, :3, 3]

        u1 = end_points - start_points
        heights = np.linalg.norm(u1, axis=1).reshape(-1, 1)
        u1 /= heights

        u2 = np.zeros((len(self.bones), 3))
        mask = np.all(np.isclose(u1, np.array([0, 0, 1])), axis=1)
        u2[mask, :] = np.array([1, 0, 0])
        u2[~mask, :] = np.array([0, 0, 1])

        u3 = np.cross(u1, u2)
        u3 /= np.linalg.norm(u3, axis=1).reshape(-1, 1)

        u2 = np.cross(u3, u1)
        u2 /= np.linalg.norm(u2, axis=1).reshape(-1, 1)

        S = np.zeros((len(self.bones), 4, 4))
        S[:, 0, 0] = 1;S[:, 1, 1] = 1;S[:, 3, 3] = 1;S[:, 2, 2] = 1
        SE = S.copy()
        S_inv = S.copy()
        SE_inv = S.copy()

        S[:, 2, 2] = heights.reshape(-1)
        SE[:, :3, 3] = (start_points + end_points) / 2
        R = np.stack((u2, u3, u1), axis=-1)
        SE[:, :3, :3] = R

        transform = (SE @ S) @ self.prev_bone_inv_transforms

        S_inv[:, 2, 2] = 1 / heights.reshape(-1)
        R_t = R.swapaxes(1, 2)
        SE_inv[:, :3, :3] = R_t
        SE_inv[:, :3, 3:4] = -R_t @ SE[:, :3, 3:4]

        self.prev_bone_inv_transforms = S_inv @ SE_inv

        for i in range(len(self.bones)):
            self.bones[i].transform(transform[i])
            self.main_vis.update_geometry(self.bones[i])

    def update_axes(self, joint_pos):
        """
        Move and orient the axis frame to match the given transformation matrix exactly.
        """
        R_t = joint_pos[:, :3, :3].swapaxes(1, 2)
        T_inv = np.zeros((self.num_sphere, 4, 4))
        T_inv[:, 3, 3] = 1
        T_inv[:, :3, :3] = R_t
        T_inv[:, :3, 3:4] = -R_t @ joint_pos[:, :3, 3:4]

        transform = joint_pos @ self.prev_axis_inv_transforms

        for i in range(self.num_sphere):
            self.axes[i].transform(transform[i])
            self.main_vis.update_geometry(self.axes[i])
        self.prev_axis_inv_transforms = T_inv

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
        view_control.set_zoom(2)  # Adjust zoom level
        view_control.set_constant_z_far(1000)

    def update_sphere_positions(self, joint_pos):
        """
        Update sphere positions in Open3D.
        joint_pos: (N, 4, 4) NumPy array containing model matrices of joints.
        """
        joint_positions = joint_pos[:,:3,3]
        centers = np.zeros((self.num_sphere, 3))

        for i in range(self.num_sphere):
            centers[i] = np.asarray(self.sphere[i].get_center())

        translations = joint_positions - centers

        for i in range(self.num_sphere):
            self.sphere[i].translate(translations[i], relative=True)
            self.main_vis.update_geometry(self.sphere[i])
        self.update_axes(joint_pos)

        self.update_bone_positions(joint_pos)
        self.center_camera_on_objects()
        self.main_vis.poll_events()
        self.main_vis.update_renderer()  # Request screen refresh
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