import open3d as o3d
import numpy as np

class HandVisualizer(object):
    def __init__(self, skeleton_info):
        self.width = 2048
        self.height = 1536

        self.main_vis = o3d.visualization.Visualizer()
        self.main_vis.create_window(window_name="hand", width=self.width, height=self.height)

        # skeleton_info : List of parent idx, asuume 0 is root(wrist)
        self.num_sphere = len(skeleton_info)
        self.sphere = []
        self.axes = []
        self.cur_rot = np.array([np.eye(3) for i in range(self.num_sphere)])
        
        for i in range(self.num_sphere):
            self.sphere.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.01))
            self.sphere[i].paint_uniform_color([0.8, 0.1, 0.1])
            self.sphere[i].compute_vertex_normals()
        
            self.add_geometry({"name":"sphere"+str(i), "geometry":self.sphere[i]})

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)  # Small axes
            
            self.axes.append(axis)
            self.add_geometry({"name":"axis"+str(i), "geometry":self.axes[i]})
        
        self.bones = []
        # Create cylinders (bones) between connected joints
        for child, parent in enumerate(skeleton_info):
            if parent == -1:
                continue
            
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.01)
            cylinder.compute_vertex_normals()
            self.bones.append(cylinder)
            self.add_geometry({"name":f"bone_{parent}_{child}", "geometry":cylinder})

    def update_bone_positions(self, joint_pos):
        """
        Update cylinder positions (bones) to connect joints.
        """
        for i, (parent, child) in enumerate(self.bone_connections):
            start = joint_pos[parent]
            end = joint_pos[child]
            self.update_cylinder(self.bones[i], start, end)

            
    def update_cylinder(self, cylinder, start, end):
        """
        Move and align a cylinder between two points.
        """
        height = np.linalg.norm(end[:3,3] - start[:3,3])  # Length of the bone
        mid_point = (start[:3,3] + end[:3,3]) / 2  # Midpoint of the bone
        rotation_matrix = start[:3, :3]  # Rotation matrix of the bone

        # cylinder.scale(height, center=cylinder.get_center())  # Adjust size
        # # Transform cylinder
        # cylinder.rotate(rotation_matrix, center=cylinder.get_center())  # Rotate
        cylinder.translate(mid_point - np.asarray(cylinder.get_center()), relative=False)  # Move

    def update_axis(self, i, transform_matrix):
        """
        Move and orient the axis frame to match the given transformation matrix exactly.
        """
        # 1. 현재 회전 제거 (기존 회전 기준)
        self.axes[i].rotate(self.cur_rot[i][:3, :3].T, center=self.axes[i].get_center())

        # 2. 새로운 회전 적용 (축 기준 회전)
        new_rot = transform_matrix[i, :3, :3]
        self.axes[i].rotate(new_rot, center=self.axes[i].get_center())

        # 3. 현재 중심점에서 목표 위치까지의 translation 계산
        current_center = np.asarray(self.axes[i].get_center())
        target_translation = transform_matrix[i, :3, 3] - current_center
        self.axes[i].translate(target_translation, relative=True)

        # 4. 현재 회전 저장
        self.cur_rot[i][:3, :3] = new_rot

    
    def center_camera_on_objects(self):
        """
        Manually set the camera position.
        """
        view_control = self.main_vis.get_view_control()

        # 🔹 Define Camera Parameters
        camera_position = np.array([-1, -1, -1])  # Move behind & slightly above
        look_at = np.array([0, 0, 0])  # Center of the scene
        up_vector = np.array([1, 0, 0])  # Z-axis is "up"

        # 🔹 Apply Camera Settings
        view_control.set_lookat(look_at)
        view_control.set_up(up_vector)
        view_control.set_front(camera_position - look_at)  # Direction from eye to center
        view_control.set_zoom(5)  # Adjust zoom level
        view_control.set_constant_z_far(1000)


    def update_sphere_positions(self, joint_pos):
        """
        Update sphere positions in Open3D.
        joint_positions: (20, 4, 4) NumPy array containing (x, y, z) positions of joints.
        """
        joint_positions = joint_pos[:,:3,3]
        print(joint_positions)
        for i in range(self.num_sphere):
        # Remove old sphere
            # self.main_vis.remove_geometry(f"sphere{i}")

            # Move the sphere to new position
            # self.sphere[i].translate(joint_positions[i] - np.asarray(self.sphere[i].get_center()), relative=False)
            center = np.asarray(self.sphere[i].get_center())
            translation = joint_positions[i] - center
            # print(self.sphere[i].get_center())
            self.sphere[i].translate(translation, relative=True)
            self.sphere[i].compute_vertex_normals()
            # Re-add sphere with updated position
            # self.main_vis.add_geometry(f"sphere{i}", self.sphere[i])
            self.main_vis.update_geometry(self.sphere[i])

            self.update_axis(i, joint_pos)
            self.main_vis.update_geometry(self.axes[i])
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