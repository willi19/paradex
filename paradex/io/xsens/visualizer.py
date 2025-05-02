import open3d as o3d
import numpy as np

class XSensVisualizer(object):
    def __init__(self, title, width, height, test_obj_lst, view=None):
        self.width = width
        self.height = height

        self.main_vis = o3d.visualization.Visualizer()#title, width, height)
        self.main_vis.create_window(window_name=title, width=width, height=height)
        # self.main_vis.setup_camera(intrinsics.intrinsic_matrix, extrinsics, intrinsics.width, intrinsics.height)    

        self.sphere = []
        self.axes = []
        self.cur_rot = np.array([np.eye(3) for i in range(20)])
        for i in range(20):
            self.sphere.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.01))
            self.sphere[i].compute_vertex_normals()
        
            self.add_geometry({"name":"sphere"+str(i), "geometry":self.sphere[i]})

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)  # Small axes
            
            self.axes.append(axis)
            self.add_geometry({"name":"axis"+str(i), "geometry":self.axes[i]})
        # # Define parent-child bone connections (example: fingers, wrist)
        # self.bone_connections = [
        #     (hand_index.hand_index_parent[i], i) for i in range(1, 20)
        # ]
        # self.bones = []
        # self.cur_bone_pos = np.zeros((20, 4, 4))
        # # Create cylinders (bones) between connected joints
        # for parent, child in self.bone_connections:
        #     cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.01)
        #     cylinder.compute_vertex_normals()
        #     self.bones.append(cylinder)
        #     self.main_vis.add_geometry(f"bone_{parent}_{child}", cylinder)

    # def update_bone_positions(self, joint_pos):
    #     """
    #     Update cylinder positions (bones) to connect joints.
    #     """
    #     for i, (parent, child) in enumerate(self.bone_connections):
    #         start = joint_pos[parent]
    #         end = joint_pos[child]
    #         self.main_vis.remove_geometry(f"bone_{parent}_{child}")
    #         self.update_cylinder(self.bones[i], start, end)

    #         self.main_vis.add_geometry(f"bone_{parent}_{child}", self.bones[i])

    # def update_cylinder(self, cylinder, start, end):
    #     """
    #     Move and align a cylinder between two points.
    #     """
    #     height = np.linalg.norm(end[:3,3] - start[:3,3])  # Length of the bone
    #     mid_point = (start[:3,3] + end[:3,3]) / 2  # Midpoint of the bone
    #     rotation_matrix = start[:3, :3]  # Rotation matrix of the bone

    #     # cylinder.scale(height, center=cylinder.get_center())  # Adjust size
    #     # # Transform cylinder
    #     # cylinder.rotate(rotation_matrix, center=cylinder.get_center())  # Rotate
    #     cylinder.translate(mid_point - np.asarray(cylinder.get_center()), relative=False)  # Move

    def update_axis(self, i, transform_matrix):
        """
        Move and orient the axis frame based on the transformation matrix.
        """
        rotation = transform_matrix[i, :3, :3]# @ transform_matrix[hand_index.hand_index_parent[i],:3, :3].T 
        translation = transform_matrix[i, :3, 3]  # Extract position
        

        # Move and rotate the existing axis instead of replacing it
        self.axes[i].rotate(self.cur_rot[i].T, center=np.array([0, 0, 0]))
        self.axes[i].rotate(rotation, center=np.array([0, 0, 0]))  # Rotate around the origin
        self.axes[i].translate(translation, relative=False)

        self.cur_rot[i][:3, :3] = rotation
    
    def center_camera_on_objects(self):
        """
        Manually set the camera position.
        """
        view_control = self.main_vis.get_view_control()

        # 🔹 Define Camera Parameters
        camera_position = np.array([0, 0, -1])  # Move behind & slightly above
        look_at = np.array([0, 0, 0])  # Center of the scene
        up_vector = np.array([1, 0, 0])  # Z-axis is "up"

        # 🔹 Apply Camera Settings
        view_control.set_lookat(look_at)
        view_control.set_up(up_vector)
        view_control.set_front(camera_position - look_at)  # Direction from eye to center
        view_control.set_zoom(10)  # Adjust zoom level


    def update_sphere_positions(self, joint_pos):
        """
        Update sphere positions in Open3D.
        joint_positions: (20, 4, 4) NumPy array containing (x, y, z) positions of joints.
        """
        joint_positions = joint_pos[:,:3,3]
        for i in range(20):
        # Remove old sphere
            # self.main_vis.remove_geometry(f"sphere{i}")

            # Move the sphere to new position
            # self.sphere[i].translate(joint_positions[i] - np.asarray(self.sphere[i].get_center()), relative=False)
            sphere_vertices = np.asarray(self.sphere[i].vertices)
            sphere_vertices += joint_positions[i] - np.asarray(self.sphere[i].get_center())
            self.sphere[i].vertices = o3d.utility.Vector3dVector(sphere_vertices)
            # Re-add sphere with updated position
            # self.main_vis.add_geometry(f"sphere{i}", self.sphere[i])
            self.main_vis.update_geometry(self.sphere[i])

            self.update_axis(i, joint_pos)
            self.main_vis.update_geometry(self.axes[i])
        self.center_camera_on_objects()
        self.main_vis.poll_events()
        self.main_vis.update_renderer()  # Request screen refresh
        return
        

    
    def export_view(self):
        return self.curview

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

    def write_image(self, imagepath):
        self.main_vis.export_current_image(imagepath)

    """
    cv2 array를 위한 변환, 업데이트가 안됨
    """
    # def get_image(self):
    #     app = o3d.visualization.gui.Application.instance
    #     image = app.render_to_image(self.main_vis.scene, 0, 0) # o3d object
    #     return image

    def set_background(self, image):
        self.main_vis.set_background([1, 1, 1, 0], image)

    def remove_geometry(self, geom_name):
        self.main_vis.remove_geometry(geom_name)
    
    def run(self):
        self.main_vis.run()

    def stop(self):
        self.main_vis.destroy_window()
        self.main_vis.close()
        return