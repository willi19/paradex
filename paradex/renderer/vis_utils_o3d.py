import open3d as o3d
import numpy as np 

boc = {'y': [1, 1, 0], 'g': [0, 1, 0], 'r': [0, 0, 1], 'b': [1, 0, 0], 'black': [0, 0, 0],
       'basic': [100/255, 100/255, 100/255], 'line': [205/255, 92/255, 92/255]}


def get_pointcloud(pts_arr, color_arr=None):
    # pts_arr = NX3, color_arr NX3 [0~1]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts_arr)
    if color_arr is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(color_arr[:,:3])

    return point_cloud


def get_camLinesSet(cam_extr_mtrx, length=1):

    assert cam_extr_mtrx is not None, 'cam_extr_matrx should not be None'

    R_list = []
    t_list = []

    if type(cam_extr_mtrx) == dict:
        for cam_id in cam_extr_mtrx:
            mtrx = cam_extr_mtrx[cam_id]
            R_list.append(np.array(mtrx[:, :3]).astype(np.float64))
            t_list.append(np.array(mtrx[:, 3]).astype(np.float64))
    else: # for list
        for mtrx in cam_extr_mtrx:
            R_list.append(np.array(mtrx[:, :3]).astype(np.float64))
            t_list.append(np.array(mtrx[:, 3]).astype(np.float64))

    ip_1 = np.array([length, length, length])
    ip_2 = np.array([-length, length, length])
    ip_3 = np.array([-length, -length, length])
    ip_4 = np.array([length, -length, length])
    iview_dir = np.array([0, 0, length * 2])

    L = len(R_list)

    cam_draw_list = []

    for idx in range(L):
        R_w2c = R_list[idx]
        t_w2c = t_list[idx]

        R_c2w = np.linalg.inv(R_w2c)
        t_c2w = - \
            (R_c2w@t_w2c)

        cam_ctr = t_c2w
        p_1 = ((R_c2w@ip_1) + t_c2w)
        p_2 = ((R_c2w@ip_2) + t_c2w)
        p_3 = ((R_c2w@ip_3) + t_c2w)
        p_4 = ((R_c2w@ip_4) + t_c2w)
        view_dir = ((R_c2w@iview_dir) + t_c2w)

        cam_ctr = ((R_c2w@np.array(
            [0, 0, 0]))+t_c2w)
        cam_x = ((R_c2w@np.array(
            [length, 0, 0]))+t_c2w)
        cam_y = ((R_c2w@np.array(
            [0, length, 0]))+t_c2w)
        cam_z = ((R_c2w@np.array(
            [0, 0, length]))+t_c2w)

        # write index of camera to drawn camera: tmp_ax.text(cam_ctr[0], cam_ctr[1], cam_ctr[2], f"{idx}", size=10, color='b')
        #               0       1       2       3       4      5    6    7     8
        cam_points = [cam_ctr, cam_x, cam_y,
                      cam_x, view_dir, p_1, p_2, p_3, p_4]

        cam_lines = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [0,8], [5,6], [5,8], [6,7], [7,8]]
        cam_colors = [boc['r'],boc['g'],boc['b'],boc['y'],boc['basic'],boc['basic'],boc['basic'],boc['basic'],boc['basic'],boc['basic'],boc['basic'],boc['basic']]

        cam_lines_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(cam_points),
            lines=o3d.utility.Vector2iVector(cam_lines),
        )
        cam_lines_set.colors = o3d.utility.Vector3dVector(cam_colors)

        cam_draw_list.append(cam_lines_set)

    return cam_draw_list



class simpleViewer(object):
    def __init__(self, width=None, height=None, intrinsics=None):
        # self.cam_param.extrinsic = np.linalg.inv(T_gk)
        print("initialize viewer")
        self.main_vis = o3d.visualization.Visualizer()
        if width is not None and height is not None:
            self.main_vis.create_window(visible=True, width=width, height=height)
        else:
            self.main_vis.create_window(visible=True)
        # self.opt = self.main_vis.get_render_option()
        # self.opt.background_color = np.asarray([0,0,0])
        # self.opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        self.plane = None
        self.reset_flag = True
        # self.add_plane(mesh = True)
    
        self.ctr = self.main_vis.get_view_control()

        # self.ctr.set_constant_z_near(0.001)
        # self.ctr.set_constant_z_far(5) # 5m?
        # print(f"original given width {width}, height {height}")

        cam_param = self.ctr.convert_to_pinhole_camera_parameters() # current intrinsic
        self.width = cam_param.intrinsic.width
        self.height = cam_param.intrinsic.height
        # print(f"real window width {self.width} window height {self.height}")

        if intrinsics is not None:
            # Set Intrinsic
            self.intrinsics = intrinsics
            self.intrinsics.width = self.width
            self.intrinsics.height = self.height
            intrinsic_matrix = np.array(self.intrinsics.intrinsic_matrix)
            intrinsic_matrix[0,2] = self.width/2.0-0.5
            intrinsic_matrix[1,2] = self.height/2.0-0.5
            self.intrinsics.intrinsic_matrix = intrinsic_matrix
            self.cam_param = cam_param
            self.cam_param.intrinsic = self.intrinsics
        else:
            self.cam_param = cam_param

        self.mesh_dict = {}


    def setupcamera(self, extrinsic_matrix):

        self.cam_param.extrinsic = extrinsic_matrix
        self.ctr.convert_from_pinhole_camera_parameters(self.cam_param, True)
        # self.main_vis.setup_camera(self.intrinsics, extrinsic_matrix)

    def tick(self):
        # self.ctr.convert_from_pinhole_camera_parameters(self.cam_param, allow_arbitrary = True)
        # self.ctr.set_constant_z_near(0.001)
        # self.ctr.set_constant_z_far(10)

        self.main_vis.poll_events()
        self.main_vis.update_renderer()


    def add_plane(self, mesh = True, bound=200):
        if self.plane is None:
            plane = o3d.geometry.TriangleMesh.create_box(width=bound, height=bound, depth=0.001)
            plane.translate((-bound/2,-bound/2,-0.001))

            self.main_vis.add_geometry(plane, reset_bounding_box=self.reset_flag)
        return


    def remove_plane(self):
        # self.main_vis.remove_geometry({"name":"floor"})
        return
    

    def add_geometries_list(self, geometry_list:list):

        mesh_name_list = []
        for geometry in geometry_list:
            name = f'object_{len(self.mesh_dict)}'
            mesh_name_list.append(name)
            self.mesh_dict[name] = geometry
            self.main_vis.add_geometry(self.mesh_dict[name], reset_bounding_box=self.reset_flag)

        return mesh_name_list

    def add_geometries_dict(self, geometry_dict:dict):
        name, geometry = geometry_dict["name"], geometry_dict["geometry"]
        self.mesh_dict[name] = geometry
        self.main_vis.add_geometry(self.mesh_dict[name], reset_bounding_box=self.reset_flag)
        
        # if self.reset_flag:
        #     self.reset_flag = False

    def remove_geometry(self, name):
        if name in self.mesh_dict:
            self.main_vis.remove_geometry(self.mesh_dict[name], reset_bounding_box=self.reset_flag)
            # if self.reset_flag:
            #     self.reset_flag = False

            self.mesh_dict.pop(name)

    def remove_geometries(self): # remove all gemometries
        key_list = list(self.mesh_dict.keys())
        for nm in key_list:
            if nm in self.mesh_dict:
                self.main_vis.remove_geometry(self.mesh_dict[nm], reset_bounding_box=self.reset_flag)
                self.mesh_dict.pop(nm)
        
        # if self.reset_flag:
        #     self.reset_flag = False

    def render_image(self):
        # Save to Image
        # Render Depth
        # depth.astype(np.uint16)
        # depth_rn = np.asarray(self.main_vis.capture_depth_float_buffer(True))
        # depth_rn = depth_rn.astype(np.float32)
        # Render RGB

        rgb_rn = np.asarray(self.main_vis.capture_screen_float_buffer(True))
        rgb_rn = rgb_rn.astype(np.float32)
        return rgb_rn

    # def get_image(self, imagepath):
    #     img = cv2.imread(imagepath)
    #     return img
    
    def transform(self, name, transform_mtx):
        if name in self.mesh_dict:
            self.mesh_dict[name].transform(transform_mtx)
        else:
            print(f"{name} doesn't not exist in the scene")

    def set_background(self, image):
        self.main_vis.set_background([1, 1, 1, 0], image)
    
    def run(self):
        self.main_vis.run()
        self.reset_flag = False
        
        