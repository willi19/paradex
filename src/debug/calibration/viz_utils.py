import open3d as o3d
import numpy as np 


class simpleViewer(object):
    def __init__(self, title, width, height, test_obj_lst, view=None):
        def get_view(width, height, test_obj_lst):
            vis_viewer = o3d.visualization.Visualizer()
            vis_viewer.create_window(width=width, height=height)
            for obj in test_obj_lst:
                vis_viewer.add_geometry(obj)
            vis_viewer.run()
            vis_viewer.close()
            vis_viewer.destroy_window()
            curview = vis_viewer.get_view_control().convert_to_pinhole_camera_parameters() 
            return curview
        if view is None:
            self.curview = get_view(width, height, test_obj_lst)
        else:
            self.curview = view
        intrinsics, extrinsics = self.curview.intrinsic, self.curview.extrinsic
        self.width = intrinsics.width
        self.height = intrinsics.height

        app = o3d.visualization.gui.Application.instance
        app.initialize()
        self.main_vis = o3d.visualization.O3DVisualizer(title, intrinsics.width, intrinsics.height)
        self.main_vis.setup_camera(intrinsics.intrinsic_matrix, extrinsics, intrinsics.width, intrinsics.height)    
        self.main_vis.show_settings = False
        self.main_vis.show_skybox(False)        
        app.add_window(self.main_vis)
    
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
        self.main_vis.add_geometry(geometry)

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
        app = o3d.visualization.gui.Application.instance
        app.run()

def draw_marker(points, color=None):
    if color is None:
        color = tuple(np.random.rand(3))
    colors = [color for _ in range(4)]
    sp = o3d.geometry.TriangleMesh()
    for i in range(4):
        sptmp = o3d.geometry.TriangleMesh().create_sphere(radius=0.005)
        sptmp.translate(points[i])
        sptmp.paint_uniform_color(color)
        sp += sptmp
    sp.compute_vertex_normals()
    lines = [[0,1], [1,2], [2,3], [3,0]]
    ls = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(points),
        lines = o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls, sp


def pyramid(translation, rotation, img=None, focal_length=0.01, color=[1,0,0]): # in order to get list of points. use image size
    #points, res = [[-0.25,0.25,0.5], [-0.25,-0.25,0.5], [0.25, -0.25, 0.5], [0.25,0.25, 0.5], [0,0,0]], []
    # Use sensor size of BFLY-31s4c
    h, w = 5.3/100, 7.07/100 #0.5*H/(H+W), 0.5*W/(W+H) # apply image ratio
    # Opencv : z positive
    # scaled for better viz
    points, res = [[w,-h, 2*focal_length], [w, h, 2*focal_length], [-w,h, 2*focal_length], [-w,-h, 2*focal_length], [0,0,0]], []
    result = []
    for p in points:
        tmp = rotation.T @ (np.array(p) - translation) # world2cam matrix에 대해 cam -> world 전환
        res.append(tmp)
    sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.005)
    sphere.translate(res[4])
    sphere.paint_uniform_color(tuple(color))
    lines = [
    [0, 1],
    [0, 3],
    [1, 2],
    [2, 3],
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
    ]
    colors = [color for i in range(len(lines))]
    # define texture
    vertices = np.array([res[3], res[0], res[1], res[2]])
    indices = np.array([[2, 1, 0], [0, 3, 2], [1,2,3], [3,0,1]])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(res),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)    
    result.append(line_set)
    result.append(sphere)
    if img is not None:
        imgMesh =  o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(indices))
        imgMesh.compute_triangle_normals()
        imgMesh.paint_uniform_color((0.6,0.6,0.6))
        result.append(imgMesh)
    return result #line_set