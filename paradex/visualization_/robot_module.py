from paradex.robot import RobotWrapper
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot
import xml.dom.minidom
import trimesh
import copy

# def extract_mesh(o3d_mesh, tg_type='array', device='cuda:0'):
#     # Extract vertices and faces
#     vertices_arr = np.asarray(o3d_mesh.vertices)
#     faces_arr = np.asarray(o3d_mesh.triangles)
#     vertex_colors_arr = np.asarray(o3d_mesh.vertex_colors)
#     if tg_type=='array':
#         return vertices_arr, faces_arr, vertex_colors_arr
#     else:
#         vertices = torch.tensor(vertices_arr, dtype=torch.float32).to(device)  # (V, 3)
#         faces = torch.tensor(faces_arr, dtype=torch.int64).to(device)  # (F, 3)
#         vertex_colors = torch.tensor(vertex_colors_arr, dtype=torch.float32).to(device)  # (V, 3)
#         vertex_colors = vertex_colors.clamp(0, 1)

#         return vertices, faces, vertex_colors
    
class Mesh_Object:
    def __init__(self, obj_path):
        self.target_path = obj_path
        self.mesh_dict = dict()

    def load(self):
        self.mesh_parts["base"] = o3d.io.read_triangle_mesh(self.target_path)
        self.mesh_parts["base"].compute_vertex_normals()
        return

    def get_mesh(self):
        tmp = o3d.geometry.TriangleMesh()
        for k, m in self.mesh_dict.items():
            tmp += m
        tmp.compute_vertex_normals()
        return tmp

class robot_info(Mesh_Object):
    """ Load SAPIEN URDF file """
    def __init__(self, xml_path, init_trans=None, init_rot=None, down_sample=False, pretransform=False, mesh_processing=True):
        super().__init__(xml_path)
        
        self.root_path = Path(xml_path).parent
        self.default_T = dict()
        self.mesh_dict = dict() # per part-mesh
        self.trimesh_dict = dict()
        self.down_sample = down_sample
        self.pretransform = pretransform

        self.link_org = dict()
        self.joint_info = dict()

        self.init_trans = np.zeros(3) if init_trans is None else init_trans  
        self.init_rot = sRot.from_euler('x', 90, degrees=True).as_matrix() if init_rot is None else init_rot # z방향으로 올려야됨
        self.init_transf = np.eye(4)
        self.init_transf[:3,:3] = self.init_rot
        self.init_transf[:3, 3] = self.init_trans

        self.mesh_processing = mesh_processing

        self.link2contact_info = dict()
        self.link2contact_arrows = dict()

        self.load()

    def load(self):
        dom = xml.dom.minidom.parse(str(self.target_path))

        # Extract robot name
        robot = dom.getElementsByTagName("robot")[0]
        robot_name = robot.getAttribute("name")
        print(f"Robot Name: {robot_name}")

        # extract material
        materials = [node for node in robot.childNodes if node.nodeName == "material"]
        materialcolor = dict()
        for material in materials:
            mname = material.getAttribute("name").lower()
            color_nodes = material.getElementsByTagName("color")
            color_nodes[0].getAttribute("rgba")
            materialcolor[mname] = np.array(color_nodes[0].getAttribute("rgba").split(), dtype=np.float64)
        self.materialcolor = materialcolor

        # Extract Joints
        joints = dom.getElementsByTagName("joint")
        print("\nJoints:")
        
        for joint in joints:
            joint_name = joint.getAttribute("name")
            joint_type = joint.getAttribute("type")
            # parent
            # print("------------")
            # print(joint_name, joint.getElementsByTagName("parent"), joint.toxml())
            if len(joint.getElementsByTagName("parent")) == 0:
                continue
            parent = joint.getElementsByTagName("parent")[0].getAttribute("link")
            child = joint.getElementsByTagName("child")[0].getAttribute("link")

            if len(joint.getElementsByTagName("origin"))>0:
                origin_rpy = np.array(joint.getElementsByTagName("origin")[0].getAttribute("rpy").split(), dtype=np.float64)
                origin_xyz = np.array(joint.getElementsByTagName("origin")[0].getAttribute("xyz").split(), dtype=np.float64)
            else:
                origin_rpy, origin_xyz = None, None
            # axis
            axis = [np.array(axis_item.getAttribute("xyz").split(), np.float64) for axis_item in joint.getElementsByTagName("axis")]

            self.joint_info[joint_name] = {'type':joint_type, 'parent':parent, 'child':child, 'origin_xyz':origin_xyz, 'origin_rpy':origin_rpy, 'axis':axis}

        # Extract links
        links = dom.getElementsByTagName("link")
        # print("\nLinks:")
        for link in links:
            link_name = link.getAttribute("name")
            # print(f"- {link_name}")
            visuals = link.getElementsByTagName("visual")
            self.mesh_dict[link_name] = [] # per part 
            self.trimesh_dict[link_name] = []
            #o3d.geometry.TriangleMesh()
            for visual in visuals:
                visual_name = visual.getAttribute("name")

                mesh_subpath = visual.getElementsByTagName("geometry")[0].getElementsByTagName("mesh")[0].getAttribute("filename")
                mesh_path = str(self.root_path/mesh_subpath)
                if len(visual.getElementsByTagName("material")[0].getElementsByTagName("color")) > 0:
                    mesh_color = np.array(visual.getElementsByTagName("material")[0].getElementsByTagName("color")[0].getAttribute("rgba").split(), dtype=np.float64)
                else:
                    mesh_color = self.materialcolor[visual.getElementsByTagName("material")[0].getAttribute("name").lower()]

                o3d_mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=self.mesh_processing)
                # part_mesh.transform(self.init_transf)
                o3d_mesh.compute_vertex_normals()
                o3d_mesh.paint_uniform_color(mesh_color[:3])
                # transform if mesh rpy and xyz is exist
                '''
                    rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                    translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                    pts = np.matmul(rotation, pts.T).T + translation    
                '''

                if o3d_mesh.has_vertex_colors():
                    vertex_colors = np.asarray(o3d_mesh.vertex_colors)   # Scale [0,1] → [0,255]
                    # vertex_colors_with_alpha = np.hstack((vertex_colors, np.full((vertex_colors.shape[0], 1), 255)))  # Add alpha
                
                    trimesh_mesh = trimesh.Trimesh(
                        vertices=np.asarray(o3d_mesh.vertices),
                        faces=np.asarray(o3d_mesh.triangles), vertex_colors=vertex_colors*255, process=self.down_sample)
                else:
                    trimesh_mesh = trimesh.Trimesh(
                        vertices=np.asarray(o3d_mesh.vertices),
                        faces=np.asarray(o3d_mesh.triangles))
                    
                self.mesh_dict[link_name].append(o3d_mesh)
                self.trimesh_dict[link_name].append(trimesh_mesh)

                if len(visual.getElementsByTagName("origin")) > 0 :
                    origin_xyz = visual.getElementsByTagName("origin")[0].getAttribute("xyz")
                    origin_rpy = visual.getElementsByTagName("origin")[0].getAttribute("rpy")
                elif len(visual.getElementsByTagName("inertia")) > 0  and len(visual.getElementsByTagName("inertia")[0].getElementsByTagName("origin"))>0:
                    origin = visual.getElementsByTagName("inertia")[0].getElementsByTagName("origin")[0]
                    origin_xyz = origin.getAttribute("xyz")
                    origin_rpy = origin.getAttribute("rpy")
                elif link_name in self.joint_info and self.joint_info[link_name]['origin_xyz'] is not None:
                    origin_xyz = self.joint_info[joint_name]['origin_xyz']
                    origin_rpy = self.joint_info[joint_name]['origin_rpy']

                self.link_org[link_name]=(origin_xyz, origin_rpy)

    def get_link_mesh(self, link_name, mesh_type='o3d'):
        assert link_name in self.mesh_dict, f'Link Name: {link_name} does not exist in mesh dictionary'

        if mesh_type == 'o3d':
            return self.mesh_dict[link_name]
        elif mesh_type == 'trimesh':
            return self.trimesh_dict[link_name]
        else:
            raise 'Mesh Type is not defined'
                    
    # mesh list 
    def get_mesh(self):
        outlst = []
        for linkn, mlist in self.mesh_dict.items():
            outlst += mlist
        return outlst

def combine_mesh_features(link_list, mesh_info):
    combined_verts = []
    combined_faces = []
    combined_textures = []
    link2vertex_mapping = {}
    offset = 0

    for link_name in link_list:
        vertices, faces, textures = mesh_info[link_name]
        combined_verts.append(vertices)
        combined_faces.append(faces+offset)
        link2vertex_mapping[link_name] = (offset, offset+vertices.shape[0])
        offset+=vertices.shape[0]
        combined_textures.append(textures)
    return np.vstack(combined_verts), np.vstack(combined_faces), np.vstack(combined_textures), link2vertex_mapping

class Robot_Module:
    def __init__(self, urdf_file,  mesh_tg ='all', state=None):
        
        self.robot_obj = robot_info(urdf_file, down_sample=True)
        self.robot_wrapper = RobotWrapper(urdf_file)
        self.mesh_tg = mesh_tg

        self.state = state
        if self.state is not None:
            self.length = self.state.shape[0]
        else:
            self.length = None # for real time

        all_link_names = list(self.robot_obj.mesh_dict.keys())
        self.link_names = {'all': all_link_names, \
                        'hand': [link_nm for link_nm in all_link_names if link_nm.split("_")[0] in \
                            ['thumb','index','middle','ring','palm']],
                        'arm': [link_nm for link_nm in all_link_names if link_nm.split("_")[0] not in \
                            ['thumb','index','middle','ring','palm']],
                 }
        self.link_list = []
        for link_nm, mesh_items in self.robot_obj.mesh_dict.items():
            if mesh_items!=[] and link_nm in self.link_names[self.mesh_tg]:
                self.link_list.append(link_nm)


        self.mesh_info = {}
        for link_name in self.link_list:
            o3d_mesh = self.robot_obj.get_link_mesh(link_name, mesh_type='o3d')
            
            vertices = np.asarray(o3d_mesh[0].vertices)
            faces = np.asarray(o3d_mesh[0].triangles)
            textures = np.asarray(o3d_mesh[0].vertex_colors)            
            
            self.mesh_info[link_name] = (vertices, faces, textures)
        _, self.combined_faces, self.combined_textures, self.link2vertex_mapping = combine_mesh_features(self.link_list, self.mesh_info)
    
    def forward_kinematic(self, fidx, state_realtime=None):
        if state_realtime is None:
            if fidx >= self.state.shape[0]:
                fidx = self.state.shape[0]-1
            self.robot_wrapper.compute_forward_kinematics(self.state[fidx])
        else:
            self.robot_wrapper.compute_forward_kinematics(state_realtime)
            

    def get_mesh(self, fidx, base_T=np.eye(4), mesh_tg = 'all', state_realtime=None):

        self.forward_kinematic(fidx, state_realtime=state_realtime)

        vis_list = []
        for link_nm in self.link_list:
            
            link_pose = base_T@self.robot_wrapper.get_link_pose(self.robot_wrapper.get_link_index(link_nm))
            # print(f'{link_nm}: {link_pose}')
            for mesh in self.robot_obj.mesh_dict[link_nm]:
                vis_list.append(copy.deepcopy(mesh).transform(link_pose))

        return vis_list
    

    def get_T_dict(self, fidx, base_T=np.eye(4), state_realtime=None):
        self.forward_kinematic(fidx, state_realtime=state_realtime)

        T_dict = {}
        for link_nm in self.link_list:
            link_pose = base_T@self.robot_wrapper.get_link_pose(self.robot_wrapper.get_link_index(link_nm))
            T_dict[link_nm] = link_pose
        return T_dict
    

    def get_vertices(self, fidx, base_T=np.eye(4), mesh_tg = 'all', state_realtime=None):
        
        # make mapping to link_nm vertex_index
        self.forward_kinematic(fidx, state_realtime=state_realtime)

        vertices_list = []
        for link_nm in self.link_list:
            link_pose = base_T@self.robot_wrapper.get_link_pose(self.robot_wrapper.get_link_index(link_nm))
            transformed_vertices = ((link_pose[:3,:3]@self.mesh_info[link_nm][0].T).T)+link_pose[:3,3]
            vertices_list.append(transformed_vertices)

        return vertices_list
    

    def get_combined_trimesh(self, fidx, base_T=np.array(4), mesh_tg = 'all', down_sample=True, state_realtime=None):
        vertices_list = self.get_vertices(fidx, base_T, mesh_tg, state_realtime=state_realtime)

        vertices_arr = np.vstack(vertices_list)
        combined_trimesh = trimesh.Trimesh(
            vertices=vertices_arr,
            faces=self.combined_faces, vertex_colors=self.combined_textures, process=down_sample)
        
        return combined_trimesh, vertices_arr


    def set_combined_features(self):
        return self.combined_faces, self.combined_textures
    