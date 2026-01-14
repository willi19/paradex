import os
import sys

import torch
import pickle
import trimesh
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import xml.dom.minidom
import copy
from pathlib import Path

PROJECT_PATH = Path(__file__).absolute().parent.parent
sys.path.append(str(PROJECT_PATH))

from paradex.robot.robot_wrapper_deprecated import RobotWrapper
# from utils.renderer_utils import extract_mesh, combine_mesh_features
# from utils.geometry import rotation_matrix_from_vectors
# from utils.vis_utils import cmap

# robot_name = 'xarm6'
# robot_asset_dir = PROJECT_PATH/'robot'
# arm_robot = 'xarm'
# hand_robot = 'inspire'
# robot_asset_file = robot_asset_dir/f"{arm_robot}_{hand_robot}.urdf"

# from paradex.io.robot_controller.under_test import inspire_contact_info as contact_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rotation_matrix_from_vectors(v1, v2):
    """
    Returns the rotation matrix that aligns v1 to v2.
    v1 and v2 are 3-element vectors.
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # Cross product and dot product
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    # Check for special cases
    if np.allclose(dot, 1.0):
        return np.eye(3)  # No rotation needed
    if np.allclose(dot, -1.0):
        # 180 degree rotation around any axis perpendicular to v1
        orthogonal = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal)
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + 2 * (K @ K)  # since sin(π)=0, (1 - cos(π)) = 2

    # Rodrigues' formula
    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])

    R = np.eye(3) + K + K @ K * ((1 - dot) / (np.linalg.norm(cross) ** 2))
    return R

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
        
        self.scene_path = Path(xml_path).parent
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
            if len(joint_name)<=0:
                continue
            print(f"{joint_name}, {joint_type}, L: {len(joint_name)}")

            # parent
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
                mesh_path = str(self.scene_path/mesh_subpath)
                materials = visual.getElementsByTagName("material")

                if len(materials) > 0:
                    material = materials[0]

                    colors = material.getElementsByTagName("color")
                    if len(colors) > 0:
                        # case 1: <material><color .../>
                        mesh_color = np.array(
                            colors[0].getAttribute("rgba").split(),
                            dtype=np.float64
                        )
                    else:
                        # case 2: <material name="xxx"/>
                        mat_name = material.getAttribute("name").lower()
                        if mat_name in self.materialcolor:
                            mesh_color = self.materialcolor[mat_name]
                        else:
                            # fallback
                            mesh_color = np.array([0.7, 0.7, 0.7, 1.0])
                else:
                    # case 3: material 자체가 없음
                    mesh_color = np.array([0.7, 0.7, 0.7, 1.0])
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

        # make default T 
        # if self.pretransform:

        #     if parent=='world':
        #         parent_default_T = np.eye(4, dtype=np.float64)
        #     else:
        #         assert self.joint_info[joint_name]['parent'] in self.default_T, f'Parent of {joint_name} should be processed first'
        #         parent_default_T = self.default_T

        #     cur_default_T = np.eye(4, dtype=np.float64)
        #     if self.joint_info[joint_name]['origin_rpy'] is not None and len(self.joint_info[joint_name]['origin_rpy'])>0 :
        #         # NOTE: visualizer fails if we use joint_origin_xyz from urdf file.
        #         cur_default_T[:3,:3] = transforms3d.euler.euler2mat(*self.joint_info[joint_name]['origin_rpy'])
        #     # joint_origin_xyz = np.zeros(3)
        #     # cur_default_T[:3,3] = joint_origin_xyz
        #     self.default_T[joint_name] = cur_default_T@parent_default_T

        #     # rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
        #     # translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
        #     if link_name in self.joint_info:
        #         o3d_mesh = o3d_mesh.transform(self.default_T[link_name])
        #         print(f'{link_name} use default T for mesh transformation')
        
        # for link_nm in contact_info.contact_tg:
        #     mesh = self.mesh_dict[link_nm]
        #     contact_pts_list, contact_normal_list = contact_info.get_contact_ctr(link_nm, np.array(mesh[0].vertices), np.array(mesh[0].triangles),  \
        #                                                                          np.array(mesh[0].vertex_normals), np.array(mesh[0].triangle_normals))
        #     self.link2contact_info[link_nm] = (contact_pts_list, contact_normal_list)
        #     self.link2contact_arrows[link_nm] = {}
            
        #     for cidx in range(len(contact_normal_list)):
        #         contact_point, contact_normal = contact_pts_list[cidx], contact_normal_list[cidx]
        #         rotmat2normal = rotation_matrix_from_vectors(np.array([0,0,1]), contact_normal)
        #         initial_T = np.eye(4)
        #         initial_T[:3,:3] = rotmat2normal
        #         initial_T[:3,3] = contact_point
        #         self.link2contact_arrows[link_nm][f'{link_nm}_{cidx}'] = initial_T


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
    
