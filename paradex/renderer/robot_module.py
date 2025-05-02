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
import transforms3d

cur_dir = Path(__file__).parent
sys.path.append(cur_dir)
from dex_robot.utils.robot_wrapper import RobotWrapper
from dex_robot.utils.file_io import rsc_path
robot_name = 'xarm6'


robot_asset_dir = Path(rsc_path) / 'xarm6'
robot_asset_file = robot_asset_dir / "xarm6_allegro_wrist_mounted_rotate.urdf"


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
    def __init__(self, xml_path, init_trans=None, init_rot=None, down_sample=False, pretransform=False):
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

        self.load()

    def load(self):
        dom = xml.dom.minidom.parse(str(self.target_path))

        # Extract robot name
        robot = dom.getElementsByTagName("robot")[0]
        robot_name = robot.getAttribute("name")
        # print(f"Robot Name: {robot_name}")

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
        # print("\nJoints:")
        
        for joint in joints:
            joint_name = joint.getAttribute("name")
            joint_type = joint.getAttribute("type")
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
                mesh_path = str(self.root_path/mesh_subpath)
                if len(visual.getElementsByTagName("material")[0].getElementsByTagName("color")) > 0:
                    mesh_color = np.array(visual.getElementsByTagName("material")[0].getElementsByTagName("color")[0].getAttribute("rgba").split(), dtype=np.float64)
                else:
                    mesh_color = self.materialcolor[visual.getElementsByTagName("material")[0].getAttribute("name").lower()]

                o3d_mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
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
    

class Robot_Module:
    def __init__(self, urdf_file:str=str(robot_asset_file), state:np.ndarray=None):
        self.robot_obj = robot_info(urdf_file)
        self.robot_wrapper = RobotWrapper(urdf_file)

        self.state = state
        self.length = self.state.shape[0]

        all_link_names = list(self.robot_obj.mesh_dict.keys())
        self.link_names = {'all': all_link_names, \
                        'hand': [link_nm for link_nm in all_link_names if link_nm.split("_")[0] in \
                            ['thumb','index','middle','ring','palm']],
                        'arm': [link_nm for link_nm in all_link_names if link_nm.split("_")[0] not in \
                            ['thumb','index','middle','ring','palm']],
                 }

    def get_mesh(self, fidx, base_T=np.eye(4), mesh_tg = 'all'):

        self.robot_wrapper.compute_forward_kinematics(self.state[fidx])
        # all link
        link_list = [ link_nm for link_nm, mesh_items in self.robot_obj.mesh_dict.items() if mesh_items!=[] ] # 28

        vis_list = []
        for link_nm in link_list:
            if link_nm in self.link_names[mesh_tg]:
                link_pose = base_T@self.robot_wrapper.get_link_pose(self.robot_wrapper.get_link_index(link_nm))
                for mesh in self.robot_obj.mesh_dict[link_nm]:
                    vis_list.append(copy.deepcopy(mesh).transform(link_pose))

        return vis_list
    

def get_trajectory(capture_dir):
    hand_action = np.load(capture_dir/'hand'/'action.npy') # FX16
    hand_state = np.load(capture_dir/'hand'/'state.npy') # FX16
    arm_action = np.load(capture_dir/'arm'/'action.npy') # FX6
    arm_state = np.load(capture_dir/'arm'/'state.npy') # FX6

    robot_traj = np.concatenate([arm_state, hand_state], axis=1) # FX22

    return robot_traj


if __name__ == "__main__":
    robot_obj = robot_info(str(robot_asset_file)) # parsed urdf
    # load action and state
    capture_dir = Path('/home/jisoo/data2/paradex/captured_data/spray/0')
    hand_action = np.load(capture_dir/'hand'/'action.npy') # FX16
    hand_state = np.load(capture_dir/'hand'/'state.npy') # FX16
    arm_action = np.load(capture_dir/'arm'/'action.npy') # FX6
    arm_state = np.load(capture_dir/'arm'/'state.npy') # FX6

    robot_traj = np.concatenate([arm_state, hand_state], axis=1) # FX22

    robot_by_robotwrapper = RobotWrapper(str(robot_asset_file))
    fidx = 0
    robot_by_robotwrapper.compute_forward_kinematics(robot_traj[fidx])
    link_list = [ link_nm for link_nm, mesh_items in robot_obj.mesh_dict.items() if mesh_items!=[] ] # 28

    vis_list = []
    for link_nm in link_list:
        link_pose = robot_by_robotwrapper.get_link_pose(robot_by_robotwrapper.get_link_index(link_nm))
        print(f'{link_nm}: {link_pose}')
        for mesh in robot_obj.mesh_dict[link_nm]:
            vis_list.append(copy.deepcopy(mesh).transform(link_pose))
    
    o3d.visualization.draw_geometries(vis_list)
