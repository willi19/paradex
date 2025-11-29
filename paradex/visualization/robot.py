from __future__ import annotations

from typing import List, Tuple

import numpy as np
import yourdfpy
import trimesh
from trimesh.scene import Scene

import os
from scipy.spatial.transform import Rotation as R

class RobotModule():
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.urdf = yourdfpy.URDF.load(urdf_path, build_scene_graph=True)
        
        self.joint_names = self.urdf.actuated_joint_names
        self.num_joints = len(self.joint_names)
        self.joint_limits = {joint.name: (joint.limit.lower, joint.limit.upper) for joint in self.urdf.actuated_joints if joint.type != 'fixed' and joint.limit is not None}
    
    def get_joint_limits(self) -> dict:
        return self.joint_limits
    
    def get_num_joints(self) -> int:
        return self.num_joints
    
    def get_joint_names(self) -> List[str]:
        return self.joint_names
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> dict:
        assert len(joint_angles) == self.num_joints, "Joint angles length must match number of joints"
        cfg = {name: angle for name, angle in zip(self.joint_names, joint_angles)}
        self.urdf.update_cfg(cfg)
        link_poses = {}
        for link in self.urdf.links:
            T = self.urdf.get_transform(link.name, self.urdf.base_link)
            link_poses[link.name] = T
        return link_poses

    def update_cfg(self, cfg):
        self.urdf.update_cfg(cfg)
    
    @property
    def scene(self) -> Scene:
        return self.urdf.scene
    
    @property
    def collision_scene(self) -> Scene:
        return self.urdf.collision_scene
    
    @property
    def joint_map(self) -> dict:
        return self.urdf.joint_map

    def get_transform(self, from_link: str, to_link: str, collision_geometry: bool) -> np.ndarray:
        return self.urdf.get_transform(from_link, to_link, collision_geometry=collision_geometry)

    def get_link_mesh(self, link_name: str, collision_geometry: bool = False) -> Trimesh:
        """
        Get combined mesh for a single link.
        
        Args:
            link_name: Name of the link
            collision_geometry: Use collision mesh if True, visual mesh if False
        
        Returns:
            Combined trimesh for the link
        """
        meshes = self.get_mesh_list(link_name, collision_geometry)
        
        if len(meshes) == 0:
            return None
        
        combined = None
        for geom in meshes:
            mesh = geom.geometry  # trimesh object
            # Apply geometry origin transform
            if geom.origin is not None:
                mesh = mesh.copy()
                mesh.apply_transform(geom.origin)
            
            if combined is None:
                combined = mesh
            else:
                combined = trimesh.util.concatenate([combined, mesh])
        
        return combined
    
    def get_robot_mesh(self, collision_geometry: bool = False) -> Trimesh:
        """
        Get combined mesh for entire robot in current configuration.
        
        Args:
            collision_geometry: Use collision mesh if True, visual mesh if False
        
        Returns:
            Combined trimesh for all links
        """
        combined = None
        
        for link in self.urdf.links:
            link_mesh = self.get_link_mesh(link.name, collision_geometry)
            
            if link_mesh is None:
                continue
            
            # Get link transform
            T = self.urdf.get_transform(link.name, self.urdf.base_link)
            
            # Apply transform
            link_mesh = link_mesh.copy()
            link_mesh.apply_transform(T)
            
            # Combine
            if combined is None:
                combined = link_mesh
            else:
                combined = trimesh.util.concatenate([combined, link_mesh])
        
        return combined
    
    def get_robot_mesh_list(self, collision_geometry: bool = False) -> List[Tuple[str, Trimesh]]:
        """
        Get list of meshes for each link with transforms applied.
        
        Args:
            collision_geometry: Use collision mesh if True, visual mesh if False
        
        Returns:
            List of (link_name, mesh) tuples
        """
        mesh_list = []
        
        for link in self.urdf.links:
            link_mesh = self.get_link_mesh(link.name, collision_geometry)
            
            if link_mesh is None:
                continue
            
            # Get link transform
            T = self.urdf.get_transform(link.name, self.urdf.base_link)
            
            # Apply transform
            link_mesh = link_mesh.copy()
            link_mesh.apply_transform(T)
            
            mesh_list.append((link.name, link_mesh))
        
        return mesh_list