import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R
from trimesh import Scene
import viser
import trimesh
from typing import List, Tuple

from paradex.visualization.robot import RobotModule  

class ViserRobotModule():
    def __init__(self, target,#: viser.ViserServer | viser.ClientHandle,
                 urdf_path, 
                 scale: float = 1.0,
                 root_node_name: str = "/",
                 load_meshes=True, 
                 load_collision_meshes=False):
        self._urdf = RobotModule(urdf_path)
        self._target = target
        self._scale = scale
        self._load_meshes = load_meshes
        self._load_collision_meshes = load_collision_meshes
        self._joint_frames: List[viser.FrameHandle] = []
        self._meshes: List[viser.MeshHandle] = []
        num_joints_to_repeat = 0
        if load_meshes:
            if self.urdf.scene is not None:
                num_joints_to_repeat += 1
                self._visual_root_frame = self._add_joint_frames_and_meshes(
                    self.urdf.scene,
                    root_node_name,
                    collision_geometry=False
                )

        if load_collision_meshes:
            if self.urdf.collision_scene is not None:
                num_joints_to_repeat += 1
                self._collision_root_frame = self._add_joint_frames_and_meshes(
                    self.urdf.collision_scene,
                    root_node_name,
                    collision_geometry=True
                )
        self._joint_map_values = [*self._urdf.joint_map.values()] * num_joints_to_repeat
                
    def change_color(self, name_list, color: Tuple[float, float, float]) -> None:
        """Change the color of the visualized URDF."""
        for name in name_list:
            self._target.scene.set_mesh_color(name, color)
    
    @property
    def show_visual(self) -> bool:
        """Returns whether the visual meshes are currently visible."""
        return self._visual_root_frame is not None and self._visual_root_frame.visible

    @show_visual.setter
    def show_visual(self, visible: bool) -> None:
        """Set whether the visual meshes are currently visible."""
        if self._visual_root_frame is not None:
            self._visual_root_frame.visible = visible

    @property
    def show_collision(self) -> bool:
        """Returns whether the collision meshes are currently visible."""
        return (
            self._collision_root_frame is not None
            and self._collision_root_frame.visible
        )

    @show_collision.setter
    def show_collision(self, visible: bool) -> None:
        """Set whether the collision meshes are currently visible."""
        if self._collision_root_frame is not None:
            self._collision_root_frame.visible = visible

    @property
    def urdf(self) -> RobotModule:
        """Get the underlying URDF object."""
        return self._urdf
    
    def remove(self) -> None:
        """Remove URDF from scene."""
        # Some of this will be redundant, since children are removed when
        # parents are removed.
        for frame in self._joint_frames:
            frame.remove()
        for mesh in self._meshes:
            mesh.remove()
               
    def update_cfg(self, configuration: np.ndarray) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        for joint, frame_handle in zip(self._joint_map_values, self._joint_frames):
            assert isinstance(joint, yourdfpy.Joint)
            T_parent_child = self._urdf.get_transform(
                joint.child, joint.parent, collision_geometry=not self._load_meshes
            )
            frame_handle.wxyz = R.from_matrix(T_parent_child[:3, :3]).as_quat()
            frame_handle.position = T_parent_child[:3, 3] * self._scale
    
    # def _add_joint_frames_and_meshes(
    #     self,
    #     scene: Scene,
    #     root_node_name: str,
    #     collision_geometry: bool,
    #     mesh_color_override: tuple[float, float, float]
    #     | tuple[float, float, float, float]
    #     | None,
    # ) -> viser.FrameHandle:
    #     """
    #     Helper function to add joint frames and meshes to the ViserUrdf object.
    #     """
    #     prefix = "collision" if collision_geometry else "visual"
    #     prefixed_root_node_name = (f"{root_node_name}/{prefix}").replace("//", "/")
    #     root_frame = self._target.scene.add_frame(
    #         prefixed_root_node_name, show_axes=False
    #     )

    #     # Add coordinate frame for each joint.
    #     for joint in self._urdf.joint_map.values():
    #         assert isinstance(joint, yourdfpy.Joint)
    #         self._joint_frames.append(
    #             self._target.scene.add_frame(
    #                 _viser_name_from_frame(
    #                     scene,
    #                     joint.child,
    #                     prefixed_root_node_name,
    #                 ),
    #                 show_axes=False,
    #             )
    #         )

    #     # Add the URDF's meshes/geometry to viser.
    #     for link_name, mesh in scene.geometry.items():
    #         assert isinstance(mesh, trimesh.Trimesh)
    #         T_parent_child = self._urdf.get_transform(
    #             link_name,
    #             scene.graph.transforms.parents[link_name],
    #             collision_geometry=collision_geometry,
    #         )
    #         name = _viser_name_from_frame(scene, link_name, prefixed_root_node_name)

    #         # Scale + transform the mesh. (these will mutate it!)
    #         #
    #         # It's important that we use apply_transform() instead of unpacking
    #         # the rotation/translation terms, since the scene graph transform
    #         # can also contain scale and reflection terms.
    #         mesh = mesh.copy()
    #         mesh.apply_scale(self._scale)
    #         mesh.apply_transform(T_parent_child)

    #         if mesh_color_override is None:
    #             self._meshes.append(self._target.scene.add_mesh_trimesh(name, mesh))
    #         elif len(mesh_color_override) == 3:
    #             self._meshes.append(
    #                 self._target.scene.add_mesh_simple(
    #                     name,
    #                     mesh.vertices,
    #                     mesh.faces,
    #                     color=mesh_color_override,
    #                 )
    #             )
    #         elif len(mesh_color_override) == 4:
    #             self._meshes.append(
    #                 self._target.scene.add_mesh_simple(
    #                     name,
    #                     mesh.vertices,
    #                     mesh.faces,
    #                     color=mesh_color_override[:3],
    #                     opacity=mesh_color_override[3],
    #                 )
    #             )
    #         else:
    #             assert_never(mesh_color_override)
    #     return root_frame
            
    def _add_joint_frames_and_meshes(
        self,
        scene: Scene,
        root_node_name: str,
        collision_geometry: bool
    ) :
        """
        Helper function to add joint frames and meshes to the ViserUrdf object.
        """
        prefix = "collision" if collision_geometry else "visual"
        prefixed_root_node_name = (f"{root_node_name}/{prefix}").replace("//", "/")
        root_frame = self._target.scene.add_frame(
            prefixed_root_node_name, show_axes=False
        )
        # Add coordinate frame for each joint.
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            self._joint_frames.append(
                self._target.scene.add_frame(
                    _viser_name_from_frame(
                        scene,
                        joint.child,
                        prefixed_root_node_name,
                    ),
                    show_axes=True,
                )
            )

        # Add the URDF's meshes/geometry to viser.
        for link_name, mesh in scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = self._urdf.get_transform(
                link_name,
                scene.graph.transforms.parents[link_name],
                collision_geometry=collision_geometry,
            )
            name = _viser_name_from_frame(scene, link_name, prefixed_root_node_name)
            # Scale + transform the mesh. (these will mutate it!)
            #
            # It's important that we use apply_transform() instead of unpacking
            # the rotation/translation terms, since the scene graph transform
            # can also contain scale and reflection terms.
            mesh = mesh.copy()
            mesh.apply_scale(self._scale)
            mesh.apply_transform(T_parent_child)

            self._meshes.append(self._target.scene.add_mesh_trimesh(name, mesh))
            # elif len(mesh_color_override) == 3:
            #     self._meshes.append(
            #         self._target.scene.add_mesh_simple(
            #             name,
            #             mesh.vertices,
            #             mesh.faces,
            #             color=mesh_color_override,
            #         )
            #     )
            # elif len(mesh_color_override) == 4:
            #     self._meshes.append(
            #         self._target.scene.add_mesh_simple(
            #             name,
            #             mesh.vertices,
            #             mesh.faces,
            #             color=mesh_color_override[:3],
            #             opacity=mesh_color_override[3],
            #         )
            #     )
            # else:
            #     assert_never(mesh_color_override)
        return root_frame
    

def _viser_name_from_frame(
    scene: Scene,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")
    frames = []
    while frame_name != scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = scene.graph.transforms.parents[frame_name]
    if root_node_name != "/":
        frames.append(root_node_name)
    return "/".join(frames[::-1])
