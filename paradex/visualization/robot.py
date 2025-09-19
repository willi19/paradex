import trimesh
import numpy as np
import pinocchio as pin

from paradex.robot.robot_wrapper import RobotWrapper
# Implementing

class RobotModule:
    def __init__(self, urdf_path, coll_sphere_path = None):
        self.mesh = None
        self.robot = RobotWrapper(urdf_path)
        
        init_q = np.zeros(self.robot.dof)
        self.robot.compute_forward_kinematics(init_q)
        
        self._base_link = "world"
        
        self.visual_model = pin.buildGeomFromUrdf(
            self.robot.model, urdf_path, pin.GeometryType.VISUAL
        )
        
        self.collision_model = pin.buildGeomFromUrdf(
            self.robot.model, urdf_path, pin.GeometryType.COLLISION
        )
    
    def _create_scene(
        self,
        use_collision_geometry=False,
        load_geometry=True,
        force_mesh=False,
        force_single_geometry_per_link=False,
    ):
        s = trimesh.scene.Scene(base_frame=self._base_link)

        for j in self.robot.joints:
            matrix, _ = self._forward_kinematics_joint(j)

            s.graph.update(frame_from=j.parent, frame_to=j.child, matrix=matrix)

        for l in self.robot.links:
            if l.name not in s.graph.nodes and l.name != s.graph.base_frame:
                _logger.warning(
                    f"{l.name} not connected via joints. Will add link to base frame."
                )
                s.graph.update(frame_from=s.graph.base_frame, frame_to=l.name)

            meshes = l.collisions if use_collision_geometry else l.visuals
            self._add_geometries_to_scene(
                s,
                geometries=meshes,
                link_name=l.name,
                load_geometry=load_geometry,
                force_mesh=force_mesh,
                force_single_geometry=force_single_geometry_per_link,
                skip_materials=use_collision_geometry,
            )

        return s
    
    def _add_geometries_to_scene(
        self,
        s,
        geometries,
        link_name,
        load_geometry,
        force_mesh,
        force_single_geometry,
        skip_materials,
    ):
        if force_single_geometry:
            tmp_scene = trimesh.Scene(base_frame=link_name)

        first_geom_name = None

        for v in geometries:
            if v.geometry is not None:
                if first_geom_name is None:
                    first_geom_name = v.name

                new_s = self._geometry2trimeshscene(
                    geometry=v.geometry,
                    load_file=load_geometry,
                    force_mesh=force_mesh,
                    skip_materials=skip_materials,
                )
                if new_s is not None:
                    origin = v.origin if v.origin is not None else np.eye(4)

                    if force_single_geometry:
                        for name in new_s.graph.nodes_geometry:
                            T, geom_name = new_s.graph.get(name)
                            geom = new_s.geometry[geom_name]

                            if isinstance(v, Visual):
                                apply_visual_color(geom, v, self._material_map)
                            tmp_scene.add_geometry(
                                geometry=geom,
                                geom_name=v.name,
                                parent_node_name=link_name,
                                transform=origin @ T,
                            )
                    else:
                        for name in new_s.graph.nodes_geometry:
                            T, geom_name = new_s.graph.get(name)
                            geom = new_s.geometry[geom_name]
                            
                            if isinstance(v, Visual):
                                apply_visual_color(geom, v, self._material_map)
                            s.add_geometry(
                                geometry=geom,
                                geom_name=v.name,
                                parent_node_name=link_name,
                                transform=origin @ T,
                            )

        if force_single_geometry and len(tmp_scene.geometry) > 0:
            s.add_geometry(
                geometry=tmp_scene.dump(concatenate=True),
                geom_name=first_geom_name,
                parent_node_name=link_name,
                transform=np.eye(4),
            )
            
    def update():
        pass