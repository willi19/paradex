"""Geometric and kinematic validation of a robot grasp sequence.

The validator deliberately answers the question that meshes and trajectories
can answer: whether a sequence is collision-free where it should be, has the
requested hand-object contacts, and keeps the object rigidly attached to a
reference link while grasping.  Force closure, friction, actuator torque, and
object dynamics require additional physical parameters and are out of scope.

All meshes and transforms are assumed to use the same metric coordinate frame.
Transforms follow ``world_T_local`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

ArrayLike = Union[np.ndarray, Sequence[float]]
PoseProvider = Callable[[np.ndarray], Mapping[str, np.ndarray]]


def _as_finite_array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _validate_times(times: ArrayLike) -> np.ndarray:
    result = _as_finite_array(times, "times").reshape(-1)
    if result.size == 0:
        raise ValueError("times must not be empty")
    if result.size > 1 and np.any(np.diff(result) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return result


def _validate_transform(transform: ArrayLike, name: str) -> np.ndarray:
    result = _as_finite_array(transform, name)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.999:
        raise ValueError(f"{name} rotation is not right-handed")
    return result


def _rotation_error(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = rotation_a.T @ rotation_b
    return float(np.linalg.norm(Rotation.from_matrix(delta).as_rotvec()))


class SampledTrajectory:
    """Piecewise-linear trajectory for robot/hand parameters."""

    def __init__(self, times: ArrayLike, values: ArrayLike):
        self.times = _validate_times(times)
        self.values = _as_finite_array(values, "values")
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        if self.values.ndim != 2:
            raise ValueError(
                f"values must have shape (N, D), got {self.values.shape}"
            )
        if self.values.shape[0] != self.times.size:
            raise ValueError("times and values must have the same length")

    @property
    def start_time(self) -> float:
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        return float(self.times[-1])

    def at(self, time: float) -> np.ndarray:
        if time < self.start_time - 1e-12 or time > self.end_time + 1e-12:
            raise ValueError(
                f"time {time} is outside [{self.start_time}, {self.end_time}]"
            )
        if self.times.size == 1:
            return self.values[0].copy()
        return np.array(
            [
                np.interp(time, self.times, self.values[:, index])
                for index in range(self.values.shape[1])
            ],
            dtype=float,
        )


class CallableTrajectory:
    """A bounded parameter trajectory backed by ``parameters = f(time)``."""

    def __init__(
        self,
        evaluator: Callable[[float], ArrayLike],
        start_time: float,
        end_time: float,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("trajectory bounds must be finite")
        if end_time < start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        self.evaluator = evaluator
        self._start_time = float(start_time)
        self._end_time = float(end_time)

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def end_time(self) -> float:
        return self._end_time

    def at(self, time: float) -> np.ndarray:
        if time < self.start_time - 1e-12 or time > self.end_time + 1e-12:
            raise ValueError(
                f"time {time} is outside [{self.start_time}, {self.end_time}]"
            )
        value = _as_finite_array(
            self.evaluator(float(time)), "trajectory value"
        )
        return value.reshape(-1)


class PoseTrajectory:
    """SE(3) trajectory with linear translation and quaternion interpolation.

    Accepted pose formats are:

    * ``(N, 4, 4)`` homogeneous transforms;
    * ``(N, 6)`` as ``[x, y, z, rx, ry, rz]`` where rotation is a rotation
      vector by default, or Euler angles when ``rotation_format="euler_xyz"``;
    * ``(N, 7)`` as ``[x, y, z, qx, qy, qz, qw]``.
    """

    def __init__(
        self,
        times: ArrayLike,
        poses: ArrayLike,
        rotation_format: str = "rotvec",
    ):
        self.times = _validate_times(times)
        self.poses = self._to_transforms(poses, rotation_format)
        if self.poses.shape[0] != self.times.size:
            raise ValueError("times and poses must have the same length")
        self._rotations = Rotation.from_matrix(self.poses[:, :3, :3])
        self._slerp = (
            Slerp(self.times, self._rotations) if self.times.size > 1 else None
        )

    @staticmethod
    def _to_transforms(poses: ArrayLike, rotation_format: str) -> np.ndarray:
        array = _as_finite_array(poses, "poses")
        if array.ndim == 2 and array.shape == (4, 4):
            array = array[None, ...]
        if array.ndim == 3 and array.shape[1:] == (4, 4):
            transforms = array.copy()
        elif array.ndim == 2 and array.shape[1] in (6, 7):
            transforms = np.repeat(np.eye(4)[None, ...], array.shape[0], axis=0)
            transforms[:, :3, 3] = array[:, :3]
            if array.shape[1] == 7:
                rotations = Rotation.from_quat(array[:, 3:7])
            elif rotation_format == "rotvec":
                rotations = Rotation.from_rotvec(array[:, 3:6])
            elif rotation_format == "euler_xyz":
                rotations = Rotation.from_euler("xyz", array[:, 3:6])
            else:
                raise ValueError(
                    "rotation_format must be 'rotvec' or 'euler_xyz' for 6D poses"
                )
            transforms[:, :3, :3] = rotations.as_matrix()
        else:
            raise ValueError(
                "poses must have shape (N, 4, 4), (N, 6), or (N, 7)"
            )
        for index, transform in enumerate(transforms):
            _validate_transform(transform, f"poses[{index}]")
        return transforms

    @property
    def start_time(self) -> float:
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        return float(self.times[-1])

    def at(self, time: float) -> np.ndarray:
        if time < self.start_time - 1e-12 or time > self.end_time + 1e-12:
            raise ValueError(
                f"time {time} is outside [{self.start_time}, {self.end_time}]"
            )
        if self.times.size == 1:
            return self.poses[0].copy()
        result = np.eye(4)
        result[:3, 3] = [
            np.interp(time, self.times, self.poses[:, axis, 3])
            for axis in range(3)
        ]
        result[:3, :3] = self._slerp(float(time)).as_matrix()
        return result


@dataclass(frozen=True)
class LinkGeometry:
    """A link mesh expressed in the link's local frame."""

    name: str
    mesh: trimesh.Trimesh
    group: str = "robot"
    local_transform: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("link name must not be empty")
        if self.group not in ("robot", "hand"):
            raise ValueError("link group must be 'robot' or 'hand'")
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError(f"mesh for {self.name} must be trimesh.Trimesh")
        if self.mesh.vertices.size == 0 or self.mesh.faces.size == 0:
            raise ValueError(f"mesh for {self.name} must not be empty")
        transform = _validate_transform(
            self.local_transform, f"{self.name}.local_transform"
        )
        object.__setattr__(self, "local_transform", transform.copy())


class ArticulatedMeshModel:
    """Link-local meshes plus a parameter-to-link-pose FK callback."""

    def __init__(
        self,
        links: Sequence[LinkGeometry],
        pose_provider: PoseProvider,
        disabled_self_collision_pairs: Iterable[Tuple[str, str]] = (),
        joint_lower: Optional[ArrayLike] = None,
        joint_upper: Optional[ArrayLike] = None,
        velocity_limits: Optional[ArrayLike] = None,
    ):
        if not links:
            raise ValueError("links must not be empty")
        if not callable(pose_provider):
            raise TypeError("pose_provider must be callable")
        names = [link.name for link in links]
        if len(names) != len(set(names)):
            raise ValueError("link names must be unique")
        self.links = {link.name: link for link in links}
        self.pose_provider = pose_provider
        self.disabled_self_collision_pairs = {
            frozenset(pair) for pair in disabled_self_collision_pairs
        }
        if any(len(pair) != 2 for pair in self.disabled_self_collision_pairs):
            raise ValueError("each disabled self-collision pair needs two links")
        self.joint_lower = self._optional_vector(joint_lower, "joint_lower")
        self.joint_upper = self._optional_vector(joint_upper, "joint_upper")
        self.velocity_limits = self._optional_vector(
            velocity_limits, "velocity_limits"
        )
        if (self.joint_lower is None) != (self.joint_upper is None):
            raise ValueError("joint_lower and joint_upper must be provided together")
        if self.joint_lower is not None:
            joint_upper = self.joint_upper
            assert joint_upper is not None
            if self.joint_lower.shape != joint_upper.shape:
                raise ValueError("joint limit arrays must have the same shape")
            if np.any(self.joint_lower > joint_upper):
                raise ValueError("joint_lower must not exceed joint_upper")
        if self.velocity_limits is not None and np.any(self.velocity_limits <= 0.0):
            raise ValueError("velocity_limits must be positive")

    @staticmethod
    def _optional_vector(
        value: Optional[ArrayLike], name: str
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        return _as_finite_array(value, name).reshape(-1)

    @classmethod
    def from_robot_module(
        cls,
        robot_module: Any,
        hand_links: Iterable[str] = (),
        collision_geometry: bool = False,
        disabled_self_collision_pairs: Iterable[Tuple[str, str]] = (),
        disable_adjacent_collisions: bool = True,
        velocity_limits: Optional[ArrayLike] = None,
    ) -> "ArticulatedMeshModel":
        """Build from :class:`paradex.visualization.robot.RobotModule`.

        The adapter is intentionally duck-typed to keep this module independent
        of Open3D and URDF parsing imports.
        """

        scene = (
            getattr(robot_module, "collision_scene", None)
            if collision_geometry
            else getattr(robot_module, "scene", None)
        )
        if scene is None:
            geometry_name = "collision" if collision_geometry else "visual"
            raise ValueError(f"robot module has no {geometry_name} mesh scene")

        hand_link_set = set(hand_links)
        link_meshes: Dict[str, List[trimesh.Trimesh]] = {}
        for node_name in scene.graph.nodes_geometry:
            link_name = scene.graph.transforms.parents.get(node_name)
            while link_name is not None and link_name not in robot_module.urdf.link_map:
                link_name = scene.graph.transforms.parents.get(link_name)
            if link_name is None:
                raise ValueError(
                    f"cannot resolve parent link for mesh node {node_name}"
                )
            local_transform, geometry_key = scene.graph.get(node_name, link_name)
            mesh = scene.geometry[geometry_key].copy()
            mesh.apply_transform(local_transform)
            link_meshes.setdefault(link_name, []).append(mesh)

        links = []
        for link_name, meshes in link_meshes.items():
            mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
            links.append(
                LinkGeometry(
                    name=link_name,
                    mesh=mesh,
                    group="hand" if link_name in hand_link_set else "robot",
                )
            )

        joint_names = list(robot_module.get_joint_names())
        limit_map = robot_module.get_joint_limits()
        lower: Optional[List[float]]
        upper: Optional[List[float]]
        if all(name in limit_map for name in joint_names):
            lower = [limit_map[name][0] for name in joint_names]
            upper = [limit_map[name][1] for name in joint_names]
        else:
            lower = upper = None
        disabled_pairs = list(disabled_self_collision_pairs)
        if disable_adjacent_collisions:
            disabled_pairs.extend(
                (joint.parent, joint.child)
                for joint in robot_module.urdf.joint_map.values()
                if joint.parent in link_meshes and joint.child in link_meshes
            )
        return cls(
            links=links,
            pose_provider=robot_module.forward_kinematics,
            disabled_self_collision_pairs=disabled_pairs,
            joint_lower=lower,
            joint_upper=upper,
            velocity_limits=velocity_limits,
        )

    def poses(self, parameters: np.ndarray) -> Dict[str, np.ndarray]:
        raw_poses = self.pose_provider(np.asarray(parameters, dtype=float))
        missing = set(self.links).difference(raw_poses)
        if missing:
            raise ValueError(f"pose provider omitted links: {sorted(missing)}")
        return {
            name: _validate_transform(pose, f"pose[{name}]")
            for name, pose in raw_poses.items()
        }

    def world_meshes(
        self, poses: Mapping[str, np.ndarray]
    ) -> Dict[str, trimesh.Trimesh]:
        result = {}
        for name, link in self.links.items():
            mesh = link.mesh.copy()
            mesh.apply_transform(poses[name] @ link.local_transform)
            result[name] = mesh
        return result


@dataclass(frozen=True)
class GraspPhase:
    """A closed-hand interval in which the object should follow a link."""

    start_time: float
    end_time: float
    reference_link: str
    contact_links: Tuple[str, ...]
    min_contact_links: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_time) or not np.isfinite(self.end_time):
            raise ValueError("grasp phase bounds must be finite")
        if self.end_time < self.start_time:
            raise ValueError("grasp phase end_time must not precede start_time")
        if not self.reference_link:
            raise ValueError("reference_link must not be empty")
        contacts = tuple(self.contact_links)
        if len(contacts) != len(set(contacts)):
            raise ValueError("contact_links must be unique")
        if self.min_contact_links < 0 or self.min_contact_links > len(contacts):
            raise ValueError("min_contact_links is inconsistent with contact_links")
        object.__setattr__(self, "contact_links", contacts)

    def contains(self, time: float) -> bool:
        return self.start_time - 1e-12 <= time <= self.end_time + 1e-12


@dataclass(frozen=True)
class GraspValidationConfig:
    """Thresholds are in metres, radians, and seconds."""

    sample_dt: float = 0.02
    contact_distance: float = 0.003
    forbidden_clearance: float = 0.001
    allowed_penetration: float = 0.002
    collision_penetration_tolerance: float = 0.0
    self_collision_clearance: float = 0.0
    max_relative_translation_error: float = 0.01
    max_relative_rotation_error: float = np.deg2rad(10.0)
    max_geometry_samples: int = 1024
    check_self_collision: bool = True
    require_matching_time_range: bool = True

    def __post_init__(self) -> None:
        numeric_nonnegative = (
            "contact_distance",
            "forbidden_clearance",
            "allowed_penetration",
            "collision_penetration_tolerance",
            "self_collision_clearance",
            "max_relative_translation_error",
            "max_relative_rotation_error",
        )
        if not np.isfinite(self.sample_dt) or self.sample_dt <= 0.0:
            raise ValueError("sample_dt must be positive")
        for name in numeric_nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_geometry_samples < 8:
            raise ValueError("max_geometry_samples must be at least 8")


class ViolationCode(str, Enum):
    JOINT_LIMIT = "joint_limit"
    JOINT_VELOCITY = "joint_velocity"
    SELF_COLLISION = "self_collision"
    FORBIDDEN_OBJECT_COLLISION = "forbidden_object_collision"
    EXCESSIVE_PENETRATION = "excessive_penetration"
    INSUFFICIENT_CONTACT = "insufficient_contact"
    RELATIVE_POSE_DRIFT = "relative_pose_drift"


@dataclass(frozen=True)
class Violation:
    time: float
    code: ViolationCode
    message: str
    links: Tuple[str, ...] = ()
    measured: Optional[float] = None
    limit: Optional[float] = None


@dataclass
class ValidationReport:
    sample_times: np.ndarray
    violations: List[Violation]
    contacts_by_time: Dict[float, Tuple[str, ...]]
    minimum_object_distance: Dict[str, float]
    maximum_relative_translation_error: float = 0.0
    maximum_relative_rotation_error: float = 0.0
    limitations: Tuple[str, ...] = (
        "Geometric sampling is conservative but not a continuous-time collision proof.",
        "Force closure, friction, torque limits, deformation, and dynamics "
        "are not evaluated.",
    )

    @property
    def valid(self) -> bool:
        return not self.violations

    def violations_of(self, code: ViolationCode) -> List[Violation]:
        return [violation for violation in self.violations if violation.code == code]


@dataclass(frozen=True)
class _DistanceResult:
    surface_distance: float
    penetration: float


class _GeometryDistance:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples

    def _samples(self, mesh: trimesh.Trimesh) -> np.ndarray:
        vertices = np.asarray(mesh.vertices)
        centers = np.asarray(mesh.triangles_center)
        interior_candidates = np.asarray(
            [mesh.centroid, mesh.center_mass], dtype=float
        ).reshape(-1, 3)
        points = np.vstack((vertices, centers, interior_candidates))
        if points.shape[0] > self.max_samples:
            indices = np.linspace(
                0, points.shape[0] - 1, self.max_samples, dtype=int
            )
            points = points[indices]
        return points

    @staticmethod
    def _aabb_separation(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> float:
        a_min, a_max = mesh_a.bounds
        b_min, b_max = mesh_b.bounds
        gap = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
        return float(np.linalg.norm(gap))

    def between(
        self,
        mesh_a: trimesh.Trimesh,
        mesh_b: trimesh.Trimesh,
        query_distance: float,
    ) -> _DistanceResult:
        aabb_separation = self._aabb_separation(mesh_a, mesh_b)
        if aabb_separation > query_distance:
            return _DistanceResult(aabb_separation, 0.0)

        samples_a = self._samples(mesh_a)
        samples_b = self._samples(mesh_b)
        _, distances_a, _ = trimesh.proximity.closest_point(mesh_b, samples_a)
        _, distances_b, _ = trimesh.proximity.closest_point(mesh_a, samples_b)
        surface_distance = float(min(np.min(distances_a), np.min(distances_b)))

        signed_a = trimesh.proximity.signed_distance(mesh_b, samples_a)
        signed_b = trimesh.proximity.signed_distance(mesh_a, samples_b)
        penetration = float(max(0.0, np.max(signed_a), np.max(signed_b)))
        return _DistanceResult(surface_distance, penetration)


class GraspSequenceValidator:
    """Validate sampled robot/hand and object trajectories."""

    def __init__(
        self,
        robot: ArticulatedMeshModel,
        object_mesh: trimesh.Trimesh,
        config: Optional[GraspValidationConfig] = None,
    ):
        if not isinstance(object_mesh, trimesh.Trimesh):
            raise TypeError("object_mesh must be trimesh.Trimesh")
        if object_mesh.vertices.size == 0 or object_mesh.faces.size == 0:
            raise ValueError("object_mesh must not be empty")
        self.robot = robot
        self.object_mesh = object_mesh.copy()
        self.config = config or GraspValidationConfig()
        self._distance = _GeometryDistance(self.config.max_geometry_samples)

    def validate(
        self,
        robot_trajectory: Union[SampledTrajectory, CallableTrajectory],
        object_trajectory: PoseTrajectory,
        grasp_phases: Sequence[GraspPhase],
        sample_times: Optional[ArrayLike] = None,
    ) -> ValidationReport:
        phases = tuple(sorted(grasp_phases, key=lambda phase: phase.start_time))
        self._validate_domains(robot_trajectory, object_trajectory, phases)
        times = self._sample_times(
            robot_trajectory, object_trajectory, phases, sample_times
        )
        violations: List[Violation] = []
        contacts: Dict[float, Tuple[str, ...]] = {}
        min_distances = {name: np.inf for name in self.robot.links}
        relative_baselines = self._relative_baselines(
            robot_trajectory, object_trajectory, phases
        )

        previous_time: Optional[float] = None
        previous_parameters: Optional[np.ndarray] = None
        max_translation_error = 0.0
        max_rotation_error = 0.0

        for time in times:
            parameters = robot_trajectory.at(float(time))
            poses = self.robot.poses(parameters)
            object_transform = object_trajectory.at(float(time))
            world_meshes = self.robot.world_meshes(poses)
            world_object = self.object_mesh.copy()
            world_object.apply_transform(object_transform)

            violations.extend(self._check_joint_limits(float(time), parameters))
            if previous_parameters is not None:
                violations.extend(
                    self._check_joint_velocity(
                        float(time),
                        parameters,
                        previous_time,
                        previous_parameters,
                    )
                )
            if self.config.check_self_collision:
                violations.extend(
                    self._check_self_collision(float(time), world_meshes)
                )

            phase = self._active_phase(float(time), phases)
            frame_contacts, distance_updates, object_violations = (
                self._check_object_interactions(
                    float(time), world_meshes, world_object, phase
                )
            )
            contacts[float(time)] = tuple(sorted(frame_contacts))
            for name, distance in distance_updates.items():
                min_distances[name] = min(min_distances[name], distance)
            violations.extend(object_violations)

            if phase is not None:
                if len(frame_contacts) < phase.min_contact_links:
                    violations.append(
                        Violation(
                            time=float(time),
                            code=ViolationCode.INSUFFICIENT_CONTACT,
                            message=(
                                f"only {len(frame_contacts)} of "
                                f"{phase.min_contact_links} required contact links "
                                "touch the object"
                            ),
                            links=tuple(sorted(frame_contacts)),
                            measured=float(len(frame_contacts)),
                            limit=float(phase.min_contact_links),
                        )
                    )
                translation_error, rotation_error = self._relative_pose_error(
                    poses[phase.reference_link],
                    object_transform,
                    relative_baselines[id(phase)],
                )
                max_translation_error = max(
                    max_translation_error, translation_error
                )
                max_rotation_error = max(max_rotation_error, rotation_error)
                if (
                    translation_error
                    > self.config.max_relative_translation_error + 1e-12
                    or rotation_error
                    > self.config.max_relative_rotation_error + 1e-12
                ):
                    violations.append(
                        Violation(
                            time=float(time),
                            code=ViolationCode.RELATIVE_POSE_DRIFT,
                            message=(
                                "object pose drifted relative to grasp reference "
                                f"link {phase.reference_link}: "
                                f"{translation_error:.6g} m, "
                                f"{np.rad2deg(rotation_error):.6g} deg"
                            ),
                            links=(phase.reference_link,),
                            measured=max(
                                translation_error
                                / max(
                                    self.config.max_relative_translation_error,
                                    1e-15,
                                ),
                                rotation_error
                                / max(
                                    self.config.max_relative_rotation_error,
                                    1e-15,
                                ),
                            ),
                            limit=1.0,
                        )
                    )
            previous_time = float(time)
            previous_parameters = parameters

        return ValidationReport(
            sample_times=times,
            violations=violations,
            contacts_by_time=contacts,
            minimum_object_distance=min_distances,
            maximum_relative_translation_error=max_translation_error,
            maximum_relative_rotation_error=max_rotation_error,
        )

    def _validate_domains(
        self,
        robot_trajectory: Union[SampledTrajectory, CallableTrajectory],
        object_trajectory: PoseTrajectory,
        phases: Sequence[GraspPhase],
    ) -> None:
        if self.config.require_matching_time_range and (
            not np.isclose(
                robot_trajectory.start_time, object_trajectory.start_time, atol=1e-9
            )
            or not np.isclose(
                robot_trajectory.end_time, object_trajectory.end_time, atol=1e-9
            )
        ):
            raise ValueError("robot and object trajectory time ranges must match")
        start = max(robot_trajectory.start_time, object_trajectory.start_time)
        end = min(robot_trajectory.end_time, object_trajectory.end_time)
        if end < start:
            raise ValueError("robot and object trajectories do not overlap")
        seen_end = -np.inf
        for phase in phases:
            if phase.start_time < start - 1e-12 or phase.end_time > end + 1e-12:
                raise ValueError("grasp phase is outside the trajectory time range")
            if phase.start_time < seen_end - 1e-12:
                raise ValueError("grasp phases must not overlap")
            unknown_contacts = set(phase.contact_links).difference(self.robot.links)
            if unknown_contacts:
                raise ValueError(
                    f"unknown contact links: {sorted(unknown_contacts)}"
                )
            non_hand = [
                name
                for name in phase.contact_links
                if self.robot.links[name].group != "hand"
            ]
            if non_hand:
                raise ValueError(
                    f"object contact links must belong to the hand: {non_hand}"
                )
            seen_end = phase.end_time

    def _sample_times(
        self,
        robot_trajectory: Union[SampledTrajectory, CallableTrajectory],
        object_trajectory: PoseTrajectory,
        phases: Sequence[GraspPhase],
        sample_times: Optional[ArrayLike],
    ) -> np.ndarray:
        start = max(robot_trajectory.start_time, object_trajectory.start_time)
        end = min(robot_trajectory.end_time, object_trajectory.end_time)
        if sample_times is None:
            count = max(1, int(np.ceil((end - start) / self.config.sample_dt)))
            times = np.linspace(start, end, count + 1)
        else:
            times = _validate_times(sample_times)
            if times[0] < start - 1e-12 or times[-1] > end + 1e-12:
                raise ValueError("sample_times are outside the trajectory overlap")
        boundaries = [
            value
            for phase in phases
            for value in (phase.start_time, phase.end_time)
        ]
        return np.unique(np.concatenate((times, np.asarray(boundaries, dtype=float))))

    def _relative_baselines(
        self,
        robot_trajectory: Union[SampledTrajectory, CallableTrajectory],
        object_trajectory: PoseTrajectory,
        phases: Sequence[GraspPhase],
    ) -> Dict[int, np.ndarray]:
        result = {}
        for phase in phases:
            poses = self.robot.poses(robot_trajectory.at(phase.start_time))
            if phase.reference_link not in poses:
                raise ValueError(
                    f"pose provider omitted reference link {phase.reference_link}"
                )
            result[id(phase)] = (
                np.linalg.inv(poses[phase.reference_link])
                @ object_trajectory.at(phase.start_time)
            )
        return result

    @staticmethod
    def _active_phase(
        time: float, phases: Sequence[GraspPhase]
    ) -> Optional[GraspPhase]:
        # At a shared boundary, the newly starting phase owns the sample.
        for phase in reversed(phases):
            if phase.contains(time):
                return phase
        return None

    def _check_joint_limits(
        self, time: float, parameters: np.ndarray
    ) -> List[Violation]:
        joint_lower = self.robot.joint_lower
        joint_upper = self.robot.joint_upper
        if joint_lower is None or joint_upper is None:
            return []
        if parameters.shape != joint_lower.shape:
            raise ValueError(
                "trajectory parameter dimension does not match joint limit dimension"
            )
        outside = np.flatnonzero(
            (parameters < joint_lower - 1e-12)
            | (parameters > joint_upper + 1e-12)
        )
        return [
            Violation(
                time=time,
                code=ViolationCode.JOINT_LIMIT,
                message=(
                    f"parameter {index}={parameters[index]:.6g} is outside "
                    f"[{joint_lower[index]:.6g}, "
                    f"{joint_upper[index]:.6g}]"
                ),
                links=(str(index),),
                measured=float(parameters[index]),
            )
            for index in outside
        ]

    def _check_joint_velocity(
        self,
        time: float,
        parameters: np.ndarray,
        previous_time: Optional[float],
        previous_parameters: np.ndarray,
    ) -> List[Violation]:
        if self.robot.velocity_limits is None or previous_time is None:
            return []
        if parameters.shape != self.robot.velocity_limits.shape:
            raise ValueError(
                "trajectory parameter dimension does not match velocity limit dimension"
            )
        velocity = np.abs(parameters - previous_parameters) / (time - previous_time)
        exceeded = np.flatnonzero(velocity > self.robot.velocity_limits + 1e-12)
        return [
            Violation(
                time=time,
                code=ViolationCode.JOINT_VELOCITY,
                message=(
                    f"parameter {index} velocity {velocity[index]:.6g} exceeds "
                    f"{self.robot.velocity_limits[index]:.6g}"
                ),
                links=(str(index),),
                measured=float(velocity[index]),
                limit=float(self.robot.velocity_limits[index]),
            )
            for index in exceeded
        ]

    def _check_self_collision(
        self, time: float, meshes: Mapping[str, trimesh.Trimesh]
    ) -> List[Violation]:
        violations = []
        query_distance = max(
            self.config.self_collision_clearance,
            self.config.collision_penetration_tolerance,
        )
        for name_a, name_b in combinations(meshes, 2):
            if frozenset((name_a, name_b)) in self.robot.disabled_self_collision_pairs:
                continue
            distance = self._distance.between(
                meshes[name_a], meshes[name_b], query_distance
            )
            if (
                distance.penetration
                > self.config.collision_penetration_tolerance + 1e-12
                or distance.surface_distance
                < self.config.self_collision_clearance - 1e-12
            ):
                violations.append(
                    Violation(
                        time=time,
                        code=ViolationCode.SELF_COLLISION,
                        message=(
                            f"{name_a} and {name_b} violate self-collision "
                            f"clearance; distance={distance.surface_distance:.6g}, "
                            f"penetration={distance.penetration:.6g}"
                        ),
                        links=(name_a, name_b),
                        measured=distance.penetration,
                        limit=self.config.collision_penetration_tolerance,
                    )
                )
        return violations

    def _check_object_interactions(
        self,
        time: float,
        meshes: Mapping[str, trimesh.Trimesh],
        object_mesh: trimesh.Trimesh,
        phase: Optional[GraspPhase],
    ) -> Tuple[Set[str], Dict[str, float], List[Violation]]:
        contacts: Set[str] = set()
        distances: Dict[str, float] = {}
        violations: List[Violation] = []
        contact_links = set(phase.contact_links) if phase is not None else set()
        query_distance = max(
            self.config.contact_distance,
            self.config.forbidden_clearance,
            self.config.allowed_penetration,
            self.config.collision_penetration_tolerance,
        )
        for name, mesh in meshes.items():
            distance = self._distance.between(mesh, object_mesh, query_distance)
            distances[name] = distance.surface_distance
            allowed_contact = name in contact_links
            if allowed_contact:
                if distance.surface_distance <= self.config.contact_distance + 1e-12:
                    contacts.add(name)
                if distance.penetration > self.config.allowed_penetration + 1e-12:
                    violations.append(
                        Violation(
                            time=time,
                            code=ViolationCode.EXCESSIVE_PENETRATION,
                            message=(
                                f"allowed contact {name} penetrates object by "
                                f"{distance.penetration:.6g} m"
                            ),
                            links=(name,),
                            measured=distance.penetration,
                            limit=self.config.allowed_penetration,
                        )
                    )
            elif (
                distance.penetration
                > self.config.collision_penetration_tolerance + 1e-12
                or distance.surface_distance
                < self.config.forbidden_clearance - 1e-12
            ):
                violations.append(
                    Violation(
                        time=time,
                        code=ViolationCode.FORBIDDEN_OBJECT_COLLISION,
                        message=(
                            f"object is too close to forbidden link {name}; "
                            f"distance={distance.surface_distance:.6g}, "
                            f"penetration={distance.penetration:.6g}"
                        ),
                        links=(name,),
                        measured=distance.surface_distance,
                        limit=self.config.forbidden_clearance,
                    )
                )
        return contacts, distances, violations

    @staticmethod
    def _relative_pose_error(
        reference_pose: np.ndarray,
        object_pose: np.ndarray,
        baseline_reference_T_object: np.ndarray,
    ) -> Tuple[float, float]:
        current = np.linalg.inv(reference_pose) @ object_pose
        translation_error = float(
            np.linalg.norm(current[:3, 3] - baseline_reference_T_object[:3, 3])
        )
        rotation_error = _rotation_error(
            baseline_reference_T_object[:3, :3], current[:3, :3]
        )
        return translation_error, rotation_error
