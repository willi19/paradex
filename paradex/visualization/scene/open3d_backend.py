"""Headless open3d backend for the shared :class:`Scene` API.

``Open3DScene`` mirrors ``ViserScene``'s call surface but renders offscreen to
images / mp4 — no browser, no client. It exists so the ~20 hand-rolled
``OffscreenRenderer`` figure/video scripts can share one scene + color API
instead of duplicating material/camera/convert boilerplate.

Robot/object state lives in a registry; each rendered frame rebuilds the o3d
scene from that state (o3d geometry is static, so re-adding is the simplest
correct model).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import open3d as o3d
import trimesh

from paradex.visualization.robot import RobotModule
from paradex.visualization.scene.base import Color, Pose, Scene, Timeline, normalize_color

_DEFAULT_ROBOT = (0.7, 0.7, 0.9, 1.0)
_DEFAULT_OBJECT = (0.8, 0.8, 0.8, 1.0)


def _material(color) -> o3d.visualization.rendering.MaterialRecord:
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = list(color)
    return mat


def _to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o = o3d.geometry.TriangleMesh()
    o.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o.compute_vertex_normals()
    return o


class _RobotEntry:
    def __init__(self, urdf_path: str, pose: np.ndarray, color):
        self.robot = RobotModule(urdf_path)
        self.qpos = np.zeros(self.robot.get_num_joints())
        self.pose = pose
        self.color = list(color)
        self.hidden = False


class _ObjectEntry:
    def __init__(self, mesh: trimesh.Trimesh, pose: np.ndarray, color):
        self.mesh = mesh
        self.pose = pose
        self.color = list(color)
        self.hidden = False


class Open3DScene(Scene):
    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0,
                 background=(1.0, 1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self._default_fps = fps
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.renderer.scene.set_background(list(background))
        self.renderer.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.MED_SHADOWS,
            np.array([0.0, -1.0, -1.0]),
        )
        self._camera = None  # ("lookat", eye, center, up, fov) | ("intrinsic", K, E)

    # ------------------------------------------------------------------ #
    # Scene construction
    # ------------------------------------------------------------------ #
    def add_robot(self, name: str, urdf_path: str, pose: Optional[Pose] = None,
                  color: Color = None) -> _RobotEntry:
        rgb, opacity = normalize_color(color)
        c = (*rgb, opacity if opacity is not None else 1.0) if rgb is not None else _DEFAULT_ROBOT
        entry = _RobotEntry(urdf_path, np.eye(4) if pose is None else np.asarray(pose, float), c)
        self._robots[name] = entry
        return entry

    def add_object(self, name: str, mesh: trimesh.Trimesh, pose: Optional[Pose] = None,
                   color: Color = None) -> _ObjectEntry:
        rgb, opacity = normalize_color(color)
        c = (*rgb, opacity if opacity is not None else 1.0) if rgb is not None else _DEFAULT_OBJECT
        entry = _ObjectEntry(mesh, np.eye(4) if pose is None else np.asarray(pose, float), c)
        self._objects[name] = entry
        return entry

    add_trimesh = add_object

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def set_qpos(self, name: str, qpos: np.ndarray) -> None:
        self._robots[name].qpos = np.asarray(qpos)

    def set_pose(self, name: str, pose: Pose) -> None:
        T = np.asarray(pose, dtype=float)
        if name in self._robots:
            self._robots[name].pose = T
        elif name in self._objects:
            self._objects[name].pose = T
        else:
            raise KeyError(f"'{name}' is not a known robot or object")

    def change_color(self, name: str, color: Color,
                     links: Optional[Sequence[str]] = None) -> None:
        rgb, opacity = normalize_color(color)
        if rgb is None:
            return
        entry = self._robots.get(name) or self._objects.get(name)
        if entry is None:
            raise KeyError(f"'{name}' is not a known robot or object")
        a = opacity if opacity is not None else entry.color[3]
        entry.color = [rgb[0], rgb[1], rgb[2], a]

    def set_opacity(self, name: str, opacity: float) -> None:
        entry = self._robots.get(name) or self._objects.get(name)
        if entry is None:
            raise KeyError(f"'{name}' is not a known robot or object")
        entry.color[3] = float(opacity)

    def set_visible(self, name: str, visible: bool) -> None:
        entry = self._robots.get(name) or self._objects.get(name)
        if entry is None:
            raise KeyError(f"'{name}' is not a known robot or object")
        entry.hidden = not visible

    def remove(self, name: str) -> None:
        self._robots.pop(name, None)
        self._objects.pop(name, None)

    def clear(self) -> None:
        self._robots.clear()
        self._objects.clear()
        self._timeline = None
        self._frame = 0

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #
    def set_camera(self, eye=(1.5, 1.5, 1.5), center=(0.0, 0.0, 0.15),
                   up=(0.0, 0.0, 1.0), fov: float = 60.0) -> None:
        self._camera = ("lookat", np.asarray(eye, float), np.asarray(center, float),
                        np.asarray(up, float), fov)

    def set_intrinsic(self, K: np.ndarray, extrinsic: np.ndarray) -> None:
        """Pin the camera from a 3x3 intrinsic K and 4x4 world-from-cam (or
        cam-from-world) extrinsic, matching the raw-o3d scripts' setup_camera."""
        self._camera = ("intrinsic", np.asarray(K, float), np.asarray(extrinsic, float))

    def _apply_camera(self) -> None:
        if self._camera is None:
            self.set_camera()
        cam = self._camera
        if cam[0] == "lookat":
            _, eye, center, up, fov = cam
            self.renderer.setup_camera(fov, center.tolist(), eye.tolist(), up.tolist())
        else:
            _, K, E = cam
            self.renderer.setup_camera(K, E, self.width, self.height)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        scene = self.renderer.scene
        scene.clear_geometry()
        for name, entry in self._robots.items():
            if entry.hidden:
                continue
            entry.robot.update_cfg(entry.qpos)
            mesh = entry.robot.get_robot_mesh(collision_geometry=False)
            if mesh is None:
                continue
            mesh = mesh.copy()
            mesh.apply_transform(entry.pose)
            scene.add_geometry(f"robot::{name}", _to_o3d(mesh), _material(entry.color))
        for name, entry in self._objects.items():
            if entry.hidden:
                continue
            mesh = entry.mesh.copy()
            mesh.apply_transform(entry.pose)
            scene.add_geometry(f"object::{name}", _to_o3d(mesh), _material(entry.color))

    def render_frame(self, t: Optional[int] = None) -> np.ndarray:
        """Render the current (or frame ``t``) scene state to an RGB uint8 array."""
        if t is not None:
            self.goto_frame(t)
        self._rebuild()
        self._apply_camera()
        return np.asarray(self.renderer.render_to_image())

    def export_video(self, path: str, fps: Optional[float] = None) -> str:
        import imageio.v2 as imageio

        if self.num_frames == 0:
            raise RuntimeError("no timeline installed; call animate() first")
        fps = fps or self.fps or self._default_fps
        writer = imageio.get_writer(path, fps=int(fps))
        try:
            for t in range(self.num_frames):
                writer.append_data(self.render_frame(t))
        finally:
            writer.close()
        return path
