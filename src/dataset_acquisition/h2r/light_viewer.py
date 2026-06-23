"""Lightweight ViserViewer drop-in for h2r capture_object6d.

Same API surface as paradex.visualization.visualizer.viser.ViserViewer (robot_dict,
obj_dict, add_robot/add_object/add_traj/add_floor/add_camera/add_player/update/
update_scene/render_*), but robots are loaded with viser.extras.ViserUrdf and
animated via `update_cfg(qpos)` instead of re-uploading per-link meshes through
add_mesh_simple. Cuts startup and per-frame work on heavy URDFs.

Intentional differences from the original:
  - No add_contact_module (it depended on the custom RobotModule).
  - Robots are wrapped in `_LightRobot` so `.urdf.get_cfg()` and
    `_visual_root_frame` continue to work for add_traj / base re-posing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import trimesh
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf


def _wxyz(mat: np.ndarray):
    qx, qy, qz, qw = R.from_matrix(np.asarray(mat)[:3, :3]).as_quat()
    return float(qw), float(qx), float(qy), float(qz)


class _LightRobot:
    """Thin wrapper around viser.extras.ViserUrdf so callers can use the same
    API as the original ViserRobotModule (update_cfg, urdf.get_cfg,
    _visual_root_frame)."""

    def __init__(self, server: viser.ViserServer, name: str, urdf_path: str,
                 base_pose: Optional[np.ndarray] = None):
        self._server = server
        self._root_name = f"/robot/{name}"
        if base_pose is None:
            base_pose = np.eye(4)
        base_pose = np.asarray(base_pose, dtype=np.float64)
        # Parent frame: moving this re-poses the whole URDF.
        self._visual_root_frame = server.scene.add_frame(
            self._root_name,
            position=base_pose[:3, 3],
            wxyz=_wxyz(base_pose),
            show_axes=False,
        )
        # yourdfpy.URDF kept for joint names / cfg defaults (no mesh loaded twice;
        # ViserUrdf builds its own scene internally from the same path).
        self.urdf = yourdfpy.URDF.load(urdf_path)
        self._viser_urdf = ViserUrdf(
            server,
            urdf_or_path=Path(urdf_path),
            root_node_name=self._root_name,
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._cfg = np.array(self.urdf.cfg, dtype=np.float64).copy()

    # Mirror the original API used by add_traj.
    @property
    def joint_names(self):
        return list(self._viser_urdf.get_actuated_joint_names())

    def get_num_joints(self) -> int:
        return len(self.joint_names)

    def update_cfg(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        self._cfg = qpos
        self._viser_urdf.update_cfg(qpos)

    # Original code reads `.urdf.get_cfg()` to tile defaults in add_traj.
    # yourdfpy's URDF doesn't have get_cfg; expose a shim returning last cfg.
    class _UrdfShim:
        def __init__(self, parent):
            self._parent = parent

        def get_cfg(self):
            return np.asarray(self._parent._cfg, dtype=np.float64).copy()

    # Keep .urdf accessible as yourdfpy.URDF AND expose .get_cfg() via attr name
    # used by the original ViserViewer.add_traj.
    def __getattr__(self, item):
        # Only triggered when normal lookup fails.
        if item == "urdf_get_cfg":
            return self._cfg.copy()
        raise AttributeError(item)


class LightViserViewer:
    """Drop-in lightweight replacement for paradex ViserViewer."""

    def __init__(self, up_direction=np.array([0, 0, 1]), scene_title=None):
        self.frame_nodes: dict[str, viser.FrameHandle] = {}
        self.up_direction = up_direction
        self.view_state_dir = Path(".viser_views")
        self.view_state_path = self.view_state_dir / "viser_current_view.json"
        self.robot_dict: Dict[str, _LightRobot] = {}
        self.obj_dict: Dict[str, dict] = {}
        self.camera_dict: Dict[str, dict] = {}

        self.traj_list = []
        self.num_frames = 0

        self._load_server()
        self._add_lights()
        self._add_player()

        self.scene_title = scene_title

    # ------------------------------------------------------------------ server
    def _load_server(self):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(dark_mode=True)
        self.server.scene.set_up_direction(self.up_direction)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            near_slider = client.gui.add_slider(
                "Near", min=0.01, max=10.0, step=0.001, initial_value=client.camera.near
            )
            far_slider = client.gui.add_slider(
                "Far", min=1, max=1000.0, step=0.001, initial_value=client.camera.far
            )

            @near_slider.on_update
            def _(_):
                client.camera.near = near_slider.value

            @far_slider.on_update
            def _(_):
                client.camera.far = far_slider.value

            self.load_saved_view(client, verbose=False)

    def _get_active_client(self, event=None) -> Optional[viser.ClientHandle]:
        c = getattr(event, "client", None)
        if c is not None:
            return c
        clients = list(self.server.get_clients().values())
        return clients[0] if clients else None

    # ------------------------------------------------------------------ view io
    def _serialize_camera_state(self, camera: viser.CameraHandle) -> dict:
        return {
            "position": np.asarray(camera.position, dtype=float).tolist(),
            "look_at": np.asarray(camera.look_at, dtype=float).tolist(),
            "up_direction": np.asarray(camera.up_direction, dtype=float).tolist(),
            "wxyz": np.asarray(camera.wxyz, dtype=float).tolist(),
            "fov": float(camera.fov),
            "near": float(camera.near),
            "far": float(camera.far),
        }

    def _set_view_state_path_for_object(self, object_name: str) -> None:
        safe = Path(str(object_name)).name.strip().replace(" ", "_") or "viser_current"
        self.view_state_path = self.view_state_dir / f"{safe}_view.json"

    def save_current_view(self, event=None) -> bool:
        client = self._get_active_client(event)
        if client is None:
            print("No clients connected; cannot save current view.")
            return False
        payload = self._serialize_camera_state(client.camera)
        self.view_state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.view_state_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved current view to: {self.view_state_path}")
        return True

    def load_saved_view(self, client=None, *, verbose: bool = True) -> bool:
        target = client or self._get_active_client()
        if target is None or not self.view_state_path.exists():
            if verbose:
                print(f"No view at: {self.view_state_path}")
            return False
        with self.view_state_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        cam = target.camera
        for k in ("near", "far", "fov"):
            if k in payload:
                setattr(cam, k, float(payload[k]))
        for k in ("position", "look_at", "up_direction"):
            if k in payload:
                setattr(cam, k, np.asarray(payload[k], dtype=float))
        if verbose:
            print(f"Loaded view from: {self.view_state_path}")
        return True

    # ------------------------------------------------------------------ scene
    def add_robot(self, name: str, urdf_path: str, pose=None,
                  include_arm_meshes: bool = True):
        # include_arm_meshes kept for signature compatibility (ignored: ViserUrdf
        # always shows all links; filter at URDF level if needed).
        del include_arm_meshes
        robot = _LightRobot(self.server, name, urdf_path, base_pose=pose)
        self.robot_dict[name] = robot
        return robot

    def add_object(self, name: str, obj: trimesh.Trimesh, obj_T,
                   opacity: float = 1.0):
        obj_T = np.asarray(obj_T, dtype=np.float64)
        frame_handle = self.server.scene.add_frame(
            f"/objects/{name}_frame",
            position=obj_T[:3, 3],
            wxyz=_wxyz(obj_T),
            show_axes=True,
            axes_length=0.05,
            axes_radius=0.002,
        )
        obj_name = f"/objects/{name}_frame/{name}"
        self._set_view_state_path_for_object(name)

        if opacity < 0.999:
            color = np.array([200, 200, 200], dtype=np.uint8)
            vc = getattr(obj.visual, "vertex_colors", None)
            if vc is not None:
                vc_arr = np.asarray(vc)
                if vc_arr.ndim == 2 and vc_arr.shape[0] > 0 and vc_arr.shape[1] >= 3:
                    color = vc_arr[:, :3].mean(axis=0).astype(np.uint8)
            mesh_handle = self.server.scene.add_mesh_simple(
                name=obj_name, vertices=obj.vertices, faces=obj.faces, color=color,
            )
            mesh_handle.opacity = float(np.clip(opacity, 0.0, 1.0))
        else:
            mesh_handle = self.server.scene.add_mesh_trimesh(name=obj_name, mesh=obj)

        self.obj_dict[name] = {
            "mesh": obj, "transform": obj_T,
            "frame": frame_handle, "handle": mesh_handle,
        }
        self.frame_nodes[name] = frame_handle

    def add_traj(self, name: str, robot_traj: Dict, obj_traj: Dict = {}):
        if len(robot_traj) == 0:
            return
        traj_len = robot_traj[next(iter(robot_traj))].shape[0]
        new = {"robot": {}, "object": {}}
        for rname, robot in self.robot_dict.items():
            if rname in robot_traj:
                new["robot"][rname] = robot_traj[rname]
            else:
                new["robot"][rname] = np.tile(robot._cfg, (traj_len, 1))
        for oname, obj in self.obj_dict.items():
            if oname in obj_traj:
                new["object"][oname] = obj_traj[oname]
                obj["transform"] = obj_traj[oname][-1]
            else:
                new["object"][oname] = np.tile(obj["transform"][None, :, :], (traj_len, 1, 1))
        self.traj_list.append((name, new, traj_len))
        self.num_frames += traj_len
        self.gui_timestep.max = self.num_frames - 1
        print(f"[traj] added {name}: {traj_len} frames (total {self.num_frames})")

    def add_floor(self, height: float = 0.0):
        self.floor_size_val = 1.0
        size = self.floor_size_val
        self.grid_handles = self.server.scene.add_grid(
            name="floor/grid",
            width=size * 2, height=size * 2, plane="xy",
            position=(0.0, 0.0, float(height)),
            cell_size=0.1, section_color=(0.5, 0.5, 0),
        )

    def update_floor(self):
        if hasattr(self, "grid_handles"):
            handles = self.grid_handles if isinstance(self.grid_handles, (list, tuple)) else [self.grid_handles]
            for h in handles:
                h.visible = self.grid_visible.value

    # ---------------------------------------------------------------- playback
    def update_scene(self, timestep: int):
        cumulative = 0
        current = None
        local = timestep
        for tname, tdata, tlen in self.traj_list:
            if timestep < cumulative + tlen:
                current = tdata
                local = timestep - cumulative
                break
            cumulative += tlen
        if current is None:
            return

        with self.server.atomic():
            for rname, robot in self.robot_dict.items():
                if rname in current["robot"]:
                    robot.update_cfg(current["robot"][rname][local])
            for oname, obj in self.obj_dict.items():
                if oname in current["object"]:
                    T = current["object"][oname][local]
                    fh = obj["frame"]
                    fh.wxyz = _wxyz(T)
                    fh.position = T[:3, 3]
        self.server.flush()
        if self.render_png.value:
            self.render_current_frame(timestep)

    def update(self):
        if self.gui_playing.value and len(self.traj_list) > 0:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames
        time.sleep(1.0 / max(1e-3, self.gui_framerate.value))

    def start_viewer(self):
        try:
            while True:
                self.update()
        except KeyboardInterrupt:
            pass

    # ---------------------------------------------------------------- gui
    def _add_player(self):
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Timestep", min=0, max=max(self.num_frames - 1, 1), step=1, initial_value=0,
            )
            self.gui_next_frame = self.server.gui.add_button("Next Frame", disabled=True)
            self.gui_prev_frame = self.server.gui.add_button("Prev Frame", disabled=True)
            self.gui_playing = self.server.gui.add_checkbox("Playing", True)
            self.gui_pause = self.server.gui.add_button("Pause")
            self.gui_resume = self.server.gui.add_button("Resume", disabled=True)
            self.render_png = self.server.gui.add_checkbox("Render to PNG", False)
            self.gui_framerate = self.server.gui.add_slider("FPS", min=1, max=120, step=0.1, initial_value=10)
            framerate_opts = self.server.gui.add_button_group("FPS options", ("10", "20", "30", "60", "120"))
            gui_up = self.server.gui.add_vector3("Up Direction", initial_value=(0.0, 0.0, 1.0), step=0.01)

        with self.server.gui.add_folder("Scene"):
            self.grid_visible = self.server.gui.add_checkbox("Show Grid", True)

        with self.server.gui.add_folder("Camera"):
            self.save_view_btn = self.server.gui.add_button("Save Current View")
            self.load_view_btn = self.server.gui.add_button("Load Saved View")

        with self.server.gui.add_folder("Video Rendering"):
            self.video_width = self.server.gui.add_number("Video Width", initial_value=1920, min=640, max=3840)
            self.video_height = self.server.gui.add_number("Video Height", initial_value=1080, min=480, max=2160)
            self.video_fps = self.server.gui.add_slider("Video FPS", min=10, max=60, step=1, initial_value=30)
            self.render_video_btn = self.server.gui.add_button("Render Full Video")

        @gui_up.on_update
        def _(_):
            self.server.scene.set_up_direction(gui_up.value)

        @self.grid_visible.on_update
        def _(_):
            self.update_floor()

        @self.save_view_btn.on_click
        def _(event):
            self.save_current_view(event)

        @self.load_view_btn.on_click
        def _(event):
            self.load_saved_view(self._get_active_client(event))

        @self.gui_timestep.on_update
        def _(_):
            self.update_scene(self.gui_timestep.value)

        @framerate_opts.on_click
        def _(_):
            self.gui_framerate.value = int(framerate_opts.value)

        @self.gui_next_frame.on_click
        def _(_):
            self.gui_timestep.value = (self.gui_timestep.value + 10) % max(self.num_frames, 1)

        @self.gui_prev_frame.on_click
        def _(_):
            self.gui_timestep.value = (self.gui_timestep.value - 1) % max(self.num_frames, 1)

        @self.gui_pause.on_click
        def _(_):
            self.gui_playing.value = False

        @self.gui_resume.on_click
        def _(_):
            self.gui_playing.value = True

        @self.gui_playing.on_update
        def _(_):
            self.gui_timestep.disabled = self.gui_playing.value
            self.gui_next_frame.disabled = self.gui_playing.value
            self.gui_prev_frame.disabled = self.gui_playing.value
            self.gui_pause.disabled = not self.gui_playing.value
            self.gui_resume.disabled = self.gui_playing.value

        @self.render_video_btn.on_click
        def _(_):
            self.render_full_video(self.scene_title)

    def _add_lights(self):
        self.server.scene.add_transform_controls("/system/control_light0", position=(5, 1.0, 0.5), scale=0.5)
        self.server.scene.add_label("/system/control_light0/label", "Point")
        self.server.scene.add_light_point("/system/control_light0/point_light", color=(255, 255, 255), intensity=100.0, visible=True)
        self.server.scene.add_transform_controls("/system/control_light1", position=(5, -1.0, 0.5), scale=0.5)
        self.server.scene.add_label("/system/control_light1/label", "Point")
        self.server.scene.add_light_point("/system/control_light1/point_light", color=(255, 255, 255), intensity=100.0, visible=True)
        self.server.scene.enable_default_lights(True)

    # ---------------------------------------------------------------- rendering
    def render_current_frame(self, timestep, render_dir: Optional[Path] = None):
        if render_dir is None:
            render_dir = Path("rendered")
        render_dir.mkdir(exist_ok=True)
        clients = list(self.server.get_clients().values())
        if not clients:
            return
        try:
            img = clients[0].get_render(height=int(self.video_height.value),
                                        width=int(self.video_width.value))
            cv2.imwrite(str(render_dir / f"{timestep:05d}.jpeg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"render failed {timestep}: {e}")

    def render_full_video(self, scene_title=None):
        if self.num_frames <= 0:
            print("no frames to render")
            return
        clients = list(self.server.get_clients().values())
        if not clients:
            return
        client = clients[0]
        w, h, fps = int(self.video_width.value), int(self.video_height.value), int(self.video_fps.value)
        rdir = Path("rendered_video")
        fdir = rdir / "frames"
        rdir.mkdir(exist_ok=True)
        fdir.mkdir(exist_ok=True)
        vpath = rdir / (f"rendered_{scene_title}.mp4" if scene_title else "rendered.mp4")
        writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            print(f"open writer failed: {vpath}")
            return
        prev_play, prev_png = self.gui_playing.value, self.render_png.value
        self.gui_playing.value, self.render_png.value = False, False
        try:
            for t in range(self.num_frames):
                self.update_scene(t)
                try:
                    img = client.get_render(height=h, width=w)
                except Exception as e:
                    print(f"frame {t}: {e}")
                    continue
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(fdir / f"{t:05d}.jpeg"), bgr)
                writer.write(bgr)
        finally:
            writer.release()
            self.gui_playing.value, self.render_png.value = prev_play, prev_png
        print(f"saved: {vpath}")

    # ---------------------------------------------------------------- helpers
    def add_frame(self, name: str, T: np.ndarray):
        self.frame_nodes[name] = self.server.scene.add_frame(
            name=f"/{name}/frame", show_axes=True,
            axes_length=0.05, axes_radius=0.002,
            position=T[:3, 3], wxyz=_wxyz(T),
        )

    def add_sphere(self, name: str, position, radius: float = 0.05, color=(1.0, 0, 0)):
        self.server.scene.add_icosphere(
            name=f"/spheres/{name}", radius=radius,
            color=tuple(int(c * 255) for c in color), position=position,
        )

    def add_arrow(self, name: str, start, end, color=(0, 255, 0),
                  shaft_radius: float = 0.01, head_radius: float = 0.02, head_length: float = 0.5):
        del head_radius, head_length
        self.server.scene.add_spline_catmull_rom(
            name=f"/arrows/{name}",
            positions=np.array([start, end]),
            color=tuple(c / 255.0 for c in color),
            line_width=shaft_radius * 1000,
        )

    def add_camera(self, name: str, extrinsic, intrinsic,
                   color=(0, 255, 0), size: float = 0.1,
                   show_axes: bool = True, image=None,
                   fov_scale: float = 1.0,
                   fov_override: Optional[float] = None,
                   aspect_override: Optional[float] = None):
        extrinsic = np.asarray(extrinsic)
        if extrinsic.shape == (3, 4):
            ext = np.eye(4); ext[:3, :] = extrinsic
        else:
            ext = extrinsic
        cam_pos = ext[:3, 3]; cam_rot = ext[:3, :3]

        if isinstance(intrinsic, dict):
            if "intrinsics_undistort" in intrinsic:
                K = np.asarray(intrinsic["intrinsics_undistort"])
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            else:
                fx, fy = intrinsic.get("fx"), intrinsic.get("fy")
                cx, cy = intrinsic.get("cx"), intrinsic.get("cy")
            width = intrinsic.get("width", 640); height = intrinsic.get("height", 480)
        else:
            K = np.asarray(intrinsic)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            width, height = cx * 2, cy * 2

        frame_handle = self.server.scene.add_frame(
            f"/cameras/{name}_frame",
            position=cam_pos, wxyz=_wxyz(cam_rot),
            show_axes=show_axes, axes_length=size * 0.5, axes_radius=size * 0.01,
        )
        color_n = tuple(c / 255.0 for c in color)
        fov_scale = max(1e-3, float(fov_scale))
        frustum_handle = None

        if image is not None:
            fov = float(fov_override) if fov_override is not None else 2.0 * np.arctan2(float(height) * 0.5, float(fy))
            fov = float(np.clip(fov * fov_scale, 1e-4, np.pi - 1e-3))
            aspect = float(aspect_override) if aspect_override is not None else float(width) / max(float(height), 1.0)
            frustum_handle = self.server.scene.add_camera_frustum(
                name=f"/cameras/{name}_frame/frustum",
                fov=fov, aspect=aspect, scale=float(size),
                line_width=2.0, color=color, image=image,
                variant="filled", position=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        else:
            d = size
            if fov_override is not None:
                fov = float(np.clip(float(fov_override) * fov_scale, 1e-4, np.pi - 1e-3))
                aspect = float(aspect_override) if aspect_override is not None else float(width) / max(float(height), 1.0)
                hh = np.tan(0.5 * fov) * d; hw = hh * aspect
                corners = np.array([[-hw, -hh, d], [hw, -hh, d], [hw, hh, d], [-hw, hh, d]])
            else:
                corners = np.array([
                    [((0 - cx) / fx) * d * fov_scale, ((0 - cy) / fy) * d * fov_scale, d],
                    [((width - cx) / fx) * d * fov_scale, ((0 - cy) / fy) * d * fov_scale, d],
                    [((width - cx) / fx) * d * fov_scale, ((height - cy) / fy) * d * fov_scale, d],
                    [((0 - cx) / fx) * d * fov_scale, ((height - cy) / fy) * d * fov_scale, d],
                ])
            origin = np.zeros(3)
            for i, corner in enumerate(corners):
                self.server.scene.add_spline_catmull_rom(
                    f"/cameras/{name}_frame/edge_{i}",
                    positions=np.array([origin, corner]), color=color_n, line_width=2.0,
                )
            for i in range(4):
                self.server.scene.add_spline_catmull_rom(
                    f"/cameras/{name}_frame/rect_{i}",
                    positions=np.array([corners[i], corners[(i + 1) % 4]]),
                    color=color_n, line_width=2.0,
                )
            self.server.scene.add_icosphere(
                f"/cameras/{name}_frame/center", radius=size * 0.05,
                color=color_n, position=origin,
            )

        self.camera_dict[name] = {
            "frame": frame_handle, "frustum": frustum_handle,
            "intrinsic": intrinsic, "extrinsic": ext,
        }
        return frame_handle

    def add_point_cloud(self, name: str, points: np.ndarray,
                        colors: Optional[np.ndarray] = None, point_size: float = 0.004):
        kw = {"points": np.asarray(points, dtype=np.float32), "point_size": float(point_size)}
        if colors is not None:
            kw["colors"] = np.asarray(colors, dtype=np.uint8)
        return self.server.scene.add_point_cloud(f"/clouds/{name}", **kw)

    def change_color(self, name: str, color):
        if name in self.obj_dict:
            h = self.obj_dict[name]["handle"]
            if hasattr(h, "color"):
                h.color = tuple(int(c * 255) for c in color)
