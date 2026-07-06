"""Interactive viser backend for the shared :class:`Scene` API.

``ViserScene`` renders in a browser and owns a built-in playback GUI, so callers
never re-implement a slider/loop. It reuses the proven URDF mesh loader
``ViserRobotModule`` from the legacy ``visualizer.viser`` module.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from paradex.visualization.scene.base import Color, Pose, Scene, Timeline, normalize_color
from paradex.visualization.visualizer.viser import ViserRobotModule


def _wxyz(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> viser wxyz quaternion."""
    return R.from_matrix(rot).as_quat()[[3, 0, 1, 2]]


class ViserScene(Scene):
    def __init__(self, up_direction=(0, 0, 1), port: int = 8080, dark: bool = True) -> None:
        super().__init__()
        self.port = port
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(dark_mode=dark)
        self.server.scene.set_up_direction(np.asarray(up_direction, dtype=float))

        self._object_meta: dict = {}   # name -> {frame, handle, mesh}
        self._robot_pose: dict = {}     # name -> 4x4 root pose
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._add_lights()
        self._build_player()

    # ------------------------------------------------------------------ #
    # Scene construction
    # ------------------------------------------------------------------ #
    def add_robot(self, name: str, urdf_path: str, pose: Optional[Pose] = None,
                  color: Color = None) -> ViserRobotModule:
        robot = ViserRobotModule(
            target=self.server,
            urdf_path=urdf_path,
            scale=1.0,
            root_node_name=f"/robot/{name}",
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._robots[name] = robot
        self._robot_pose[name] = np.eye(4)
        if pose is not None:
            self.set_pose(name, pose)
        if color is not None:
            self.change_color(name, color)
        return robot

    def add_object(self, name: str, mesh: trimesh.Trimesh, pose: Optional[Pose] = None,
                   color: Color = None):
        obj_T = np.eye(4) if pose is None else np.asarray(pose, dtype=float)
        frame = self.server.scene.add_frame(
            f"/objects/{name}_frame",
            position=obj_T[:3, 3],
            wxyz=_wxyz(obj_T[:3, :3]),
            show_axes=False,
        )
        handle = self.server.scene.add_mesh_trimesh(
            name=f"/objects/{name}_frame/{name}",
            mesh=mesh,
            cast_shadow=True,
            receive_shadow=True,
        )
        self._objects[name] = handle
        self._object_meta[name] = {"frame": frame, "handle": handle, "mesh": mesh}
        if color is not None:
            self.change_color(name, color)
        return handle

    # trimesh alias for parity with the old API
    add_trimesh = add_object

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def set_qpos(self, name: str, qpos: np.ndarray) -> None:
        self._robots[name].update_cfg(np.asarray(qpos))

    def set_pose(self, name: str, pose: Pose) -> None:
        T = np.asarray(pose, dtype=float)
        if name in self._robots:
            robot = self._robots[name]
            self._robot_pose[name] = T
            if hasattr(robot, "_visual_root_frame"):
                robot._visual_root_frame.position = T[:3, 3]
                robot._visual_root_frame.wxyz = _wxyz(T[:3, :3])
        elif name in self._object_meta:
            frame = self._object_meta[name]["frame"]
            frame.position = T[:3, 3]
            frame.wxyz = _wxyz(T[:3, :3])
        else:
            raise KeyError(f"'{name}' is not a known robot or object")

    def change_color(self, name: str, color: Color,
                     links: Optional[Sequence[str]] = None) -> None:
        rgb, opacity = normalize_color(color)
        if rgb is None:
            return
        if name in self._robots:
            rgba = (*rgb, opacity if opacity is not None else 1.0)
            self._robots[name].change_color(list(links) if links else [], rgba)
        elif name in self._object_meta:
            handle = self._object_meta[name]["handle"]
            handle.color = rgb
            if opacity is not None:
                handle.opacity = opacity
        else:
            raise KeyError(f"'{name}' is not a known robot or object")

    def set_visible(self, name: str, visible: bool) -> None:
        if name in self._robots:
            self._robots[name].show_visual = visible
        elif name in self._object_meta:
            self._object_meta[name]["frame"].visible = visible

    def set_opacity(self, name: str, opacity: float) -> None:
        """Set an object's opacity (0-1) without touching its RGB. Robots have
        no RGB-independent opacity handle — pass RGBA to ``change_color`` for
        those."""
        if name in self._object_meta:
            self._object_meta[name]["handle"].opacity = float(opacity)
        elif name in self._robots:
            raise NotImplementedError(
                "per-robot opacity: use change_color(name, (r, g, b, a)) instead")
        else:
            raise KeyError(f"'{name}' is not a known robot or object")

    def remove(self, name: str) -> None:
        """Remove a robot or object from the scene."""
        if name in self._robots:
            try:
                self._robots[name].remove()
            except Exception:
                pass
            self._robots.pop(name, None)
            self._robot_pose.pop(name, None)
        elif name in self._object_meta:
            meta = self._object_meta.pop(name)
            self._objects.pop(name, None)
            for key in ("handle", "frame"):
                try:
                    meta[key].remove()
                except Exception:
                    pass

    def clear(self) -> None:
        """Remove every robot and object and drop the timeline."""
        for name in list(self._robots) + list(self._object_meta):
            self.remove(name)
        self._timeline = None
        self._frame = 0

    # ------------------------------------------------------------------ #
    # Extras (markers)
    # ------------------------------------------------------------------ #
    def add_sphere(self, name: str, position, radius: float = 0.05, color=(1.0, 0.0, 0.0)):
        rgb, _ = normalize_color(color)
        return self.server.scene.add_icosphere(
            name=f"/spheres/{name}", radius=radius, color=rgb, position=position,
            cast_shadow=True, receive_shadow=True,
        )

    def add_frame(self, name: str, T: Pose, scale: float = 0.1):
        return self.server.scene.add_frame(
            name=f"/frames/{name}", show_axes=True,
            axes_length=scale, axes_radius=scale * 0.04,
            position=T[:3, 3], wxyz=_wxyz(T[:3, :3]),
        )

    def add_grid(self, size: float = 10.0, cell_size: float = 0.1, height: float = 0.0):
        return self.server.scene.add_grid(
            name="/grid", width=size, height=size, plane="xy",
            position=(0.0, 0.0, height), cell_size=cell_size,
        )

    # ------------------------------------------------------------------ #
    # Appearance panel (replaces the hand-copied add_appearance_gui in ~40 files)
    # ------------------------------------------------------------------ #
    def add_appearance_panel(self, folder: str = "Appearance") -> None:
        """Auto-generate per-entity color / opacity / visibility controls for
        every robot and object currently in the scene. Wired to
        ``change_color`` / ``set_visible`` — no per-app copy needed.

        Call *after* you've added your robots/objects.
        """
        with self.server.gui.add_folder(folder):
            for name in list(self._robots) + list(self._object_meta):
                with self.server.gui.add_folder(name):
                    rgb = self.server.gui.add_rgb("Color", (200, 200, 200))
                    opacity = self.server.gui.add_slider(
                        "Opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0)
                    visible = self.server.gui.add_checkbox("Visible", True)

                def _apply(_=None, name=name, rgb=rgb, opacity=opacity):
                    r, g, b = (c / 255.0 for c in rgb.value)
                    self.change_color(name, (r, g, b, opacity.value))

                def _vis(_=None, name=name, visible=visible):
                    self.set_visible(name, visible.value)

                rgb.on_update(_apply)
                opacity.on_update(_apply)
                visible.on_update(_vis)

    # ------------------------------------------------------------------ #
    # Capture panel: interpolated camera-path video + PNG (was copied ~everywhere)
    # ------------------------------------------------------------------ #
    def add_capture_panel(self, folder: str = "Capture") -> None:
        """GUI to set a start/end camera view, record an interpolated fly-through
        video, and grab stills. Needs a connected browser client."""
        with self.server.gui.add_folder(folder):
            self.cap_width = self.server.gui.add_number("Width", initial_value=1280)
            self.cap_height = self.server.gui.add_number("Height", initial_value=720)
            self.cap_fps = self.server.gui.add_slider("FPS", min=10, max=60, step=1, initial_value=30)
            self.cap_duration = self.server.gui.add_slider("Duration (s)", min=1.0, max=10.0, step=0.5, initial_value=3.0)
            self.cap_out = self.server.gui.add_text("Output Path", initial_value="")
            set_start = self.server.gui.add_button("Set Start View")
            set_end = self.server.gui.add_button("Set End View")
            rec = self.server.gui.add_button("Record Fly-through")
            png = self.server.gui.add_button("Capture PNG")

        def _client(self):
            clients = self.server.get_clients()
            return next(iter(clients.values())) if clients else None

        @set_start.on_click
        def _(_):
            c = _client(self)
            if c:
                self._start_view = {"position": c.camera.position, "wxyz": c.camera.wxyz}

        @set_end.on_click
        def _(_):
            c = _client(self)
            if c:
                self._end_view = {"position": c.camera.position, "wxyz": c.camera.wxyz}

        @rec.on_click
        def _(_):
            out = self.cap_out.value.strip() or "flythrough.mp4"
            if not out.endswith(".mp4"):
                out += ".mp4"
            self.record_flythrough(out)

        @png.on_click
        def _(_):
            import datetime as _dt
            out = self.cap_out.value.strip() or f"capture_{_dt.datetime.now():%Y%m%d_%H%M%S}.png"
            self.capture_png(out, width=int(self.cap_width.value), height=int(self.cap_height.value))

    def record_flythrough(self, out_path: str, n_frames: Optional[int] = None) -> str:
        """Render an interpolated video between the start/end views set in the
        capture panel (position lerp + rotation slerp)."""
        import os
        import shutil
        import subprocess
        import tempfile

        if not hasattr(self, "_start_view") or not hasattr(self, "_end_view"):
            raise RuntimeError("set both start and end views first")
        clients = self.server.get_clients()
        if not clients:
            raise RuntimeError("no browser client connected")
        client = next(iter(clients.values()))

        fps = int(self.cap_fps.value)
        n = n_frames or int(fps * float(self.cap_duration.value))
        w, h = int(self.cap_width.value), int(self.cap_height.value)
        p0, p1 = np.array(self._start_view["position"]), np.array(self._end_view["position"])
        q0, q1 = np.array(self._start_view["wxyz"]), np.array(self._end_view["wxyz"])
        r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
        r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
        slerp = Slerp([0, 1], R.concatenate([r0, r1]))

        tmp = tempfile.mkdtemp()
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            client.camera.position = tuple(p0 * (1 - t) + p1 * t)
            q = slerp(t).as_quat()
            client.camera.wxyz = (q[3], q[0], q[1], q[2])
            self.capture_png(os.path.join(tmp, f"f_{i:04d}.png"), width=w, height=h)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "warning", "-framerate", str(fps),
            "-i", os.path.join(tmp, "f_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path,
        ], check=True)
        shutil.rmtree(tmp)
        return out_path

    def capture_turntable(self, out_path: str, center=(0, 0, 0), radius: float = 1.5,
                          height: float = 0.5, n_frames: int = 60, fps: int = 30,
                          width: int = 1280, img_height: int = 720) -> str:
        """Orbit the camera 360° around ``center`` and render an mp4. Needs a
        connected browser client (uses ``client.camera.look_at``)."""
        import os
        import shutil
        import subprocess
        import tempfile

        clients = self.server.get_clients()
        if not clients:
            raise RuntimeError("no browser client connected")
        client = next(iter(clients.values()))
        center = np.asarray(center, float)

        tmp = tempfile.mkdtemp()
        for i, theta in enumerate(np.linspace(0, 2 * np.pi, n_frames, endpoint=False)):
            eye = center + np.array([radius * np.cos(theta), radius * np.sin(theta), height])
            client.camera.position = tuple(eye)
            client.camera.look_at = tuple(center)
            self.capture_png(os.path.join(tmp, f"f_{i:04d}.png"), width=width, height=img_height)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "warning", "-framerate", str(fps),
            "-i", os.path.join(tmp, "f_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path,
        ], check=True)
        shutil.rmtree(tmp)
        return out_path

    def add_view_io_panel(self, folder: str = "View I/O") -> None:
        """Save / load / level the current camera view (JSON)."""
        import json
        import os

        with self.server.gui.add_folder(folder):
            path = self.server.gui.add_text("File", initial_value="view.json")
            save = self.server.gui.add_button("Save View")
            load = self.server.gui.add_button("Load View")

        @save.on_click
        def _(_):
            clients = self.server.get_clients()
            if not clients:
                return
            c = next(iter(clients.values()))
            with open(path.value.strip() or "view.json", "w") as f:
                json.dump({"position": list(c.camera.position), "wxyz": list(c.camera.wxyz)}, f, indent=2)

        @load.on_click
        def _(_):
            clients = self.server.get_clients()
            p = path.value.strip() or "view.json"
            if not clients or not os.path.exists(p):
                return
            c = next(iter(clients.values()))
            with open(p) as f:
                v = json.load(f)
            c.camera.position = tuple(v["position"])
            c.camera.wxyz = tuple(v["wxyz"])

    # ------------------------------------------------------------------ #
    # Playback GUI + loop
    # ------------------------------------------------------------------ #
    def _add_lights(self) -> None:
        self.server.scene.add_light_directional(
            name="/system/key_light", color=(255, 255, 255), intensity=1.5,
            wxyz=R.from_euler("xyz", [45, 0, 45], degrees=True).as_quat()[[3, 0, 1, 2]],
            cast_shadow=True,
        )
        self.server.scene.add_light_point(
            name="/system/fill_light", color=(255, 255, 255), intensity=2.0,
            position=(-3.0, -3.0, 3.0), cast_shadow=False,
        )

    def _build_player(self) -> None:
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Frame", min=0, max=1, step=1, initial_value=0)
            self.gui_playing = self.server.gui.add_checkbox("Playing", True)
            self.gui_fps = self.server.gui.add_slider(
                "FPS", min=1, max=120, step=1, initial_value=30)

        @self.gui_timestep.on_update
        def _(_):
            if not self.gui_playing.value:
                self.goto_frame(int(self.gui_timestep.value))

        @self.gui_playing.on_update
        def _(_):
            self.gui_timestep.disabled = self.gui_playing.value

    def _on_timeline_set(self, timeline: Timeline) -> None:
        self.gui_fps.value = int(np.clip(round(timeline.fps), 1, 120))
        with self.server.atomic():
            self.gui_timestep.max = max(timeline.num_frames - 1, 1)
            self.gui_timestep.value = 0

    def _render_frame(self) -> None:
        if self.gui_timestep.value != self._frame:
            self.gui_timestep.value = self._frame
        self.server.flush()

    def _advance(self) -> None:
        if self.gui_playing.value and self.num_frames > 0:
            self.goto_frame((self._frame + 1) % self.num_frames)
        time.sleep(1.0 / max(self.gui_fps.value, 1))

    def show(self, block: bool = True) -> None:
        """Run the playback loop. ``block=False`` runs it in a daemon thread."""
        self._running = True
        if block:
            while self._running:
                self._advance()
        else:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while self._running:
            self._advance()

    def stop(self) -> None:
        self._running = False
        try:
            self.server.stop()
        except Exception as e:  # pragma: no cover
            print(f"[ViserScene.stop] {e!r}")

    # ------------------------------------------------------------------ #
    # Capture / export (needs a connected browser client)
    # ------------------------------------------------------------------ #
    def capture_png(self, out_path: str, width: int = 720, height: int = 720) -> str:
        import imageio.v2 as imageio
        while len(self.server.get_clients()) == 0:
            time.sleep(0.1)
        client = next(iter(self.server.get_clients().values()))
        img = client.get_render(height=height, width=width, transport_format="png")
        imageio.imwrite(out_path, img)
        return out_path

    def export_video(self, path: str, fps: Optional[float] = None,
                     width: int = 1280, height: int = 720) -> str:
        """Render the installed timeline to mp4. Requires a connected browser
        client (viser renders through a client). For headless rendering use the
        open3d backend instead."""
        import os
        import shutil
        import subprocess

        if self.num_frames == 0:
            raise RuntimeError("no timeline installed; call animate() first")
        fps = fps or self.fps
        was_playing = self.gui_playing.value
        self.gui_playing.value = False

        tmp = path + ".frames"
        os.makedirs(tmp, exist_ok=True)
        for t in range(self.num_frames):
            self.goto_frame(t)
            self.capture_png(os.path.join(tmp, f"f_{t:06d}.png"), width=width, height=height)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "warning", "-framerate", str(int(fps)),
            "-i", os.path.join(tmp, "f_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path,
        ], check=True)
        shutil.rmtree(tmp)
        self.gui_playing.value = was_playing
        return path
