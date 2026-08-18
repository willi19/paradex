"""Non-blocking Viser display for live Allegro V5 feedback and tactile data."""

from __future__ import annotations

import time
from threading import Event, Thread
from typing import Any, Mapping, Optional

import numpy as np
import trimesh

from paradex.utils.path import rsc_path
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer


ALLEGRO_FINGERS = ("index", "middle", "ring", "thumb")
ALLEGRO_V5_TIP_LINKS = {
    "index": "link_3_0_tip",
    "middle": "link_7_0_tip",
    "ring": "link_11_0_tip",
    "thumb": "link_15_0_tip",
}
ALLEGRO_V5_VISUAL_LINKS = frozenset(
    ("palm_link",)
    + tuple(f"link_{index}_0" for index in range(16))
    + tuple(ALLEGRO_V5_TIP_LINKS.values())
)
# Fixed V5 fingertip mesh vertices, matching visualize_all.py's explicit
# ``TACTILE_VERTEX_MAP -> compute_contact_arrow`` contract. All four V5 tip
# meshes use the same vertex topology.
ALLEGRO_V5_TACTILE_VERTEX_IDS = (1783, 1601, 2222, 1588)
DEFAULT_ALLEGRO_V5_URDF = (
    rsc_path + "/robot/allegro_v5/allegro_right_A.urdf"
)


def allegro_tactile_finger_levels(tactile: Any) -> Optional[dict[str, float]]:
    """Reduce the unnamed tactile vector to one magnitude per finger.

    The driver publishes four equally-sized contiguous blocks in
    index/middle/ring/thumb order. Invalid packets are rejected rather than
    partially displayed with a guessed layout.
    """
    if tactile is None:
        return None
    values = np.asarray(tactile, dtype=np.float64).reshape(-1)
    if values.size < len(ALLEGRO_FINGERS) or not np.all(np.isfinite(values)):
        return None
    blocks = np.array_split(np.abs(values), len(ALLEGRO_FINGERS))
    if any(block.size == 0 for block in blocks):
        return None
    return {
        finger: float(np.max(block))
        for finger, block in zip(ALLEGRO_FINGERS, blocks)
    }


def tactile_arrow_length(
    level: float,
    *,
    display_max: float,
    max_length: float,
) -> float:
    """Map every positive tactile magnitude to a bounded arrow length."""
    level = float(level)
    display_max = float(display_max)
    max_length = float(max_length)
    if not np.isfinite(level) or level <= 0.0:
        return 0.0
    return float(np.clip(level / display_max, 0.0, 1.0) * max_length)


def make_tactile_arrow_head(
    direction: np.ndarray,
) -> trimesh.Trimesh:
    """Build one fixed arrowhead; only its position changes at runtime."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError("Arrow direction must be finite and nonzero")
    direction = direction / norm
    head_height = 0.015
    head = trimesh.creation.cone(radius=0.005, height=head_height, sections=12)
    # Put the cone tip at the local origin, then orient it along the fixed
    # fingertip normal. Runtime updates only translate this mesh handle.
    head.apply_translation((0.0, 0.0, -head_height * 0.5))
    head.apply_transform(
        trimesh.geometry.align_vectors((0.0, 0.0, 1.0), direction)
    )
    return head


def centered_robot_offset(robot: RobotModule) -> np.ndarray:
    """Translate the canonical hand mesh so its bounding box is at the origin."""
    mesh = canonical_allegro_mesh(robot)
    if mesh is None or len(mesh.vertices) == 0:
        return np.zeros(3, dtype=np.float64)
    return -np.asarray(mesh.bounding_box.centroid, dtype=np.float64)


def canonical_allegro_mesh(robot: RobotModule) -> trimesh.Trimesh:
    """Combine only Allegro visual geometry into one static qpos=0 mesh."""
    meshes = []
    scene = robot.scene
    for geometry_name, mesh in scene.geometry.items():
        parent_link = scene.graph.transforms.parents[geometry_name]
        if parent_link not in ALLEGRO_V5_VISUAL_LINKS:
            continue
        transformed = mesh.copy()
        transformed.apply_transform(scene.graph.get(geometry_name)[0])
        meshes.append(transformed)
    if not meshes:
        raise ValueError("No Allegro V5 visual geometry found in URDF")
    return trimesh.util.concatenate(meshes)


def fingertip_surface_arrow_frame(
    robot: RobotModule,
    link_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Use fixed fingertip vertices exactly like visualize_all.py."""
    scene = robot.scene
    geometry_name = next(
        (
            node
            for node, parent in scene.graph.transforms.parents.items()
            if parent == link_name and node in scene.geometry
        ),
        None,
    )
    if geometry_name is None:
        raise ValueError(f"No visual fingertip mesh is attached to {link_name}")

    tip_mesh = scene.geometry[geometry_name].copy()
    tip_mesh.apply_transform(scene.graph.get(geometry_name)[0])
    vertex_ids = np.asarray(ALLEGRO_V5_TACTILE_VERTEX_IDS, dtype=int)
    if np.max(vertex_ids) >= len(tip_mesh.vertices):
        raise ValueError(
            f"Fingertip mesh for {link_name} does not match the V5 tactile vertex map"
        )
    anchor = np.asarray(tip_mesh.vertices[vertex_ids]).mean(axis=0)
    normal = np.asarray(tip_mesh.vertex_normals[vertex_ids]).mean(axis=0)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 1.0e-9:
        raise ValueError(f"Invalid tactile surface normal for {link_name}")
    normal /= normal_norm
    return anchor, normal


class AllegroRealtimeViser:
    """Render a static canonical hand with lightweight live tactile arrows."""

    _ARROW_COLORS = {
        "index": (255, 80, 60, 255),
        "middle": (255, 150, 40, 255),
        "ring": (250, 220, 40, 255),
        "thumb": (80, 220, 255, 255),
    }

    def __init__(
        self,
        hand: Any,
        *,
        update_rate_hz: float = 100.0,
        tactile_display_max: float = 1000.0,
        max_arrow_length: float = 0.1,
        tactile_max_age_s: float = 0.25,
        urdf_path: str = DEFAULT_ALLEGRO_V5_URDF,
    ) -> None:
        positive = {
            "update_rate_hz": update_rate_hz,
            "tactile_display_max": tactile_display_max,
            "max_arrow_length": max_arrow_length,
            "tactile_max_age_s": tactile_max_age_s,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value")
        self.hand = hand
        self.update_period = 1.0 / float(update_rate_hz)
        self.tactile_display_max = float(tactile_display_max)
        self.max_arrow_length = float(max_arrow_length)
        self.tactile_max_age_s = float(tactile_max_age_s)
        self.exit_event = Event()
        self.thread: Optional[Thread] = None
        self.arrow_handles = {}

        self.viewer = ViserViewer(
            scene_title="Live Allegro feedback + tactile",
            show_player=False,
        )
        self.robot = RobotModule(urdf_path)
        self.robot.update_cfg(np.zeros(self.robot.get_num_joints()))
        self.scene_offset = centered_robot_offset(self.robot)
        static_mesh = canonical_allegro_mesh(self.robot)
        static_mesh.apply_translation(self.scene_offset)
        # One combined mesh, no robot joint frames and no per-link scene nodes.
        self.robot_mesh_handle = self.viewer.server.scene.add_mesh_trimesh(
            "/allegro",
            static_mesh,
        )
        self._create_static_arrow_handles()

    def _create_static_arrow_handles(self) -> None:
        """Create four shafts and heads once; never recreate mesh geometry."""
        for finger in ALLEGRO_FINGERS:
            anchor, direction = fingertip_surface_arrow_frame(
                self.robot,
                ALLEGRO_V5_TIP_LINKS[finger],
            )
            anchor = anchor + self.scene_offset
            color = self._ARROW_COLORS[finger]
            shaft = self.viewer.server.scene.add_spline_catmull_rom(
                f"/tactile/{finger}/shaft",
                points=np.stack((anchor, anchor)),
                line_width=6.0,
                color=color[:3],
                visible=False,
            )
            head_mesh = make_tactile_arrow_head(direction)
            head = self.viewer.server.scene.add_mesh_simple(
                f"/tactile/{finger}/head",
                vertices=np.asarray(head_mesh.vertices),
                faces=np.asarray(head_mesh.faces),
                color=color[:3],
                position=anchor,
                visible=False,
            )
            self.arrow_handles[finger] = {
                "anchor": anchor,
                "direction": direction,
                "shaft": shaft,
                "head": head,
            }

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = Thread(
            target=self._run,
            name="allegro-realtime-viser",
            daemon=True,
        )
        self.thread.start()
        print("Static Allegro/tactile Viser started at http://localhost:8080")

    def _clear_arrows(self) -> None:
        for handles in self.arrow_handles.values():
            handles["shaft"].visible = False
            handles["head"].visible = False

    def _tactile_is_fresh(self, feedback: Mapping[str, Any]) -> bool:
        tactile_time = feedback.get("tactile_time")
        return (
            tactile_time is not None
            and np.isfinite(tactile_time)
            and time.perf_counter() - float(tactile_time) <= self.tactile_max_age_s
        )

    def _update_arrows(self, levels: Mapping[str, float]) -> None:
        for finger in ALLEGRO_FINGERS:
            level = float(levels[finger])
            length = tactile_arrow_length(
                level,
                display_max=self.tactile_display_max,
                max_length=self.max_arrow_length,
            )
            handles = self.arrow_handles[finger]
            if length <= 0.0:
                handles["shaft"].visible = False
                handles["head"].visible = False
                continue
            endpoint = handles["anchor"] + handles["direction"] * length
            handles["shaft"].points = np.stack((handles["anchor"], endpoint))
            handles["head"].position = endpoint
            handles["shaft"].visible = True
            handles["head"].visible = True

    def update_once(self, feedback: Mapping[str, Any]) -> bool:
        levels = (
            allegro_tactile_finger_levels(feedback.get("tactile"))
            if self._tactile_is_fresh(feedback)
            else None
        )
        with self.viewer.server.atomic():
            if levels is None:
                self._clear_arrows()
            else:
                self._update_arrows(levels)
        return levels is not None

    def _run(self) -> None:
        while not self.exit_event.is_set():
            started = time.perf_counter()
            try:
                self.update_once(self.hand.get_data())
            except Exception as exc:
                print(f"Allegro tactile visualization error: {exc}")
                self.exit_event.set()
            remaining = self.update_period - (time.perf_counter() - started)
            self.exit_event.wait(timeout=max(0.0, remaining))

    def render_bgr(self, height: int, width: int) -> Optional[np.ndarray]:
        """Return the latest browser-rendered Viser view for OpenCV preview."""
        clients = tuple(self.viewer.server.get_clients().values())
        if not clients:
            return None

        # Keep browser rendering bounded; the camera preview scales this image
        # into its full-size right panel after it arrives.
        scale = min(1.0, 720.0 / height, 960.0 / width)
        render_height = max(1, int(round(height * scale)))
        render_width = max(1, int(round(width * scale)))
        rgb = clients[0].get_render(
            height=render_height,
            width=render_width,
            transport_format="jpeg",
        )
        return np.asarray(rgb[..., :3])[..., ::-1].copy()

    def close(self) -> None:
        self.exit_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        # Viser owns an atexit shutdown hook. Calling ``server.stop()`` here
        # would make that hook stop an already-closed asyncio loop.
        with self.viewer.server.atomic():
            self._clear_arrows()
