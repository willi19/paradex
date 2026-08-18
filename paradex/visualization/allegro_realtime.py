"""Non-blocking Viser display for live Allegro V5 feedback and tactile data."""

from __future__ import annotations

import time
from threading import Event, Thread
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import trimesh

from paradex.utils.path import rsc_path
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer


ALLEGRO_FINGERS = ("index", "middle", "ring", "thumb")
ALLEGRO_V5_JOINT_NAMES = tuple(f"joint_{index}_0" for index in range(16))
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


def named_allegro_qpos(
    qpos: Any,
    joint_names: Optional[Sequence[str]],
) -> Optional[dict[str, float]]:
    """Validate and reorder ROS feedback for the V5 URDF."""
    values = np.asarray(qpos, dtype=np.float64).reshape(-1)
    names = tuple(joint_names or ())
    if values.size != len(names) or not np.all(np.isfinite(values)):
        return None
    by_name = dict(zip(names, values))
    if any(name not in by_name for name in ALLEGRO_V5_JOINT_NAMES):
        return None
    return {name: float(by_name[name]) for name in ALLEGRO_V5_JOINT_NAMES}


def tactile_arrow_length(
    level: float,
    *,
    threshold: float,
    display_max: float,
    max_length: float,
) -> float:
    """Map a raw tactile magnitude to a bounded visible arrow length."""
    level = float(level)
    threshold = float(threshold)
    display_max = float(display_max)
    max_length = float(max_length)
    if not np.isfinite(level) or level <= threshold:
        return 0.0
    span = max(display_max - threshold, 1.0)
    return float(np.clip((level - threshold) / span, 0.0, 1.0) * max_length)


def make_tactile_arrow(
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    color: tuple[int, int, int, int],
) -> Optional[trimesh.Trimesh]:
    """Build the compact arrow mesh used by ``visualize_all.py``."""
    if length <= 1.0e-6:
        return None
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-9:
        return None
    direction = direction / norm
    shaft_height = length * 0.72
    head_height = length - shaft_height
    shaft = trimesh.creation.cylinder(
        radius=min(0.002, length * 0.10),
        height=shaft_height,
        sections=12,
    )
    head = trimesh.creation.cone(
        radius=min(0.004, length * 0.18),
        height=head_height,
        sections=12,
    )
    shaft.apply_translation((0.0, 0.0, shaft_height * 0.5))
    head.apply_translation((0.0, 0.0, shaft_height + head_height * 0.5))
    arrow = trimesh.util.concatenate((shaft, head))
    arrow.apply_transform(trimesh.geometry.align_vectors((0.0, 0.0, 1.0), direction))
    arrow.apply_translation(np.asarray(start, dtype=np.float64))
    arrow.visual.vertex_colors = np.tile(
        np.asarray(color, dtype=np.uint8),
        (len(arrow.vertices), 1),
    )
    return arrow


def centered_robot_offset(robot: RobotModule) -> np.ndarray:
    """Translate the canonical hand mesh so its bounding box is at the origin."""
    mesh = robot.get_robot_mesh(collision_geometry=False)
    if mesh is None or len(mesh.vertices) == 0:
        return np.zeros(3, dtype=np.float64)
    return -np.asarray(mesh.bounding_box.centroid, dtype=np.float64)


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
    """Render live ROS2 feedback without blocking the capture loop."""

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
        update_rate_hz: float = 20.0,
        tactile_threshold: float = 200.0,
        tactile_display_max: float = 5000.0,
        max_arrow_length: float = 0.06,
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
        if not np.isfinite(tactile_threshold) or tactile_threshold < 0.0:
            raise ValueError("tactile_threshold must be a non-negative finite value")

        self.hand = hand
        self.update_period = 1.0 / float(update_rate_hz)
        self.tactile_threshold = float(tactile_threshold)
        self.tactile_display_max = float(tactile_display_max)
        self.max_arrow_length = float(max_arrow_length)
        self.tactile_max_age_s = float(tactile_max_age_s)
        self.exit_event = Event()
        self.thread: Optional[Thread] = None
        self.arrow_handles = {finger: None for finger in ALLEGRO_FINGERS}

        self.viewer = ViserViewer(
            scene_title="Live Allegro feedback + tactile",
            show_player=False,
        )
        self.viewer.add_robot(
            "allegro_feedback",
            urdf_path,
            include_arm_meshes=True,
            # Create only Allegro mesh nodes. This is an allowlist applied
            # before Viser nodes are created, not a visibility toggle.
            mesh_link_names=ALLEGRO_V5_VISUAL_LINKS,
        )
        self.viser_robot = self.viewer.robot_dict["allegro_feedback"]
        self.robot = self.viser_robot.urdf
        self.scene_offset = centered_robot_offset(self.robot)
        # Keep the hand itself centered. All tactile meshes receive this same
        # translation because they live directly under the Viser world root.
        self.viser_robot._visual_root_frame.position = self.scene_offset

        with self.viewer.server.gui.add_folder("Live Allegro", expand_by_default=True):
            self.connection_status = self.viewer.server.gui.add_text(
                "ROS2 feedback",
                initial_value="WAITING",
                disabled=True,
            )
            self.tactile_status = self.viewer.server.gui.add_text(
                "Tactile",
                initial_value="WAITING",
                disabled=True,
            )
            self.tactile_sliders = {
                finger: self.viewer.server.gui.add_slider(
                    f"{finger} tactile raw",
                    min=0.0,
                    max=self.tactile_display_max,
                    step=1.0,
                    initial_value=0.0,
                    disabled=True,
                )
                for finger in ALLEGRO_FINGERS
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
        print("Allegro feedback/tactile Viser started at http://localhost:8080")

    def _clear_arrows(self) -> None:
        for finger, handle in self.arrow_handles.items():
            if handle is not None:
                handle.remove()
                self.arrow_handles[finger] = None

    def _clear_tactile_display(self, status: str) -> None:
        self.tactile_status.value = status
        for slider in self.tactile_sliders.values():
            slider.value = 0.0
        self._clear_arrows()

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
            self.tactile_sliders[finger].value = float(
                np.clip(level, 0.0, self.tactile_display_max)
            )
            length = tactile_arrow_length(
                level,
                threshold=self.tactile_threshold,
                display_max=self.tactile_display_max,
                max_length=self.max_arrow_length,
            )
            link_name = ALLEGRO_V5_TIP_LINKS[finger]
            anchor, direction = fingertip_surface_arrow_frame(
                self.robot,
                link_name,
            )
            arrow = make_tactile_arrow(
                anchor + self.scene_offset,
                direction,
                length,
                self._ARROW_COLORS[finger],
            )
            old_handle = self.arrow_handles[finger]
            if old_handle is not None:
                old_handle.remove()
                self.arrow_handles[finger] = None
            if arrow is not None:
                self.arrow_handles[finger] = (
                    self.viewer.server.scene.add_mesh_trimesh(
                        f"/tactile/{finger}",
                        arrow,
                    )
                )

    def update_once(self, feedback: Mapping[str, Any]) -> bool:
        if not feedback.get("is_connected", False):
            self.connection_status.value = "WAITING FOR JOINT FEEDBACK"
            with self.viewer.server.atomic():
                self._clear_tactile_display("WAITING FOR JOINT FEEDBACK")
            return False
        qpos = named_allegro_qpos(
            feedback.get("qpos"),
            feedback.get("joint_names"),
        )
        if qpos is None:
            self.connection_status.value = "INVALID JOINT FEEDBACK"
            with self.viewer.server.atomic():
                self._clear_tactile_display("WAITING FOR VALID JOINT FEEDBACK")
            return False

        topic = feedback.get("state_topic", "/right/allegroHand_0/joint_states")
        with self.viewer.server.atomic():
            self.viser_robot.update_cfg(qpos)
            self.connection_status.value = f"CONNECTED: {topic}"

            levels = (
                allegro_tactile_finger_levels(feedback.get("tactile"))
                if self._tactile_is_fresh(feedback)
                else None
            )
            if levels is None:
                self._clear_tactile_display("WAITING OR STALE")
            else:
                self.tactile_status.value = "LIVE: " + ", ".join(
                    f"{finger}={levels[finger]:.0f}" for finger in ALLEGRO_FINGERS
                )
                self._update_arrows(levels)
        return True

    def _run(self) -> None:
        while not self.exit_event.is_set():
            started = time.perf_counter()
            try:
                self.update_once(self.hand.get_data())
            except Exception as exc:
                self.connection_status.value = f"VISUALIZATION ERROR: {exc}"
            remaining = self.update_period - (time.perf_counter() - started)
            self.exit_event.wait(timeout=max(0.0, remaining))

    def close(self) -> None:
        self.exit_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        # Viser owns an atexit shutdown hook. Calling ``server.stop()`` here
        # would make that hook stop an already-closed asyncio loop.
        with self.viewer.server.atomic():
            self._clear_arrows()
