"""Backend-agnostic scene + animation API shared by the viser and open3d backends.

The design separates two concerns that the old ``ViserViewer`` tangled together:

* **Scene** — *what* is in the world: named robots and objects. You build it once
  with :meth:`Scene.add_robot` / :meth:`Scene.add_object`.
* **Timeline** — *how* those named entities move over frames. You declare it once
  with :meth:`Scene.animate`; the base class owns the player state, frame clamping
  and video export. Backends only implement "put entity X into state Y".

Both backends (`ViserScene`, `Open3DScene`) subclass :class:`Scene` and therefore
share the *exact same* call surface. A single script can drive an interactive
viser window or render an mp4 with open3d by swapping the constructor.

Color convention everywhere: **RGB or RGBA float in 0.0–1.0**. ``None`` means
"keep the mesh's native color".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

Color = Optional[Sequence[float]]
Pose = np.ndarray  # 4x4


@dataclass
class Timeline:
    """A single shared timeline. ``apply(t)`` mutates the scene to frame ``t``."""
    num_frames: int
    fps: float
    apply: Callable[[int], None]


def normalize_color(color: Color) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
    """Split an RGB/RGBA 0-1 color into ``(rgb, opacity)``.

    Returns ``(None, None)`` for ``color is None`` (keep native color).
    ``opacity`` is ``None`` when only RGB was given, so callers can leave the
    existing opacity untouched instead of forcing it to 1.0.
    """
    if color is None:
        return None, None
    c = tuple(float(x) for x in color)
    if len(c) == 3:
        return (c[0], c[1], c[2]), None
    if len(c) == 4:
        return (c[0], c[1], c[2]), c[3]
    raise ValueError(f"color must be RGB or RGBA in 0-1, got {color!r}")


class Scene(ABC):
    """Abstract backend-agnostic scene.

    Concrete methods here (``animate``, ``sequence``, ``goto_frame``, color
    handling) are backend-independent. Everything a backend must supply is an
    ``@abstractmethod`` below.
    """

    def __init__(self) -> None:
        self._robots: Dict[str, object] = {}   # name -> backend robot handle
        self._objects: Dict[str, object] = {}  # name -> backend object handle
        self._timeline: Optional[Timeline] = None
        self._frame: int = 0

    # ------------------------------------------------------------------ #
    # Animation — the part that used to be add_traj/add_player/update_scene
    # ------------------------------------------------------------------ #
    def animate(
        self,
        robot: Optional[Dict[str, np.ndarray]] = None,
        object: Optional[Dict[str, np.ndarray]] = None,
        *,
        num_frames: Optional[int] = None,
        fps: float = 30.0,
    ) -> Union["Scene", Callable]:
        """Declare the animation. Two forms, one method:

        **Declarative** (90% case) — pass per-entity arrays that share one
        timeline and all play *together*::

            scene.animate(
                robot={"arm": qpos},        # (T, ndof)
                object={"cup": cup_poses},  # (T, 4, 4)
                fps=30,
            )

        Entity names must already exist (via ``add_robot`` / ``add_object``).
        Entities not mentioned hold their current state. ``num_frames`` is the
        longest array; shorter arrays clamp on their last value.

        **Callback** — for procedural / externally-driven per-frame logic. Use
        as a decorator; this is what replaces hand-rolled players::

            @scene.animate(num_frames=T, fps=30)
            def frame(t):
                scene.set_qpos("arm", plan.qpos[t])
                scene.set_pose("cup", track[t])

        Either form installs exactly one timeline, snaps to frame 0, and wires
        the backend player. Calling ``animate`` again *replaces* the timeline;
        use :meth:`sequence` to play clips back-to-back.
        """
        # Callback / decorator form: num_frames given, no arrays.
        if num_frames is not None and not robot and not object:
            def _decorator(fn: Callable[[int], None]) -> Callable[[int], None]:
                self._install_timeline(Timeline(int(num_frames), fps, fn))
                return fn
            return _decorator

        robot = robot or {}
        object = object or {}
        lengths = [np.asarray(a).shape[0] for a in robot.values()]
        lengths += [np.asarray(a).shape[0] for a in object.values()]
        if num_frames is None and not lengths:
            raise ValueError(
                "animate() needs either trajectory arrays (robot=/object=) "
                "or num_frames= for the callback form."
            )
        n = int(num_frames) if num_frames is not None else max(lengths)

        robot = {k: np.asarray(v) for k, v in robot.items()}
        object = {k: np.asarray(v) for k, v in object.items()}

        def _apply(t: int) -> None:
            for name, arr in robot.items():
                self.set_qpos(name, arr[min(t, len(arr) - 1)])
            for name, arr in object.items():
                self.set_pose(name, arr[min(t, len(arr) - 1)])

        self._install_timeline(Timeline(n, fps, _apply))
        return self

    def sequence(
        self,
        clips: Sequence[Dict[str, Dict[str, np.ndarray]]],
        *,
        fps: float = 30.0,
    ) -> "Scene":
        """Play several declarative clips back-to-back on one concatenated
        timeline. Each clip is ``{"robot": {...}, "object": {...}}``. This is
        the *explicit* opt-in for the old ``add_traj``-appends-in-time behavior,
        so sequencing never happens by accident.
        """
        applies = []
        offsets = []
        total = 0
        for clip in clips:
            r = {k: np.asarray(v) for k, v in clip.get("robot", {}).items()}
            o = {k: np.asarray(v) for k, v in clip.get("object", {}).items()}
            lengths = [len(a) for a in r.values()] + [len(a) for a in o.values()]
            n = max(lengths) if lengths else 0

            def _make(r=r, o=o):
                def _apply(local_t: int) -> None:
                    for name, arr in r.items():
                        self.set_qpos(name, arr[min(local_t, len(arr) - 1)])
                    for name, arr in o.items():
                        self.set_pose(name, arr[min(local_t, len(arr) - 1)])
                return _apply

            applies.append(_make())
            offsets.append(total)
            total += n

        def _apply(t: int) -> None:
            for start, fn, nxt in zip(offsets, applies, offsets[1:] + [total]):
                if t < nxt:
                    fn(t - start)
                    return
            if applies:
                applies[-1](total - offsets[-1] - 1)

        self._install_timeline(Timeline(total, fps, _apply))
        return self

    def _install_timeline(self, timeline: Timeline) -> None:
        self._timeline = timeline
        self._on_timeline_set(timeline)
        self.goto_frame(0)

    def goto_frame(self, t: int) -> None:
        """Clamp ``t`` to the timeline, apply it, and let the backend redraw."""
        if self._timeline is None:
            return
        t = int(np.clip(t, 0, max(self._timeline.num_frames - 1, 0)))
        self._frame = t
        self._timeline.apply(t)
        self._render_frame()

    @property
    def num_frames(self) -> int:
        return self._timeline.num_frames if self._timeline else 0

    @property
    def fps(self) -> float:
        return self._timeline.fps if self._timeline else 30.0

    @property
    def frame(self) -> int:
        return self._frame

    # ------------------------------------------------------------------ #
    # Backend hooks (optional overrides)
    # ------------------------------------------------------------------ #
    def _on_timeline_set(self, timeline: Timeline) -> None:
        """Called after a new timeline is installed (e.g. resize a GUI slider)."""

    def _render_frame(self) -> None:
        """Called after a frame is applied (e.g. flush the viser server)."""

    # ------------------------------------------------------------------ #
    # Scene construction & mutation — implemented per backend
    # ------------------------------------------------------------------ #
    @abstractmethod
    def add_robot(self, name: str, urdf_path: str, pose: Optional[Pose] = None,
                  color: Color = None) -> object:
        """Load a URDF robot under ``name``. ``color`` (RGBA 0-1) applies to all
        links at load time; ``pose`` is the 4x4 root transform."""

    @abstractmethod
    def add_object(self, name: str, mesh, pose: Optional[Pose] = None,
                   color: Color = None) -> object:
        """Add a ``trimesh`` object under ``name``."""

    @abstractmethod
    def set_qpos(self, name: str, qpos: np.ndarray) -> None:
        """Set a robot's joint configuration."""

    @abstractmethod
    def set_pose(self, name: str, pose: Pose) -> None:
        """Set the 4x4 root pose of a robot or object."""

    @abstractmethod
    def change_color(self, name: str, color: Color,
                     links: Optional[Sequence[str]] = None) -> None:
        """Recolor a robot or object. ``color`` is RGB/RGBA 0-1. For robots,
        ``links=None`` recolors every link; pass a list to target specific ones."""

    @abstractmethod
    def export_video(self, path: str, fps: Optional[float] = None) -> str:
        """Render the installed timeline to a video file and return its path."""
