# Visualization Scene API — Internals (for agents editing the scene code)

Read this before changing `paradex/visualization/scene/`. If you're only *calling*
the API, read [`usage.md`](usage.md) instead.

## Why this exists

The old design tangled three things into `ViserViewer`:

- **Scene state** (robots/objects), **playback** (slider + loop), and **rendering**
  all lived in one ~1100-line class, viser-only.
- Color had a real footgun: `ViserViewer.change_color(name, color, name_list)` vs
  the inner `ViserRobotModule.change_color(name_list, color)` — **swapped arg order**.
- `add_traj` required a 3-call ritual (`add_player` + `add_traj` + `update_scene(0)`)
  and silently concatenated trajectories in time, so callers hand-rolled their own
  players (see AutoDex `Visualization/paper/**/recorder_*.py`).
- The open3d path (`Open3DVideoRenderer`) was rigid (1 robot + 1 object fixed at
  construction) and unused; ~20 scripts hand-rolled raw `OffscreenRenderer` instead.

The `scene/` package factors these apart so **one call surface drives two backends**.

## File map

```
paradex/visualization/scene/
  base.py            # Scene (ABC) + Timeline + normalize_color  — backend-agnostic
  viser_backend.py   # ViserScene  — interactive browser + player GUI
  open3d_backend.py  # Open3DScene — headless OffscreenRenderer → images/mp4
  __init__.py        # lazy exports (backends imported on first attribute access)
```

Depends on the reusable, backend-neutral pieces already in the tree:
- `paradex.visualization.robot.RobotModule` — URDF → trimesh + FK (both backends).
- `paradex.visualization.visualizer.viser.ViserRobotModule` — viser mesh loader
  (reused by `ViserScene`; not reimplemented).

## The contract: what `base.Scene` owns vs delegates

**Owned (concrete, do not duplicate in backends):**
- `animate(robot=, object=, num_frames=, fps=)` — polymorphic:
  - `num_frames` set and no arrays → returns a **decorator** (callback form).
  - otherwise → declarative; builds an `apply(t)` that indexes arrays and calls
    `set_qpos` / `set_pose`.
- `sequence(clips, fps=)` — concatenated timeline; the *only* place time-append
  behavior lives, and it's opt-in.
- `goto_frame(t)` — clamps `t` to `[0, num_frames-1]`, calls `timeline.apply(t)`,
  then the `_render_frame()` hook.
- `normalize_color(color)` → `(rgb, opacity|None)`. `opacity=None` means "leave
  existing opacity alone" (don't force 1.0). RGB/RGBA 0-1 only.

**Delegated (each backend implements these `@abstractmethod`s):**
`add_robot`, `add_object`, `set_qpos`, `set_pose`, `change_color`, `export_video`.

**Optional hooks a backend may override:**
- `_on_timeline_set(timeline)` — e.g. resize the viser slider to `num_frames`.
- `_render_frame()` — e.g. `server.flush()` (viser) / no-op (open3d).

Registries `self._robots` / `self._objects` are `name -> backend handle` dicts;
base only reads their **keys** (and calls the abstract setters), never the values,
so backends are free to store whatever handle they need.

## Timeline model

`Timeline(num_frames, fps, apply)` — `apply(t)` mutates the scene to frame `t`.
Declarative and callback forms both compile down to one `Timeline`; there is a
single playback code path. This is what lets the *same* timeline drive an
interactive slider (viser) and a frame-by-frame mp4 loop (open3d).

## ViserScene specifics

- Player GUI built once in `_build_player()`: `Frame` slider, `Playing` checkbox,
  `FPS` slider. Slider `on_update` calls `goto_frame` (only when not auto-playing,
  to avoid feedback with the advance loop).
- `show(block=True)` runs the advance loop; `block=False` spawns a daemon thread.
  `_advance()` steps the frame when playing and sleeps `1/fps`.
- `set_pose` on a robot moves `_visual_root_frame`; on an object moves its frame
  handle. Robot root pose is cached in `_robot_pose`.
- `change_color` forwards to `ViserRobotModule.change_color(links, rgba)` for
  robots (note: that legacy method still takes `(name_list, color)` — we adapt at
  the call site so the **public** order stays `name, color, links`).
- `export_video` / `capture_png` render **through a connected browser client**
  (`client.get_render`). Headless → use Open3DScene.

## Open3DScene specifics

- No live scene graph updates: `_rebuild()` clears geometry and re-adds every
  entity from the registry each frame (o3d geometry is static). Robots are
  `update_cfg`'d then `get_robot_mesh()` (combined) → transformed by root pose →
  `_to_o3d` → `add_geometry` with a `MaterialRecord(base_color=rgba)`.
- Camera: `set_camera(eye, center, up, fov)` (look-at) or `set_intrinsic(K, E)`
  (matches the raw scripts' `setup_camera(K, extrinsic, W, H)`). Applied per render.
- `render_frame(t)` → RGB uint8 array; `export_video` loops frames into imageio.
- Per-link recolor (`links=`) is **not** honored here yet — the combined robot
  mesh takes one material. If you need per-link colors in open3d, render links
  separately in `_rebuild()` (get_link_mesh per link) and give each its own
  material; wire `change_color(..., links=)` to a per-link color dict.

## Invariants / gotchas when editing

- Keep the **public** color/arg conventions identical across backends: RGBA 0-1,
  `change_color(name, color, links=None)`, `add_*(name, ..., color=None)`. The
  whole point is that a script is backend-swappable.
- `animate()`'s dual return type (Scene vs decorator) is intentional; don't
  "simplify" it into two methods without updating `usage.md` and call sites.
- Backends must tolerate entities absent from a timeline (they hold current state)
  — don't assume every robot/object appears in `apply(t)`.
- Lazy imports in `__init__.py` are deliberate: never make importing `scene`
  require both `viser` and `open3d`.

## Absorbed from call sites

Features that ~40–100 subclasses hand-copied, now folded into `ViserScene`
(surveyed across AutoDex / CORL_2026_latex_prev / RSS_2026):

- `add_appearance_panel()` — the `add_appearance_gui` + `_update_*_appearance`
  cluster (41 files). Generalized to iterate all registered robots/objects.
- `add_capture_panel()` / `record_flythrough()` / `capture_turntable()` — the
  `add_video_capture_gui` + `_record_interpolated_video` + turntable cluster.
- `add_view_io_panel()` — the `add_view_save_gui` cluster.
- `clear()` / `remove()` / `set_opacity()` / `set_visible()` — the
  `clear_scene` / visibility-toggle primitives.

Deliberately **not** absorbed (app-specific, would couple the library to grasp
files / planners / scene_cfg): domain loaders (`load_scene`, `_load_current_grasp`,
`_load_grasps`), dataset-browser dropdown selectors (`_on_scene_type_change` &c.),
and cinematic sequences (`_play_animation`, `add_animation_gui`).

## Not yet done (follow-ups)

- Migrate AutoDex / CORL_2026_latex_prev / RSS_2026 call sites, then delete the
  legacy `visualizer/viser.py::ViserViewer` and `visualizer/open3d_viewer.py`.
- Per-link colors in the open3d backend (see above).
- `capture_turntable` uses `client.camera.look_at`; verify against the installed
  viser version (older versions may need manual wxyz from a look-at matrix).
