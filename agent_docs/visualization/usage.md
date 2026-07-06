# Visualization Scene API — How to Use (for humans & agents)

Read this before writing code that shows robots/objects in viser or renders a
figure/video with open3d. It replaces the old `ViserViewer` / `Open3DVideoRenderer`
pattern, where every script re-invented its own player, color handling, and
render loop.

> **Editing the scene code itself** (base class, backends, timeline)? Read
> [`internals.md`](internals.md) instead.

## TL;DR — one API, two backends

```python
from paradex.visualization.scene import ViserScene   # interactive, in browser
# from paradex.visualization.scene import Open3DScene # headless, renders mp4

scene = ViserScene(port=8080)
scene.add_robot("arm", "/path/xarm_allegro.urdf", color=(0.2, 0.6, 1.0, 1.0))
scene.add_object("cup", cup_trimesh, pose=cup_T, color=(0.8, 0.8, 0.8, 0.4))

scene.animate(robot={"arm": qpos}, object={"cup": cup_poses}, fps=30)
scene.show()   # opens the player; drag the Frame slider or leave "Playing" on
```

The **same three lines** (`add_robot` / `add_object` / `animate`) drive an
`Open3DScene` — swap the constructor and call `export_video()` instead of
`show()`. No second code path for "make the paper video".

Headless render (no browser) — same scene code, open3d backend:

```python
from paradex.visualization.scene import Open3DScene

scene = Open3DScene(width=1280, height=720)
scene.add_robot("arm", "/path/xarm_allegro.urdf", color=(0.2, 0.6, 1.0, 1.0))
scene.add_object("cup", cup_trimesh, pose=cup_T, color=(0.8, 0.8, 0.8, 1.0))
scene.set_camera(eye=(1.2, 1.2, 0.8), center=(0, 0, 0.15))
scene.animate(robot={"arm": qpos}, object={"cup": cup_poses}, fps=30)
scene.export_video("out.mp4")          # or: rgb = scene.render_frame(t)
```

## The mental model: Scene vs Timeline

Two separate concerns — keep them separate:

1. **Scene** = *what exists*. Named robots and objects. Build once with
   `add_robot(name, ...)` / `add_object(name, ...)`.
2. **Timeline** = *how they move*. One call: `animate(...)`. The base class owns
   the player, frame clamping, and video export. You never write a slider or a
   `while` loop again.

This is the fix for the old `add_traj` pain: no `add_player()` + `add_traj()` +
`update_scene(0)` ritual, and no accidental time-concatenation.

## Colors — one convention everywhere

**RGB or RGBA float in 0.0–1.0.** `None` = keep the mesh's native color.

```python
scene.add_robot("arm", urdf, color=(0.2, 0.6, 1.0))        # opaque
scene.add_object("cup", mesh, color=(0.8, 0.8, 0.8, 0.4))  # 40% opacity
scene.change_color("arm", (1.0, 0.0, 0.0, 1.0))            # recolor all links
scene.change_color("arm", (1.0, 0.0, 0.0), links=["palm"]) # one link only
```

- Pass **RGBA (4 values)** when you want to set opacity. RGB (3 values) leaves
  the existing opacity untouched — it is **not** forced to 1.0.
- `change_color(name, color, links=None)` — same argument order for robots and
  objects. `links=None` recolors every robot link.

> This replaces the old footgun where `ViserViewer.change_color(name, color, name_list)`
> and the inner `ViserRobotModule.change_color(name_list, color)` had **swapped**
> argument orders. There is now one order: **name, color, links**.

## Animating — two forms of `animate()`

### Declarative (90% of cases)

Per-entity arrays sharing one timeline; everything plays **together**.

```python
scene.animate(
    robot={"arm": qpos},          # (T, ndof)
    object={"cup": cup_poses},    # (T, 4, 4)
    fps=30,
)
```

- Names must already be added. Entities you don't mention hold their current pose.
- `num_frames` = the longest array; shorter arrays clamp on their last value.
- Calling `animate()` again **replaces** the timeline.

### Callback (procedural / external per-frame logic)

Use when the frame is computed, not a plain array — this is what absorbs the
"I'll just build my own player" urge.

```python
@scene.animate(num_frames=T, fps=30)
def frame(t):
    scene.set_qpos("arm", plan.qpos[t])
    scene.set_pose("cup", tracker.pose_at(t))
    if t == grasp_frame:
        scene.change_color("cup", (0.0, 1.0, 0.0, 1.0))
```

### Sequencing clips (explicit — never automatic)

To play clips back-to-back (the old `add_traj`-appends behavior), ask for it:

```python
scene.sequence([
    {"robot": {"arm": approach_qpos}},
    {"robot": {"arm": lift_qpos}},
], fps=30)
```

## Viewing vs rendering

| | `ViserScene` | `Open3DScene` |
|---|---|---|
| Construct | `ViserScene(port=8080)` | `Open3DScene(width, height)` |
| See it | `scene.show()` (browser player) | — (headless) |
| Camera | orbit in browser | `scene.set_camera(eye, center, up)` or `set_intrinsic(K, E)` |
| Video | `scene.export_video("out.mp4")` * | `scene.export_video("out.mp4")` |
| Still | `scene.capture_png("f.png")` * | `scene.render_frame(t)` → RGB array |

\* viser export/capture needs a **connected browser client** (it renders through
one). For fully headless rendering use `Open3DScene`.

## Manual (non-animated) updates

You don't need a timeline to just move things:

```python
scene.set_qpos("arm", qpos_now)
scene.set_pose("cup", cup_T_now)
scene.set_visible("cup", False)   # viser only
```

## Built-in GUI panels & capture tools (ViserScene)

These replace features that ~40 recorder/viewer scripts each hand-copied. Add
them *after* building the scene:

```python
scene.add_appearance_panel()   # per-robot/object Color + Opacity + Visible controls
scene.add_capture_panel()      # Set Start/End View → record fly-through mp4 + PNG
scene.add_view_io_panel()      # save / load camera view as JSON
```

- **`add_appearance_panel(folder="Appearance")`** — auto-iterates every robot and
  object in the scene and generates a color picker + opacity slider + visibility
  checkbox for each, wired to `change_color` / `set_visible`. No per-app copy.
- **`add_capture_panel(folder="Capture")`** — GUI to set a start & end camera view
  and render an interpolated fly-through video (position lerp + rotation slerp),
  plus a still-capture button. Needs a connected browser client.
- **`record_flythrough(path)`** / **`capture_turntable(path, center, radius, ...)`**
  — call these directly (no GUI) to script a fly-through or a 360° orbit render.
- **`add_view_io_panel()`** — save/load the current camera pose to JSON (handy for
  reproducing the same figure framing across runs).

Scene primitives shared by both backends:

```python
scene.set_visible("cup", False)
scene.set_opacity("cup", 0.4)     # objects; robots take RGBA via change_color
scene.remove("cup")
scene.clear()                     # drop all entities + timeline
```

## Common gotchas

- **Name must exist before you animate/recolor it.** `add_robot`/`add_object`
  first; otherwise `KeyError`.
- **RGB vs RGBA.** Want transparency? Pass 4 values.
- **viser video needs a browser open.** Headless CI → use `Open3DScene`.
- **open3d not interactive.** It has no player; you `animate()` then
  `export_video()` / `render_frame(t)`.
- Deps are optional: importing `paradex.visualization.scene` never forces both
  `viser` and `open3d` — each backend imports lazily on first use.

## Migrating from the old `ViserViewer`

| Old | New |
|-----|-----|
| `ViserViewer(port_number=8080)` | `ViserScene(port=8080)` |
| `add_robot(name, urdf, pose)` | `add_robot(name, urdf, pose, color=)` |
| `change_color(name, color, name_list)` | `change_color(name, color, links=)` |
| `add_player()` + `add_traj(name, {...})` + `update_scene(0)` | `animate(robot={...}, object={...}, fps=)` |
| custom slider / `while True` loop | `show()` (or the callback form of `animate`) |
| `render_full_video()` | `export_video(path)` |
| `capture_scene_png(path)` | `capture_png(path)` |

The old `visualizer/viser.py` (`ViserViewer`) and `visualizer/open3d_viewer.py`
(`Open3DVideoRenderer`) still exist for now; call sites migrate to `scene/`
before they are removed.
