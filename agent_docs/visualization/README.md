# Visualization

Backend-agnostic **scene + animation API** for showing robots/objects in 3D.
One call surface, two interchangeable backends:

- **`ViserScene`** — interactive browser viewer with a built-in playback GUI.
- **`Open3DScene`** — headless offscreen renderer for figures / mp4.

The same script drives both — swap the constructor, call `show()` (viser) or
`export_video()` (open3d).

```python
from paradex.visualization.scene import ViserScene      # or Open3DScene

scene = ViserScene(port=8080)
scene.add_robot("arm", urdf_path, color=(0.2, 0.6, 1.0, 1.0))
scene.add_object("cup", cup_mesh, pose=cup_T, color=(0.8, 0.8, 0.8, 0.4))
scene.animate(robot={"arm": qpos}, object={"cup": cup_poses}, fps=30)
scene.show()
```

## Which file do I read?

| You are… | Read |
|----------|------|
| **calling** the API (viewing, rendering, animating, coloring) | [`usage.md`](usage.md) |
| **editing** the scene code (base/backends/timeline) | [`internals.md`](internals.md) |

## Why this exists

It replaces the old `ViserViewer` / `Open3DVideoRenderer`, where every script
re-invented its own player, color handling, appearance GUI, and render loop
(~100 `ViserViewer` subclasses across AutoDex / CORL / RSS). The core idea:
**separate the Scene (what exists) from the Timeline (how it moves)**, and let
the library own playback, color, panels, and export.

## Code

- Library: [`paradex/visualization/scene/`](../../paradex/visualization/scene/)
  (`base.py`, `viser_backend.py`, `open3d_backend.py`).
- Reusable robot mesh/FK helper: `paradex/visualization/robot.py::RobotModule`.
- Legacy (`visualizer/viser.py::ViserViewer`, `visualizer/open3d_viewer.py`) is
  being phased out — new code should use `scene/`; existing call sites migrate
  before it's deleted.
