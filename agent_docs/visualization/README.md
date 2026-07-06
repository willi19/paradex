# Visualization

Backend-agnostic scene + animation API for showing robots/objects. One call
surface, two backends:

- **`ViserScene`** — interactive browser viewer with a built-in player.
- **`Open3DScene`** — headless offscreen renderer for figures / mp4.

```python
from paradex.visualization.scene import ViserScene
scene = ViserScene(port=8080)
scene.add_robot("arm", urdf, color=(0.2, 0.6, 1.0, 1.0))
scene.animate(robot={"arm": qpos}, fps=30)
scene.show()
```

- **Calling it?** → [`usage.md`](usage.md) (recipe, colors, animate forms, migration).
- **Editing the scene code?** → [`internals.md`](internals.md) (design, contract, invariants).

Code: [`paradex/visualization/scene/`](../../paradex/visualization/scene/).
Legacy (`visualizer/viser.py::ViserViewer`, `visualizer/open3d_viewer.py`) is
being phased out; new code should use `scene/`.
