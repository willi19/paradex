# Visualizer Validation

Smoke-test for the Viser-based 3D web viewer and robot URDF loading.

## Scripts
| File | Purpose |
|------|---------|
| `franka.py` | Launches a `ViserViewer`, loads the Franka arm URDF, and starts the web viewer. |

## Usage
```bash
python src/validate/visualizer/franka.py
```

No hardware required (Franka here is rendered from its URDF via Pinocchio/yourdfpy, not a physical arm). After launch, open the Viser URL printed in the console in a browser.

## What it validates
- `get_robot_urdf_path("franka")` resolves and the URDF loads without error.
- The Viser web server starts and serves a 3D scene.
- The Franka model renders in the browser viewer.

## Related
- [`paradex/visualization/visualizer/viser.py`](../../../paradex/visualization/visualizer/viser.py) — `ViserViewer` (`add_robot`, `start_viewer`).
- [`paradex/utils/file_io.py`](../../../paradex/utils/file_io.py) — `get_robot_urdf_path`.
- See `docs/visualization.md` (Claude memory) for the full `ViserViewer` API.
