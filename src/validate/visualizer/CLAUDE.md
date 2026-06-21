# CLAUDE.md — src/validate/visualizer

## Purpose
Minimal smoke-test that the `ViserViewer` web 3D viewer launches and can load + render a robot URDF (Franka).

## Files
- `franka.py` — four lines: `a = ViserViewer()`, `a.add_robot("franka", get_robot_urdf_path("franka"))`, `a.start_viewer()`. Confirms viewer boot + URDF load.

## paradex modules used
- `paradex.visualization.visualizer.viser.ViserViewer` — `add_robot(name, urdf_path)`, `start_viewer()`.
- `paradex.utils.file_io.get_robot_urdf_path` — resolves URDF path by robot name.

## Data flow & IO
- No data files; no hardware. Franka is a URDF render (Pinocchio/yourdfpy), not the physical arm.
- Output is a Viser web server (URL printed to console).

## When working here
- Validation script; keep runnable as `python src/validate/visualizer/franka.py`.
- Do NOT edit the `.py` for doc tasks.

## Gotchas
- `start_viewer()` likely blocks/serves indefinitely — this is a manual visual check, not an automated test.
- Viewer conventions (from project memory): colors are 0.0–1.0 float RGB; quaternions are wxyz order. See `docs/visualization.md`.
