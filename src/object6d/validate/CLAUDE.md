# CLAUDE.md — src/object6d/validate

## Purpose
Validation/sanity scripts for the object6d rig: a combined robot+image capture test, and a C2R calibration overlay validator that uses real robot qpos.

## Files
- `capture_test.py` — interactive loop. `CaptureSession(camera=False, arm, hand, hand_ip=True)` records robot data; `remote_camera_controller("image_main.py")` captures images. Keys: `c`=save_event (start take), `s`=stop_event, `q`=exit_event. On `c`: timestamped `save_path=<name>/<ts>`, `cs.start`, `save_current_camparam`, `save_current_C2R`, `rcc.start("image", False, 'shared_data/<save_path>/raw')`, `rcc.stop`, `cs.stop`. `finally`: `cs.end()` + `rcc.end()`.
- `validate_c2r_simple.py` — C2R validator. Loads `load_camparam`, `C2R.npy`. Uses undistorted images if `<scene>/undistort/images` exists, else undistorts raw via `ImageDict` (`set_camparam` → `undistort(save_path=undistort_path)`). Builds `full_qpos = zeros(12)` (6 arm + 6 hand), overrides from `raw/arm/position.npy` and `raw/hand/action.npy` (or `position.npy`) with `parse_inspire` for hand → radians. Loads `<arm>_<hand>.urdf` via `RobotModule`, `update_cfg`, `get_robot_mesh`, `mesh.apply_transform(c2r)`, then `img_dict.project_mesh(mesh)`. Saves overlays + `grid.jpg` to `<scene>/c2r_validation/`. Has a local inlined `make_image_grid` (not the `paradex.image.grid` one).

## paradex modules used
- `paradex.dataset_acqusition.capture.CaptureSession` (capture_test)
- `paradex.io.camera_system.remote_camera_controller.remote_camera_controller`
- `paradex.utils.keyboard_listener.listen_keyboard`, `paradex.utils.path.{shared_dir,rsc_path}`
- `paradex.calibration.utils.{save_current_camparam,save_current_C2R,load_camparam}`
- `paradex.image.image_dict.ImageDict`, `paradex.image.projection.BatchRenderer`, `paradex.image.overlay.overlay_mask`
- `paradex.robot.inspire.parse_inspire`
- `paradex.visualization.robot.RobotModule`

## Data flow & IO
- `capture_test.py`: out → `shared_data/<name>/<ts>/{raw,arm,hand,...}` + cam params + `C2R.npy`.
- `validate_c2r_simple.py`: in → `<scene>/C2R.npy`, cam params, `undistort/images` or raw, `raw/arm/position.npy`, `raw/hand/{action,position}.npy`. out → `<scene>/c2r_validation/{cam}.jpg` + `grid.jpg`.

## When working here
- `capture_test.py` `--name` sets the dataset folder; `--arm`/`--hand` default `None` (image-only if omitted). `hand_ip=True` is passed to `CaptureSession`.
- `validate_c2r_simple.py` applies `c2r` to the mesh (world←robot) and projects with `ImageDict`, unlike the parent dir's `validate_c2r.py` which folds `c2r` into the extrinsic and uses `BatchRenderer`.

## Gotchas
- The module docstring/Usage in `validate_c2r_simple.py` points at a stale path (`src/validate/robot/validate_c2r.py`); ignore it.
- `make_image_grid` is duplicated inline here (with a nested `import cv2`); the canonical one is `paradex.image.grid.make_image_grid`.
- Inspire hand units are 0-2000 and MUST go through `parse_inspire` before use as radians.
- `parents[...]` sys.path hacks present in parent-dir scripts are absent here; these import paradex directly.
