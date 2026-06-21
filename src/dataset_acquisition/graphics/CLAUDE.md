# CLAUDE.md — src/dataset_acquisition/graphics

## Purpose
Capture sharp images sweeping camera exposure × gain, at one xarm pose or across trajectory waypoints. Free-run cameras (syncMode=False), no CaptureSession.

## Files
- `image_capture.py` — `get_arm("xarm")` + `remote_camera_controller`. Resets to `q_deg[pose_idx]`, sleeps `--settle`, snapshots arm (`qpos.npy`, `ee_pose.npy`), then full **cartesian product** of `--exposures` × `--gains`; each cell: `camera.start("image", False, cell_rel, exposure_time, gain)` → `camera.stop()`. Reorganizes into `by_serial/`.
- `image_traj.py` — `XArmController(**network_info["xarm"]["param"])` + `remote_camera_controller`. Iterates waypoints `range(start_idx, end_idx, step)`; per waypoint `controller.move(rad, is_servo=False)` then **1:1 paired** `(exp,gain)` cells, then `controller.get_data()` → `robot.npy`/`qpos.npy`/`eef.npy`. Saves camparam at top, reorganizes into `by_serial/`.

## paradex modules used
- `paradex.io.camera_system.remote_camera_controller.remote_camera_controller`
- `paradex.io.robot_controller.get_arm` (image_capture) / `paradex.io.robot_controller.xarm_controller.XArmController` (image_traj)
- `paradex.utils.path.shared_dir`, `paradex.utils.file_io.remove_home`, `paradex.utils.system.network_info`
- `paradex.calibration.utils.save_current_camparam` (image_traj only)

## Data flow & IO
- Cell path passed to `camera.start` is RELATIVE: image_capture uses `shared_data/<rel_base>/exp..._gain.../raw`; image_traj uses `remove_home(cell_abs)`.
- image_capture out: `capture/graphics/sharp_grid/<name>/<ts>/exp{e}_gain{g}/raw/images/<serial>.png`.
- image_traj out: `capture/graphics/sharp_traj/<name>/<ts>/q{i:04d}/exp{e}_gain{g}/images/<serial>.png`.
- Both build `by_serial/<serial>/...png` via `os.link` (falls back to `shutil.copyfile` on OSError).

## When working here
- `gain:g` formatting drops trailing zeros (e.g. `gain0` not `gain0.0`); keep consistent if changing paths.
- Negative `--pose_idx` is allowed (indexes from end).

## Gotchas
- These scripts intentionally do NOT use `CaptureSession` and have NO sync generator / timestamp monitor.
- `image_traj.py` raises `SystemExit` if `len(gains) != len(exposures)`.
- Defaults differ from image_capture: image_capture gains are a 5-element list (grid), image_traj/motion_blur use 4 paired gains.
- See [`motion_blur/CLAUDE.md`](motion_blur/CLAUDE.md) for the synced video sibling.
