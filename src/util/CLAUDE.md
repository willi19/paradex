# CLAUDE.md — src/util

## Purpose
Standalone operational utilities. Four unrelated subgroups; each has its own `CLAUDE.md`. Read the relevant subdir doc before editing.

## Subdirectories
- `marker/` — ChArUco board PDF generator (`generate_charuco.py`, no paradex deps). Outputs to `outputs/` in CWD.
- `register_object/` — `box.py` / `pringles.py`: build `marker_offset` `.npy` tables via `ImageDict` triangulation; output under `{shared_dir}/RSS2026_Mingi/marker_offset/<obj>/`.
- `robot/` — `merge_urdf.py` (arm+hand -> combined URDF via xacro), `visualize.py` (Open3D link/collision-sphere inspection, scratch script), `replay.py` (live arm qpos -> Viser). `get_bounding_sphere.py` and `replay_sim.py` are EMPTY stubs.
- `upload_video/` — distributed raw-video undistort+upload on `paradex.process`: `main.py` (main PC, `run_distributed`) and `worker.py` (capture PC, `discover`/`process` reusing `undistort_raw_video`).
- `camera_tuning/` — `live_tuner.py`: capture-PC interactive per-camera gain/exposure tuner (owns cameras via `load_camera`, OpenCV trackbars, saves to `camera.json`).

## paradex modules used (by subgroup)
- register_object: `paradex.image.image_dict.ImageDict`, `paradex.utils.path.shared_dir`
- robot: `paradex.robot.*` (RobotWrapper, urdf, curobo, utils), `paradex.io.robot_controller.get_arm`, `paradex.visualization.visualizer.viser.ViserViewer`, `paradex.geometry.coordinate.DEVICE2WRIST`
- upload_video: `paradex.process` (`Job`/`serve_jobs`/`run_distributed`), `paradex.video.raw_video_processor` (`get_raw_videopath_list`/`undistort_raw_video`)
- marker: none

## When working here
- Subgroups are independent — scope changes to one subdir.
- Several scripts hard-code names/paths (object names, `link4`/`xarm_allegro`, `RSS2026_Mingi`, ZMQ 1234); edit in source rather than expecting CLI args (only `merge_urdf.py` and `replay.py` take `--arm/--hand`).

## Gotchas
- Empty stub files: `robot/get_bounding_sphere.py`, `robot/replay_sim.py`.
- `upload_video/` was migrated to `paradex.process` (was Flask/SocketIO + `process.py`/`client.py`, now `main.py`/`worker.py`).
- Do not "fix" the `dataset_acqusition` typo elsewhere in the repo; not relevant here but a repo-wide convention.
