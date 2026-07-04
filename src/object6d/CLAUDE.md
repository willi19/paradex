# CLAUDE.md — src/object6d

## Purpose
Multi-view image capture + 6D object pose estimation, plus C2R (camera-to-robot) validation. Captures synced multi-camera images, undistorts, matches each view to pre-built object templates, and optimizes a multi-view pose.

## Files
- `capture.py` — main pipeline. If `--image_path` given, reuses an existing folder; else `remote_camera_controller("image_main.py")` captures to `shared_data/<save_path>/raw`, saves cam params, undistorts via `ImageDict`. Builds `Scene(scene_path=...)`, pulls `cam2intr / cam2extr_4X4 / proj_matrix`, calls `scene.get_batched_renderer(...)`, builds `MatchProcessor(obj_names=...)` and `CaptureAndOptimizePipeline(...)`, then `pipeline.run(img_bucket=image_dict, idx=0)`. Saves `obj_T_multiview_matching.pkl` via `torch.save`.
- `image_remote.py` — interactive loop; `c` captures a timestamped set (cam param + C2R + raw + undistort into `<save_path>/<ts>/undistort`), `q` quits.
- `capture_template.py` — template variant of `image_remote.py` (saves cam param only, undistorts in place, no C2R). `parents[3]` sys.path hack. Treat as config/starting point.
- `undistort_template.py` — `undistort(image_path, save_camparam, save_c2r)` over an `ImageDict`; CLI `--save_path`. Note commented-out batch loop over obj/scene.
- `loftr_client.py` — capture-PC streamer. `MultiCameraReader` → resize 1/8 → JPEG q85 → `DataPublisher(port=1234)`; `CommandReceiver(port=6890)` listens for `exit`. No argparse.
- `validate_c2r.py` — render robot at zero qpos, overlay mask via `BatchRenderer` + `overlay_mask`, write `<scene>/c2r_validation/{cam}.jpg` + `grid.jpg`. (See `validate/validate_c2r_simple.py` for the `ImageDict.project_mesh` variant that also loads arm/hand qpos.)

## paradex modules used
- `paradex.io.camera_system.remote_camera_controller.remote_camera_controller`
- `paradex.io.camera_system.camera_reader.MultiCameraReader` (loftr_client)
- `paradex.io.capture_pc.data_sender.DataPublisher`, `...command_sender.CommandReceiver` (loftr_client)
- `paradex.image.image_dict.ImageDict` (`from_path`, `undistort`, `project_mesh`, `set_camparam`)
- `paradex.image.projection.BatchRenderer`, `paradex.image.overlay.overlay_mask`, `paradex.image.grid.make_image_grid`
- `paradex.calibration.utils` — `save_current_camparam`, `save_current_C2R`, `load_camparam`
- `paradex.visualization.robot.RobotModule`
- `paradex.utils.keyboard_listener.listen_keyboard`, `paradex.utils.path.{shared_dir,rsc_path}`
- External (NOT in paradex): `_object_6d_tracking.run.run_multiview_matching26d.CaptureAndOptimizePipeline`, `.multiview_utils.template_matcher.MatchProcessor`, `.pose_utils.scene.Scene`

## Data flow & IO
- Capture target: `shared_data/<save_path>/raw` (or `.../<timestamp>/raw`); cam params + `C2R.npy` written alongside by `save_current_*`.
- Undistort: `ImageDict.from_path(...).undistort(...)` → `images/` (or `undistort/images/`).
- Pose out: `<scene_path>/obj_T_multiview_matching.pkl` (torch).
- C2R validation out: `<scene_path>/c2r_validation/` (per-cam overlay + grid).

## When working here
- `capture.py --obj_names` is required; either `--image_path` or `--save_path` must be set (asserted).
- `loftr_client.py` runs on capture PCs; ports are hardcoded in source (1234 data, 6890 command).
- Do not "fix" the spelled-out `dataset_acqusition` import elsewhere; here `capture.py` imports `paradex.image`/`paradex.calibration`.

## Gotchas
- `capture.py` and `validate_c2r.py` rely on a sibling `_object_6d_tracking` / external deps not vendored in this repo — running them needs that package installed/on path.
- `*_template.py` files contain leftover commented code and sys.path hacks (`parents[2]`/`parents[3]`); they are templates, not stable APIs.
- Two C2R validators exist: this dir's `validate_c2r.py` (zero qpos, `BatchRenderer`) and `validate/validate_c2r_simple.py` (loads real arm/hand qpos, `ImageDict.project_mesh`, applies `c2r` to the mesh). They diverge — pick deliberately.
- DEVICE is CUDA if available; pose pipeline expects a GPU.
