# CLAUDE.md — src/calibration/extrinsic

## Purpose
Multi-camera extrinsic calibration: distributed Charuco capture (Main PC orchestrates, Capture PCs detect/stream) → COLMAP global pose solve with metric rescale.

## Files
- `capture.py` — Main PC. `run_script("python src/calibration/extrinsic/client.py")` to start clients; `remote_camera_controller("extrinsic_calibration")` + `DataCollector` (recv stream/detections) + `CommandSender`. Loop: decode JPEG previews, draw current corners (green) and accumulated saved corners (red, downscaled /8 mask), `merge_image` display. `c` → send `save` command (filename + capture_idx), accumulate saved mask; `q` → quit. No `--args`.
- `client.py` — Capture PC. `MultiCameraReader` + `detect_charuco`/`merge_charuco_detection`; `DataPublisher(port=1234)` sends downscaled JPEG ('image') + corners ('charuco_detection', /8 as float32); `CommandReceiver(port=6890)` with `exit`/`save` events. On `save`, threads `_save_camera_data` writing full-res `markers_2d/<serial>_{corner,id}.npy` + `images/<serial>.png` from event_info filename/capture_idx.
- `calculate.py` — Main PC offline. `run_calibration` (build COLMAP db from corners via `generate_db`/`parallel_processing`/`get_two_view_geometries`, `pycolmap.incremental_mapping`), `undistort`, `save_kypt_3d` (triangulate Charuco), `get_length` (board edge stats), `debug` (reproj error + refine: keep err<2px into new timestamped dir). `__main__`: solve → refine → re-solve, rescale translations by `0.06/mean_length`, write `cam_param/<name>/{intrinsics,extrinsics}.json`.

## paradex modules used
- `paradex.calibration.colmap` — DB + two-view geometry + `load_colmap_camparam`.
- `paradex.image.aruco` — detect/merge/draw Charuco, adjacency/common ids.
- `paradex.image.image_dict.ImageDict` — undistort, triangulate, project, draw.
- `paradex.io.camera_system.{remote_camera_controller,camera_reader}`; `paradex.io.capture_pc.{ssh,data_sender,command_sender}`.
- `paradex.calibration.utils` — `extrinsic_dir`, `cam_param_dir`, `load_current_intrinsic`.

## Data flow & IO
Capture PCs detect → stream to Main PC; on `c`, full-res saved to `~/shared_data/extrinsic/<filename>/<capture_idx>/markers_2d/*` + `images/*`. `calculate.py` reads those, runs COLMAP under `<name>/<first_idx>/colmap/`, writes final `~/shared_data/cam_param/<name>/{intrinsics,extrinsics}.json`. Refine pass creates a NEW timestamped extrinsic dir; final json `<name>` is the original, but model is read from the refined latest dir — see `__main__` carefully.

## When working here
- `capture.py` and `client.py` are tightly coupled by data types ('image' downscaled /8, 'charuco_detection' /8 float32) and ports (1234 publish, 6890 command). Change both together.
- Requires existing intrinsics (`load_current_intrinsic`). Run after intrinsic, before handeye.

## Gotchas
- Distributed: `capture.py` MUST run on Main PC; it spawns `client.py` on Capture PCs via SSH `run_script`.
- Corners are downscaled by 8 for streaming/preview only; saved npy are full-res.
- Metric scale assumes Charuco square edge = 0.06 m (`get_length` mean used as scale divisor).
- `reproj_error.txt` filename has a leading space (`" reproj_error.txt"`) — matches source, do not "fix" silently.
