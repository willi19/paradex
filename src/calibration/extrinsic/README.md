# Extrinsic Calibration

Multi-camera extrinsic calibration. Operators move a Charuco board through the capture volume; each Capture PC detects Charuco corners locally and streams them to the Main PC, which stores per-view 2D corners and then solves global camera poses with COLMAP (with metric scale from known board geometry).

## Scripts
| File | Purpose |
|------|---------|
| `capture.py` | **Main PC** orchestrator. Launches `client.py` on Capture PCs, shows a merged live stream with detected (green) and accumulated saved (red) corners, and saves a capture on key press. |
| `client.py` | **Capture PC** daemon (launched by `capture.py`). Reads cameras, runs Charuco detection, publishes downscaled JPEG + corners to the Main PC, and saves full-res corners/ids/images when a save command arrives. |
| `calculate.py` | **Main PC** offline solver. Builds a COLMAP database from saved 2D corners, runs incremental mapping, undistorts, triangulates, does a refine pass, rescales to metric, and writes `intrinsics.json` / `extrinsics.json`. |

## Usage
Prerequisite: intrinsic calibration must already exist (`calculate.py` calls `load_current_intrinsic()`).

1. On the **Main PC**, run the orchestrator (it auto-launches `client.py` on every Capture PC):
   ```bash
   python src/calibration/extrinsic/capture.py
   ```
   In the "Merged Stream" window:
   - move the Charuco board around the volume;
   - press **`c`** to capture the current frame (all cameras save full-res corners/ids/images; "Saving..." overlays until all cameras confirm);
   - press **`q`** to quit and stop the clients.
2. On the **Main PC**, solve:
   ```bash
   python src/calibration/extrinsic/calculate.py            # uses latest extrinsic capture dir
   python src/calibration/extrinsic/calculate.py --name 20250101_120000
   ```
   `calculate.py` runs the full pipeline twice: an initial solve + reprojection-error-based refine (keeps corners with reproj err < 2px into a new timestamped dir), then a final solve on the refined set.

## Inputs & Outputs
- Capture dir: `~/shared_data/extrinsic/<filename>/<capture_idx>/` with `markers_2d/<serial>_corner.npy`, `markers_2d/<serial>_id.npy`, `images/<serial>.png`.
- COLMAP outputs under `<extrinsic>/<name>/<first_idx>/colmap/` (`database.db`, model files); `undistort/`, `debug/`, `kypt_3d_*.npy`, `outlier/`, `reproj_error.txt` written alongside.
- Final result: `~/shared_data/cam_param/<name>/intrinsics.json` and `extrinsics.json` (translations rescaled so the mean Charuco edge length = 0.06 m).

## Related
- `paradex/calibration/colmap.py` — `COLMAPDatabase`, `get_two_view_geometries`, `load_colmap_camparam`.
- `paradex/image/aruco.py` — `detect_charuco`, `merge_charuco_detection`, `draw_charuco`, `get_adjecent_ids`, `find_common_indices`.
- `paradex/image/image_dict.py` — `ImageDict` (undistort, triangulate, project, draw).
- `paradex/io/camera_system/{remote_camera_controller,camera_reader}.py`, `paradex/io/capture_pc/{ssh,data_sender,command_sender}.py`.
- `paradex.calibration.utils` — `extrinsic_dir`, `cam_param_dir`, `load_current_intrinsic`.
- Next step: [`../handeye/`](../handeye/).
