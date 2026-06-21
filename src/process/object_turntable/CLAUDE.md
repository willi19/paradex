# CLAUDE.md — src/process/object_turntable

## Purpose
Turntable object scan → 3D reconstruction. Extract frames, triangulate charuco for per-frame turntable rotation, SAM3-segment object, build COLMAP with known poses.

## Files
- `extract_video.py` — `process_video(video_path)`: cv2 reads each frame to `images/<cam>/frame_NNNNNN.jpg` (q95). Skips checkerboard-sentinel dropped frames (30×30 corner pattern). mp.Pool over all videos; warns if a `videos/` dir != 24 files.
- `extract_charuco.py` — `process_frame((idx, root_dir))`: load_camparam, read all serials' `frame_NNNNNN.jpg` (needs ≥5), `ImageDict.triangulate_charuco`, save first board's `checkerIDs`/`checkerCorner` to `charuco_3d/`. `process_task` enumerates frames; runs frames in mp.Pool but tasks serially.
- `get_rotation.py` — base board = `get_board_cor()['2']`. Per `*_cor.npy`: `find_common_indices(base_id, charuco_id)` (need ≥4), `SOLVE_XA_B(base_pts, cur_pts)` → 4×4 → `rotation/<frame>.npy`. Wipes `rotation/` first.
- `extract_mask_sam3.py` — SAM3 video predictor. `process_video`: copies sampled frames (`idx % 40 == 1`) to `selected/<serial>/`, starts session, adds text prompt "object on the checkerboard, excluding checkerboard" until an obj is detected, `propagate_in_video`, writes `masks/<serial>/*.png` + `masked_images/<serial>/*.jpg`. `load_mask` loops serials. Bottom loops all `<obj>/<index>` with error_log.
- `generate_colmap.py` — `generate_db`: COLMAPDatabase, `add_camera` (PINHOLE fx,fy,cx,cy from `intrinsics_undistort`), `add_image` (pose = extrinsic ∘ rotation[frame], qvec from `Rotation.from_matrix`). `run_colmap`: `pycolmap.extract_features` with `mask_path`, `match_sequential` (overlap 10, quadratic). `export_initial_poses` writes `sparse/tmp_initial`. `run_point_triangulator`: `colmap point_triangulator` CLI. Bottom loops `['big_green_spray']` only.

## paradex modules used
`paradex.image.image_dict.ImageDict`, `paradex.image.aruco` (get_board_cor, find_common_indices, detect_charuco, merge_charuco_detection), `paradex.image.undistort.undistort_img`, `paradex.transforms.conversion.SOLVE_XA_B`, `paradex.calibration.utils.load_camparam`, `paradex.calibration.colmap.COLMAPDatabase`, `paradex.utils.path` (home_path, shared_dir). External: `sam3`, `pycolmap`, `colmap` CLI.

## Data flow & IO
Root: `home_path/paradex_download/capture/object_turntable/<obj>/<index>/`.
videos/ → images/ → charuco_3d/ → rotation/ ; images/ → selected/+masks/+masked_images/ ; masked_images/+rotation/ → colmap/.
COLMAP pose: `T = [extrinsic;0001] @ rotation[frame]`, qvec = scipy quat (x,y,z,w order as stored).

## When working here
- Stage order: extract_video → extract_charuco → get_rotation → extract_mask_sam3 → generate_colmap.
- Many sequences gated on exactly 24 cameras; mismatches are skipped/warned.
- `qvec` in `add_image` is built as `[x,y,z,w]` from scipy `as_quat()` then passed positionally — verify ordering matches the COLMAP DB convention if poses look wrong.

## Gotchas
- `extract_mask_sam3.py` line 159 is a bare Korean comment (`이건 realdex거든`) inside `load_mask` — it is **not valid Python** and will raise SyntaxError; fix/remove before running.
- `extract_charuco.process_frame` references undefined `obj_name`/`index` in a print on the no-charuco branch (NameError if hit).
- `generate_colmap.py` only processes `['big_green_spray']` — edit the loop for other objects.
- SAM3 mask sampling uses `idx % 40 == 1`; `detect_outlier.py` (in check/) additionally requires `idx >= 160` and deletes others — keep sampling rules consistent.
- `paradex.dataset_acqusition` misspelled (missing 'i') — intentional (not imported here but used in sibling pipelines).
- `deprecated/` and `check/check_colmap copy.py` are stale leftovers.
