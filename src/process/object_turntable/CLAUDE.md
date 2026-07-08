# CLAUDE.md — src/process/object_turntable

## Purpose
Turntable object scan → 3D reconstruction. Extract frames, triangulate charuco for per-frame turntable rotation, SAM3-segment object, build COLMAP with known poses.

## Files
- `worker.py` — **framework entrypoint.** One `paradex.process` Job per `<obj>/<index>` scan; `process()` runs all 5 stages in order with per-stage progress, importing the stage modules lazily (so `discover()` works without torch/sam3/pycolmap). Single-machine, in-process (`num_workers=1`) because each stage spawns its own `mp.Pool` and the GPU stages need real tracebacks; skip when `colmap/` exists. Run: `python src/process/object_turntable/worker.py [obj ...]`. This replaces hand-running each stage script's `__main__` in sequence.
- `extract_video.py` — `process_video(video_path)`: cv2 reads each frame to `images/<cam>/frame_NNNNNN.jpg` (q95). Skips checkerboard-sentinel dropped frames (30×30 corner pattern). mp.Pool over all videos; warns if a `videos/` dir != 24 files.
- `extract_charuco.py` — `process_frame((idx, root_dir))`: load_camparam, read all serials' `frame_NNNNNN.jpg` (needs ≥5), `ImageDict.triangulate_charuco`, save first board's `checkerIDs`/`checkerCorner` to `charuco_3d/`. `process_task` enumerates frames; runs frames in mp.Pool but tasks serially.
- `get_rotation.py` — base board = `get_board_cor()['2']`. Per `*_cor.npy`: `find_common_indices(base_id, charuco_id)` (need ≥4), `SOLVE_XA_B(base_pts, cur_pts)` → 4×4 → `rotation/<frame>.npy`. Wipes `rotation/` first.
- `extract_mask_sam3.py` — SAM3 video predictor. `process_video`: copies sampled frames (`idx % 40 == 1`) to `selected/<serial>/`, starts session, adds text prompt "object on the checkerboard, excluding checkerboard" until an obj is detected, `propagate_in_video`, writes `masks/<serial>/*.png` + `masked_images/<serial>/*.jpg`. `load_mask(root_dir)` loops serials. Predictor is built lazily via `get_video_predictor()` (not at import); the `sam3` import is guarded; `main()`/`__main__` loops all `<obj>/<index>`.
- `generate_colmap.py` — `generate_db`: COLMAPDatabase, `add_camera` (PINHOLE fx,fy,cx,cy from `intrinsics_undistort`), `add_image` (pose = extrinsic ∘ rotation[frame], qvec from `Rotation.from_matrix`). `run_colmap`: `pycolmap.extract_features` with `mask_path`, `match_sequential` (overlap 10, quadratic). `export_initial_poses` writes `sparse/tmp_initial`. `run_point_triangulator`: `colmap point_triangulator` CLI. `colmap_demo(demo_path)` = db→reconstruct→triangulate for one scan; `main(obj_filter)`/`__main__` loops (default `['big_green_spray']`).

## paradex modules used
`paradex.image.image_dict.ImageDict`, `paradex.image.aruco` (get_board_cor, find_common_indices, detect_charuco, merge_charuco_detection), `paradex.image.undistort.undistort_img`, `paradex.transforms.conversion.SOLVE_XA_B`, `paradex.calibration.utils.load_camparam`, `paradex.calibration.colmap.COLMAPDatabase`, `paradex.utils.path` (home_path, shared_dir). External: `sam3`, `pycolmap`, `colmap` CLI.

## Data flow & IO
Root: `home_path/paradex_download/capture/object_turntable/<obj>/<index>/`.
videos/ → images/ → charuco_3d/ → rotation/ ; images/ → selected/+masks/+masked_images/ ; masked_images/+rotation/ → colmap/.
COLMAP pose: `T = [extrinsic;0001] @ rotation[frame]`, qvec = scipy quat (x,y,z,w order as stored).

## When working here
- Preferred: run the whole pipeline via `worker.py` (framework: skip/exception-capture/status). Stage order is fixed: extract_video → extract_charuco → get_rotation → extract_mask_sam3 → generate_colmap. Each stage script still runs standalone via its own `__main__`.
- Single-machine only — data lives in the local `paradex_download` copy, not sharded across capture PCs, and stages fan out internally. Don't wrap it with `run_distributed`/`shard`.
- **Untested end-to-end here:** the GPU stages (sam3/colmap) need torch + `sam3` + `pycolmap` + a GPU, so `worker.py`'s glue was verified for import/discovery only — validate a full run on the GPU box.
- Many sequences gated on exactly 24 cameras; mismatches are skipped/warned.
- `qvec` in `add_image` is built as `[x,y,z,w]` from scipy `as_quat()` then passed positionally — verify ordering matches the COLMAP DB convention if poses look wrong.

## Gotchas
- `extract_charuco.process_frame` references undefined `obj_name`/`index` in a print on the no-charuco branch (NameError if hit).
- `generate_colmap.main()` defaults to `['big_green_spray']`; pass objects to `worker.py` (or `main(obj_filter=...)`) for others. Importing the module no longer runs COLMAP (was a module-level loop; now under `__main__`).
- `extract_mask_sam3.py`'s old line-159 SyntaxError (bare Korean statement) is fixed; the SAM3 predictor is now lazy (`get_video_predictor()`), so importing the module no longer needs a GPU.
- SAM3 mask sampling uses `idx % 40 == 1`; `detect_outlier.py` (in check/) additionally requires `idx >= 160` and deletes others — keep sampling rules consistent.
- `paradex.dataset_acqusition` misspelled (missing 'i') — intentional (not imported here but used in sibling pipelines).
- `deprecated/` and `check/check_colmap copy.py` are stale leftovers.
