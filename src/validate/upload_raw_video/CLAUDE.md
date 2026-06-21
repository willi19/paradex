# CLAUDE.md — src/validate/upload_raw_video

## Purpose
Validate the raw-video undistort + NAS-upload pipeline, both end-to-end (`RawVideoProcessor`) and at the single-file-worker level (`undistort_raw_video`).

## Files
- `upload_local.py` — three lines: `rvp = RawVideoProcessor()`, `rvp.process()`, `rvp.wait_and_monitor()`. The actual entry point; delegates all logic to the library class.
- `test_func.py` — defines `undistort_raw_video(video_path, progress_dict, video_id)` and immediately calls it on a **hardcoded** path with a fresh `Manager().dict()`. Worker steps: derive `serial_num` from filename, derive `root_name` by stripping `capture_path_list` prefixes, compute `out_path` (`<root>/videos/<serial>.avi`) and `nas_path` (`<shared_dir>/<root_name>/videos/<serial>.avi`); update `progress_dict[video_id]`; load intrinsics via `load_camparam`; build undistort map (`precomute_undistort_map`); read frames, `apply_undistort_map`, write MJPG; `rsync_copy(out_path, nas_path)`.

## paradex modules used
- `paradex.video.raw_video_processor.RawVideoProcessor` (upload_local.py).
- `paradex.image.undistort`: `precomute_undistort_map`, `apply_undistort_map`.
- `paradex.calibration.utils.load_camparam`.
- `paradex.utils.upload_file.rsync_copy`.
- `paradex.utils.path`: `capture_path_list`, `shared_dir`, `home_path`.
- stdlib/3rd-party: `cv2`, `multiprocessing` (`Pool`, `Manager`, `cpu_count` imported), `glob`, `pathlib`.

## Data flow & IO
- Input: `<capture_root>/.../raw/videos/<serial>.avi`. `root_dir` = three `dirname`s up from the video (i.e., the dir containing `raw/`).
- Output local: `<root_dir>/videos/<serial>.avi` (MJPG, same fps/size).
- Output remote: `<shared_dir>/<root_name>/videos/<serial>.avi` via `rsync_copy`.
- Progress is reported through a shared `progress_dict[video_id]` (designed for multiprocessing fan-out, though `test_func.py` runs one synchronously).

## When working here
- Validation scripts; keep runnable as `python src/validate/upload_raw_video/<x>.py`.
- Do NOT edit the `.py` for doc tasks.

## Gotchas
- `test_func.py` has a **hardcoded** input path (`/home/temp_id/captures1/.../22645026.avi`) executed at import/run time — it WILL try to process that file. Edit before running.
- The "already processed / skip" block and several `progress_dict` error-handling branches are commented out; `load_camparam` failures are NOT caught (will raise).
- `RawVideoProcessor` contains the real pipeline logic — look there, not in these scripts, for batching/parallelism behavior.
- `Pool`/`cpu_count` are imported in `test_func.py` but unused in the active code.
