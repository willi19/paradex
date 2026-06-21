# CLAUDE.md — src/process/object_turntable/check

## Purpose
Ad-hoc validation/cleanup scripts for the object_turntable processing tree. Not part of the main pipeline.

## Files
- `check.py` — counts sequences/videos with complete `selected/`+`masked_images/`; "valid" when, per serial, all files in `selected/<serial>` match `idx % 40 == 1`. Prints `vid_cnt/vid_tot` and valid-sequence count. (`process_task` defined but unused.)
- `check_image.py` — iterate `images/<serial>/*.jpg`, flag files where `cv2.imread` is None (corrupted). Prints list. (`process_task` defined but unused.)
- `check_colmap.py` — print `<obj>/<index>` lacking `colmap/database.db`.
- `check_colmap copy.py` — byte-for-byte duplicate of `check_colmap.py` (name has a space). Stale; ignore.
- `detect_outlier.py` — DESTRUCTIVE cleanup. For each `selected/<serial>` frame: if `idx % 40 != 1 or idx < 160`, delete from `selected/`, `masked_images/`, `masks/` (png). Else read frame, `cv2.absdiff` vs `undistort_img(dropped_frame)` (a `::2,::2`=255 sentinel), threshold>30, and if `< 5000` non-zero px treat as dropped and delete the same three files.

## paradex modules used
`paradex.image.undistort.undistort_img`, `paradex.calibration.utils.load_camparam`, `paradex.image.image_dict.ImageDict`, `paradex.image.aruco`, `paradex.transforms.conversion.SOLVE_XA_B` (several imported but unused).

## Data flow & IO
Root hard-coded: `home_path/paradex_download/capture/object_turntable/<obj>/<index>`. Checks print only; `detect_outlier.py` mutates `selected/`, `masked_images/`, `masks/` in place.

## When working here
- The sampling rule here (`idx % 40 == 1 and idx >= 160`) is stricter than `extract_mask_sam3.py` (`idx % 40 == 1`). Keep these aligned or `detect_outlier.py` will delete frames the masker produced.

## Gotchas
- `detect_outlier.py` deletes files irreversibly — confirm the dataset is backed up / on shared before running.
- `check_colmap copy.py` is a leftover duplicate.
- Most scripts carry copy-pasted unused imports and an unused `process_task`.
