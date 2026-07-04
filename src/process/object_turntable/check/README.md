# Object Turntable — Validation & Cleanup Helpers

Throwaway helper scripts for inspecting and cleaning the [object_turntable](../) processing tree. None are part of the main reconstruction flow; run individually as needed.

## Scripts
| File | Purpose |
|------|---------|
| `check.py` | Tally how many sequences/videos have a complete `selected/` + `masked_images/` set (counts frames matching `idx % 40 == 1`). Prints `<done>/<total>` summary. |
| `check_image.py` | Walk every `images/<serial>/*.jpg` and report files that `cv2.imread` returns `None` for (corrupted). |
| `check_colmap.py` | Report sequences missing `colmap/database.db`. |
| `check_colmap copy.py` | **Duplicate of `check_colmap.py`** (filename literally contains a space). Stale leftover — ignore. |
| `detect_outlier.py` | Cleanup: delete frames not matching the sampling rule (`idx % 40 == 1 and idx >= 160`) and detect/remove dropped frames (diff vs. undistorted checkerboard-sentinel image, < 5000 non-zero px). Removes the matching `images/`, `masked_images/`, and `masks/` files. **Destructive — deletes files.** |

## Usage
```bash
python src/process/object_turntable/check/check.py          # completeness summary
python src/process/object_turntable/check/check_image.py    # find corrupted jpgs
python src/process/object_turntable/check/check_colmap.py   # find missing colmap DBs
python src/process/object_turntable/check/detect_outlier.py # DELETE dropped/invalid frames
```
All scripts hard-code the root `home_path/paradex_download/capture/object_turntable` and iterate over `<obj>/<index>`.

## Inputs & Outputs
Read-only checks (`check.py`, `check_image.py`, `check_colmap.py`) only print. `detect_outlier.py` mutates the `selected/`, `masked_images/`, and `masks/` directories in place.

## Related
- Parent pipeline: [`../README.md`](../README.md)
- paradex: `paradex.image.undistort.undistort_img`, `paradex.calibration.utils.load_camparam`, `paradex.image.image_dict.ImageDict`.
