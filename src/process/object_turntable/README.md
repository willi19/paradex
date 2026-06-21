# Object Turntable Processing

Reconstructs a 3D object scanned on a charuco turntable: extract frames from multi-camera videos, triangulate the charuco board for per-frame turntable rotation, segment the object with SAM3, and build a COLMAP point cloud with known camera poses.

## Scripts
| File | Purpose |
|------|---------|
| `extract_video.py` | Extract JPEG frames from each camera video into `images/<serial>/frame_NNNNNN.jpg`. Skips frame-drop sentinel frames. Parallel over videos. |
| `extract_charuco.py` | Per frame, triangulate the charuco board across cameras (`ImageDict.triangulate_charuco`) and save 3D corner IDs/coords to `charuco_3d/`. Parallel over frames. |
| `get_rotation.py` | From triangulated charuco corners, solve board-in-world transform per frame (`SOLVE_XA_B` against base board `'2'`) → `rotation/<frame>.npy`. |
| `extract_mask_sam3.py` | Run SAM3 video predictor with text prompt to segment the object on selected frames → `masks/` (png) + `masked_images/` (jpg). Copies sampled frames into `selected/`. |
| `generate_colmap.py` | Build a COLMAP database from masked images with known per-frame poses (extrinsic ∘ rotation), feature-extract with masks, sequential match, then `point_triangulator`. |

## Usage
Run in this order (each script edits the local `paradex_download/capture/object_turntable` tree). Most loop over all `<obj>/<index>`; some hard-code an object name in the bottom loop — edit before running.
```bash
python src/process/object_turntable/extract_video.py      # videos -> images/
python src/process/object_turntable/extract_charuco.py    # images/ -> charuco_3d/
python src/process/object_turntable/get_rotation.py       # charuco_3d/ -> rotation/
python src/process/object_turntable/extract_mask_sam3.py  # images/ -> masks/, masked_images/, selected/
python src/process/object_turntable/generate_colmap.py    # masked_images/ + rotation/ -> colmap/
```
Stage order: **extract_video → extract_charuco → get_rotation → extract_mask_sam3 → generate_colmap**.

Validation/cleanup helpers live in [`check/`](check/) — run as needed (e.g. `detect_outlier.py` to strip dropped/invalid frames before COLMAP). See its README.

## Inputs & Outputs
Per sequence: `paradex_download/capture/object_turntable/<obj>/<index>/`
- Input: `videos/<serial>.<ext>` (expects 24 cameras), camera params via `load_camparam`.
- `images/<serial>/frame_NNNNNN.jpg` — extracted frames.
- `charuco_3d/{NNNNNN_id.npy, NNNNNN_cor.npy}` — triangulated board corner IDs and 3D coords.
- `rotation/<frame>.npy` — 4×4 board-in-world transform per frame.
- `selected/<serial>/<frame>.jpg` — frames sampled for masking (`idx % 40 == 1`).
- `masks/<serial>/<frame>.png`, `masked_images/<serial>/<frame>.jpg` — SAM3 object masks and masked frames.
- `colmap/database.db`, `colmap/sparse/` (with `sparse/tmp_initial/` of prior poses) — COLMAP reconstruction.

## Deprecated
`deprecated/` holds older/experimental mask-extraction scripts (SAM2 variants, gunhee mask, object-center, rename). **Stale — do not use; kept for reference only.**

## Related
- paradex: `paradex.image.image_dict.ImageDict` (`triangulate_charuco`), `paradex.image.aruco` (`get_board_cor`, `find_common_indices`, `detect_charuco`, `merge_charuco_detection`), `paradex.image.undistort.undistort_img`, `paradex.transforms.conversion.SOLVE_XA_B`, `paradex.calibration.utils.load_camparam`, `paradex.calibration.colmap.COLMAPDatabase`.
- External: `sam3` (`build_sam3_video_predictor`), `pycolmap` + `colmap` CLI.
- Capture side: [`src/dataset_acquisition/object_turntable`](../../dataset_acquisition/object_turntable)
