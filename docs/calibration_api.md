# Calibration — API

Method reference for the calibration subsystem: parameters (input) and return values (output). For
the architecture, param stores, and board semantics, see the {doc}`overview <calibration>`.

Signatures are verified against the code. The subsystem lives in `paradex/calibration/`; the
application scripts (capture/solve/board-gen) live in `src/calibration/`. Everything is keyed by
**camera serial string**.

Each entry is collapsed below — click to expand.

:::{dropdown} Loading params — `utils.py` (`paradex/calibration/utils.py`)
:open:

Path constants: `cam_param_dir`, `intrinsic_dir`, `extrinsic_dir`, `handeye_calib_path`,
`eef_calib_path` (all under `~/shared_data`).

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_current_camparam(name=None)` | `name: str\|None` | `(intrinsic, extrinsic)` | Latest (or named) `cam_param/<name>/`. `intrinsic[serial]` = `{original_intrinsics (3,3), intrinsics_undistort (3,3), dist_params, height, width}`; `extrinsic[serial]` = `(3,4)` world→cam. |
| `load_camparam(demo_path)` | `demo_path: str` | `(intrinsic, extrinsic)` | Same shape, from `<demo_path>/cam_param/`. Use for a saved dataset. |
| `load_current_intrinsic()` | — | `{serial: dict}` | Latest per-camera `intrinsic/<serial>/param/*.json`; derives `intrinsics_undistort`/`intrinsics_warped` via `getOptimalNewCameraMatrix(alpha=1)` at load time. |
| `load_current_C2R()` | — | `(4,4) ndarray` | Latest `handeye_calibration/<name>/<idx0>/C2R.npy` (camera→robot). |
| `load_c2r(demo_path)` | `demo_path: str` | `(4,4) ndarray` | `<demo_path>/C2R.npy`. |
| `load_current_eef()` / `load_eef(demo_path)` | — / `demo_path` | `ndarray` | End-effector calibration (`eef.npy`). |
| `get_cammtx(intrinsic, extrinsic)` | two dicts | `{serial: (3,4)}` | Per-serial projection `intrinsics_undistort @ extrinsic`. Feed **undistorted** 2D. |
| `save_current_camparam(save_path)` | `save_path: str` | `None` | Copy latest `cam_param/<name>/` → `<save_path>/cam_param/`. |
| `save_current_C2R(save_path)` | `save_path: str` | `None` | Write latest `C2R.npy` into `save_path`. |
| `get_handeye_calib_traj(arm_name)` | `arm_name: str` | `str` | Path to the canned hand-eye trajectory (`config_dir/hecalib/<arm>`). |

:::{warning}
`load_current_intrinsic` stores only `K`+`distortion` on disk; `intrinsics_undistort` is recomputed
on every load. `load_colmap_camparam` sets `dist_height` twice (width overwrites height) — a known
source typo; use `height`/`width` instead.
:::
:::

:::{dropdown} COLMAP solve — `colmap.py` (`paradex/calibration/colmap.py`)

Used by `src/calibration/extrinsic/calculate.py`.

| Function / class | Input | Output | Description |
|------------------|-------|--------|-------------|
| `load_colmap_camparam(path, orig_size=(2048,1536))` | reconstruction dir, size | `(intrinsics, extrinsics)` | Parse a solved `pycolmap.Reconstruction`; serial parsed from image name; builds `original`/`undistort` K + `dist_params`. |
| `get_two_view_geometries(cam1, cam2, pix1, pix2, indices, pair)` | two cam ids, matched pixels, id array, `(id1,id2)` | two-view-geometry tuple | Build a verified geometry entry for the COLMAP DB. |
| `COLMAPDatabase(sqlite3.Connection)` | — | DB handle | `.create_tables()`, `.add_camera(model, w, h, params, prior)`, `.add_image`, `.add_keypoints`, `.add_matches`, `.add_two_view_geometry`, `.get_camera()`. Camera model `4` = OPENCV (`fx,fy,cx,cy,k1,k2,p1,p2`). |
| `image_ids_to_pair_id` / `pair_id_to_image_ids` | image ids / pair id | pair id / `(id1,id2)` | COLMAP pair-id packing. |
| `array_to_blob` / `blob_to_array` | ndarray / blob | blob / ndarray | SQLite (de)serialization. |
:::

:::{dropdown} Hand-eye solver — `Tsai_Lenz.py` (`paradex/calibration/Tsai_Lenz.py`)

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `solve_ax_xb(A_list, B_list, init_X=None, max_epochs=3000, learning_rate=0.001, verbose=False)` | lists of `(4,4)` `A`/`B`, optional init | `(4,4) X` | Gradient-based `AX = XB` (camera↔EEF) solve → `C2R`. |
| `solve_axb_cpu(A, B)` | `(4,4)` `A`, `B` | `(4,4)` | Closed-form single-pair helper. |
| `logR(T)` | `(4,4)`/`(3,3)` | `(3,)` | Rotation log (axis-angle) helper. |
:::

:::{dropdown} Application scripts — `src/calibration/`

Not importable API (run with `python src/calibration/...`), listed for orientation.

| Script | Role | Notes |
|--------|------|-------|
| `intrinsic/{capture,client}.py` | distributed intrinsic capture | Main PC + Capture PC; **auto-capture, no save button**. |
| `intrinsic/calculate.py` | per-camera solve | `cv2.calibrateCamera` → `intrinsic/<serial>/param/<ts>.json`. `--serials` optional. |
| `extrinsic/{capture,client,calculate}.py` | rig extrinsic | COLMAP + refine + metric rescale → `cam_param/<name>/`. |
| `handeye/` | camera→robot | Tsai-Lenz → `C2R.npy`. |
| `generate_board.py` | print-ready ChArUco PDF | OpenCV `generateImage` from `charuco_info.json`; self-checks with `detect_charuco` before writing; exact physical scale. `--board`, `--square-mm`, `--dpi`, `--margin-mm`, `--out`. |

Board detection call surface (`detect_charuco`, `get_charuco_detector`, `merge_charuco_detection`)
is documented in {doc}`image_api`; board *semantics* (`setLegacyPattern`, scale) in
{doc}`calibration`.
:::
