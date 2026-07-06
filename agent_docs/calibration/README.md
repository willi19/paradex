# Calibration — parameters & implementation

Two audiences:
- **Consuming calibration** (you have a dataset / live rig and need camera matrices, poses, or the camera→robot transform) → [Using the parameters](#using-the-parameters).
- **Editing the calibration code** (fixing/extending the capture or solve scripts) → [Implementation](#implementation).

Pipeline order is fixed: **intrinsic → extrinsic → hand-eye**. Each stage consumes the previous stage's output.

---

## The three parameter stores

Everything lives under `~/shared_data`. Camera **serial numbers are string keys** everywhere.

| Store | Path | Written by | Read by |
|-------|------|-----------|---------|
| Per-camera intrinsics | `intrinsic/<serial>/param/<ts>.json` | `src/calibration/intrinsic/calculate.py` | `load_current_intrinsic()` |
| Combined cam params | `cam_param/<name>/{intrinsics,extrinsics}.json` | `src/calibration/extrinsic/calculate.py` | `load_current_camparam()`, `load_camparam(demo_path)` |
| Camera→robot | `handeye_calibration/<name>/<idx>/C2R.npy` | `src/calibration/handeye/calculate.py` | `load_current_C2R()` |

`<ts>`/`<name>` are `YYYYMMDD_HHMMSS`; "current" loaders pick the **latest** dir via `find_latest_directory`. All helpers are in [`paradex/calibration/utils.py`](../../paradex/calibration/utils.py).

### Why two intrinsic stores
`intrinsic/<serial>/param/*.json` is the **raw per-camera calibration** (one camera, `cv2.calibrateCamera`). The extrinsic solve reads those as fixed intrinsics, runs COLMAP, and re-emits them **plus poses** into `cam_param/<name>/`. Downstream code almost always wants `cam_param/` (via `load_current_camparam`), not the raw per-camera dir.

---

## Using the parameters

### The one call you usually want
```python
from paradex.calibration.utils import load_current_camparam, load_current_C2R
intrinsic, extrinsic = load_current_camparam()   # or load_camparam(demo_path) for a saved dataset
C2R = load_current_C2R()                          # 4x4 camera->robot
```

`intrinsic[serial]` is a dict:
| key | shape | meaning |
|-----|-------|---------|
| `original_intrinsics` | (3,3) | K for the **distorted** image (use with `dist_params`) |
| `intrinsics_undistort` | (3,3) | K for the **undistorted** image (use after `cv2.undistort`, no dist) |
| `dist_params` | (5,) or (1,5) | OpenCV `[k1,k2,p1,p2,k3]` |
| `height`, `width` | int | image size |

`extrinsic[serial]` is a **(3,4)** world→camera matrix `[R | t]` (camera-from-world). A world point `X` (3,) projects as:
```python
import numpy as np
Xh = np.append(X, 1.0)
uvw = intrinsic[serial]["intrinsics_undistort"] @ extrinsic[serial] @ Xh   # if X seen on undistorted image
uv  = uvw[:2] / uvw[2]
```
Shortcut for the full 3x4 projection matrix per camera:
```python
from paradex.calibration.utils import get_cammtx
P = get_cammtx(intrinsic, extrinsic)   # {serial: intrinsics_undistort @ extrinsic}  (3,4)
```

### `ImageDict` — the main consumer (auto-loads the file structure)
Most downstream code does **not** call the loaders directly; it uses
[`paradex.image.image_dict.ImageDict`](../../paradex/image/image_dict.py), which wraps a capture
directory *and its `cam_param/`* and pulls the calibration in for you:
```python
from paradex.image.image_dict import ImageDict

# from a saved dataset dir: if <path>/cam_param/ exists, intrinsic+extrinsic load automatically
imgs = ImageDict.from_path(path)          # calls load_camparam(path) when cam_param/ is present
# or attach the latest global calibration to any ImageDict:
imgs.load_current_camparam()              # -> load_current_camparam()
imgs.set_camparam(intrinsic, extrinsic)   # or inject explicitly
```
Once camparam is attached, the geometry methods use it:
- `undistort()` — per-serial `cv2.undistort` with `original_intrinsics`+`dist_params` → `intrinsics_undistort`.
- `triangulate_charuco()` / `triangulate_markers()` / `traingulate_points(pts2d)` — multi-view triangulation using `get_cammtx`.
- `project_pointcloud(pts3d)` / `project_mesh(obj)` — project world geometry back into every camera.
- `apply(func, use_camparam=True)` / `map_images(...)` — call `func(img, intrinsic=..., extrinsic=...)` per serial.
- `save_camparam(path)` — writes `cam_param/{intrinsics,extrinsics}.json` back out (round-trips the format).

So: to *use* calibration, prefer `ImageDict.from_path(...)` / `load_current_camparam()`; the raw
loaders above are the low-level fallback.

### Distorted vs. undistorted — pick one and stay consistent
- Raw camera frames are **distorted** → project with `original_intrinsics` + `dist_params` (`cv2.projectPoints`), or undistort the image first (`cv2.undistort(img, original_intrinsics, dist_params, None, intrinsics_undistort)`) and then use `intrinsics_undistort` with **no** distortion. Mixing the two is the most common silent bug.
- `intrinsics_undistort` comes from `cv2.getOptimalNewCameraMatrix(..., alpha=1)` (keeps all pixels, black borders). `load_current_intrinsic` also exposes `intrinsics_warped` (currently identical to undistort).

### Camera → robot
`C2R` (4x4) maps a point in **camera/world calibration frame** to the **robot base frame**: `X_robot = C2R @ [X_cam, 1]`. Extrinsics are metric (scaled so the Charuco square edge = 0.06 m), so `C2R` is metric too.

### Units & conventions
- Extrinsic translations are **meters** (rescaled in `extrinsic/calculate.py`).
- Coordinate frames follow OpenCV camera convention (+Z forward, +X right, +Y down).
- Rotations elsewhere in the repo: quaternions are **wxyz** in viser; scipy is xyzw — convert with `quat[[3,0,1,2]]`.

---

## Implementation

### Stage 1 — intrinsic (`src/calibration/intrinsic/`)
Distributed, **auto-capture, no save button**. Each Capture PC (`client.py`) reads its cameras from the daemon's shared memory, detects the Charuco board, and keeps a frame only when the detected corner-centroid is novel vs. every frame already kept for that camera (`NOVELTY_PX`, `MIN_CORNERS`, `MAX_FRAMES`). Main PC (`capture.py`) shows a merged preview + per-camera `kept/target`; `q` sends `exit`, clients save `keypoint/<ts>.npy` shape `(N, num_corners, 1, 2)` with NaN for undetected corners. `calculate.py` masks NaN per frame and runs `cv2.calibrateCamera` per serial → `param/<ts>.json`. Board object-point scale is irrelevant to K/distortion. See [`src/calibration/intrinsic/CLAUDE.md`](../../src/calibration/intrinsic/CLAUDE.md).

### Stage 2 — extrinsic (`src/calibration/extrinsic/`)
Distributed Charuco capture (`capture.py` + `client.py`, manual `c` to save all cameras' full-res corners/ids/images) → `calculate.py` builds a COLMAP database from the 2D corners, runs `pycolmap.incremental_mapping`, undistorts, triangulates the board, then a **refine pass** (keep corners with reproj err < 2px into a new timestamped dir) and re-solves, finally rescaling translations by `0.06 / mean_board_edge` → `cam_param/<name>/{intrinsics,extrinsics}.json`. Intrinsics are seeded from `load_current_intrinsic()` (stage 1). See [`src/calibration/extrinsic/CLAUDE.md`](../../src/calibration/extrinsic/CLAUDE.md).

> **Why `ba_refine_focal_length/principal_point/extra_params=True`.** These are intentionally ON: with the stage-1 intrinsics held **fixed**, COLMAP incremental mapping **fails to converge** (registration/BA does not settle). Letting BA also refine intrinsics is what makes the solve converge. The tradeoff is that per-camera intrinsics can drift from stage 1, so intrinsic quality still matters (better stage-1 coverage → less drift). Do **not** naively flip these to False to "trust stage 1" — that reintroduces the non-convergence. A real fix for "fix intrinsics AND converge" needs a different solver (Charuco PnP-initialized multi-cam BA, where poses are seeded from known board geometry instead of SfM bootstrapping), not COLMAP incremental SfM.

### Stage 3 — hand-eye (`src/calibration/handeye/`)
Tsai-Lenz `AX = XB` from paired {camera-observed board pose, robot end-effector pose} → `C2R.npy`. Needs a valid extrinsic first.

### Charuco detection (`paradex/image/aruco.py`)
`boardinfo_dict` is loaded from `system/current/charuco_info.json` (board `"3"` = 11×8 → 70 corners is the default for intrinsic/extrinsic). `get_charuco_detector()` builds one `cv2.aruco.CharucoDetector` per board (cached), honoring `setLegacyPattern`. `detect_charuco(img)` returns `{b_id: {checkerCorner (n,2), checkerIDs (n,)}}`; `merge_charuco_detection` concatenates boards with per-board id offsets. Detector runs with **default params** (no `DetectorParameters`/`CharucoParameters` tuning) — a lever if detection is flaky.

> **`setLegacyPattern` must match how the board was printed.** OpenCV ≥4.6 flipped the default Charuco layout (top-left cell went from marker → empty checker); `setLegacyPattern(True)` restores the old layout. The generator [`src/util/marker/generate_charuco.py`](../../src/util/marker/generate_charuco.py) places a marker on cell `(0,0)` (`(row+col)%2==0`), i.e. the **legacy** layout. In `charuco_info.json`, boards `"1"`/`"2"` set `setLegacyPattern:true` (consistent); board `"3"` has **no key → defaults False** — inconsistent with the others and with the generator convention. If a board's flag doesn't match its print, `detectBoard` fails wholesale or misassigns corner ids. To confirm empirically, run `detect_charuco` on one captured image with the flag both ways and see which yields corners.

### Format reference
- `intrinsic/<serial>/param/<ts>.json`: `{RMS_error, K (3x3 list), distortion (5,), width, height}`.
- `cam_param/<name>/intrinsics.json` per serial: `{original_intrinsics (9,), intrinsics_undistort (9,), dist_params, height, width}` — note the flattened 3x3.
- `cam_param/<name>/extrinsics.json` per serial: 12-length list → reshaped to (3,4).
- `C2R.npy`: (4,4) float.

### Gotchas for editors
- `capture.py`/`client.py` pairs are coupled by stream data types (`'image'` /8 JPEG, `'charuco_detection'` /8 float32) and ports (1234 publish, 6890 command). Change both together.
- `load_current_intrinsic` derives `intrinsics_undistort` at load time via `getOptimalNewCameraMatrix(alpha=1)` — it is **not** stored in the per-camera `param` json (only `K`+`distortion` are).
- `extrinsic/calculate.py` writes the refined model into a **new** timestamped extrinsic dir but the final `cam_param` uses the original `<name>` — trace `__main__` carefully before changing paths.
- Typo `dataset_acqusition` is intentional; don't "fix" it.
