# Calibration — How to Use the Parameters (for humans & agents)

Read this before writing code that needs **camera matrices, camera poses, or the camera→robot
transform**. You have a dataset (or a live rig) and you want to project, undistort, triangulate, or
map points into the robot frame. Orientation + the three parameter stores:
[`README.md`](README.md).

> **Editing the capture / solve scripts** (intrinsic, extrinsic, hand-eye), board generation, or the
> file formats — not just reading params out? Read [`internals.md`](internals.md) instead.

Pipeline order is fixed: **intrinsic → extrinsic → hand-eye**. Camera **serial numbers are string keys**
everywhere. All loaders live in [`utils.py`](../../paradex/calibration/utils.py).

## TL;DR — the one call you usually want

```python
from paradex.calibration.utils import load_current_camparam, load_current_C2R
intrinsic, extrinsic = load_current_camparam()   # latest cam_param/<name>/
# intrinsic, extrinsic = load_camparam(demo_path)   # for a saved dataset dir (<demo_path>/cam_param/)
C2R = load_current_C2R()                          # (4,4) camera->robot
```

`load_current_camparam(name=None)` picks the **latest** `cam_param/<name>/`; pass `name` to pin one.
`load_camparam(demo_path)` reads `<demo_path>/cam_param/{intrinsics,extrinsics}.json` — the copy saved
alongside a captured dataset. Both return the same two dicts.

## The intrinsic dict

`intrinsic[serial]` is a dict:

| key | shape | meaning |
|-----|-------|---------|
| `original_intrinsics` | (3,3) | K for the **distorted** image (use with `dist_params`) |
| `intrinsics_undistort` | (3,3) | K for the **undistorted** image (use after `cv2.undistort`, no dist) |
| `dist_params` | (5,) or (1,5) | OpenCV `[k1,k2,p1,p2,k3]` |
| `height`, `width` | int | image size |

`load_current_intrinsic()` (the raw per-camera store, stage 1) returns the same keys **plus**
`intrinsics_warped` — currently identical to `intrinsics_undistort`.

## The extrinsic dict + projection

`extrinsic[serial]` is a **(3,4)** world→camera matrix `[R | t]` (camera-from-world). A world point `X`
(3,) projects as:

```python
import numpy as np
Xh = np.append(X, 1.0)
uvw = intrinsic[serial]["intrinsics_undistort"] @ extrinsic[serial] @ Xh   # X seen on the UNDISTORTED image
uv  = uvw[:2] / uvw[2]
```

Shortcut for the full 3×4 projection matrix per camera:

```python
from paradex.calibration.utils import get_cammtx
P = get_cammtx(intrinsic, extrinsic)   # {serial: intrinsics_undistort @ extrinsic}  (3,4)
```

`get_cammtx` uses `intrinsics_undistort`, so `P` maps world points onto the **undistorted** image.

## `ImageDict` — the main consumer (auto-loads the file structure)

Most downstream code does **not** call the loaders directly; it uses
[`paradex.image.image_dict.ImageDict`](../../paradex/image/image_dict.py), which wraps a capture
directory *and its `cam_param/`* and pulls the calibration in for you (see
[`agent_docs/image`](../image/README.md)):

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
- `triangulate_charuco()` / `triangulate_markers()` / `traingulate_points(pts2d)` — multi-view triangulation via `get_cammtx`.
- `project_pointcloud(pts3d)` / `project_mesh(obj)` — project world geometry back into every camera.
- `apply(func, use_camparam=True)` / `map_images(...)` — call `func(img, intrinsic=..., extrinsic=...)` per serial.
- `save_camparam(path)` — writes `cam_param/{intrinsics,extrinsics}.json` back out (round-trips the format).

So: to *use* calibration, prefer `ImageDict.from_path(...)` / `load_current_camparam()`; the raw loaders
above are the low-level fallback.

## Distorted vs. undistorted — pick one and stay consistent

This is **the** calibration footgun. Two consistent recipes:

- **Distorted (raw camera frames):** project with `original_intrinsics` + `dist_params`
  (`cv2.projectPoints`).
- **Undistorted:** first `cv2.undistort(img, original_intrinsics, dist_params, None, intrinsics_undistort)`,
  then use `intrinsics_undistort` with **no** distortion (this is what `get_cammtx` / `ImageDict` assume).

Mixing the two — e.g. `intrinsics_undistort` on a raw distorted frame — is the most common silent bug.
`intrinsics_undistort` comes from `cv2.getOptimalNewCameraMatrix(..., alpha=1)` (keeps all pixels, black
borders).

## Camera → robot (`C2R`)

`load_current_C2R()` returns the **(4,4)** transform that maps a point in the **camera / world
calibration frame** to the **robot base frame**:

```python
X_robot = C2R @ np.append(X_cam, 1.0)   # (4,)
```

It picks the latest `handeye_calibration/<name>/` and loads `<idx>/C2R.npy` for the **first sorted index**.
Extrinsics are metric (rescaled so the Charuco square edge = 0.06 m — see [`internals.md`](internals.md)),
so `C2R` is metric too.

## Units & conventions

- Extrinsic translations are **meters** (rescaled in `extrinsic/calculate.py`).
- Camera frames follow the OpenCV convention: **+Z forward, +X right, +Y down**.
- Rotations elsewhere in the repo: quaternions are **wxyz** in viser, **xyzw** in scipy — convert with
  `quat[[3,0,1,2]]`.

## Gotchas

- **Latest-wins loaders.** `load_current_*` silently pick the newest timestamped dir. A stale/bad
  calibration left in `cam_param/` becomes "current" — pin `name=` if you need a specific one.
- **Serials are strings.** `intrinsic["25305460"]`, never an int.
- **`load_camparam` vs `load_current_camparam`.** The first reads a dataset's *bundled* copy
  (`<demo_path>/cam_param/`); the second reads the *global* latest under `~/shared_data/cam_param/`.
  They can disagree — use the bundled one when reprocessing an old dataset.
- Typo `dataset_acqusition` (missing 'i') is intentional — don't "fix" imports.
