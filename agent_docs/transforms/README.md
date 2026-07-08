# agent_docs/transforms — agent orientation

Docs for **AI agents working on `paradex/transforms/`** — the low-level **coordinate / rigid-transform
math** layer. No I/O, no calibration loading, no camera or robot handles: just numpy (+ `transforms3d`)
functions and a couple of hard-coded 4×4 lookup tables that everything geometric in the repo leans on.
Three tiny source files (~130 lines total). Calling one? See [`usage.md`](usage.md). Editing the math
or chasing a shape/direction trap? See [`internals.md`](internals.md).

Mental model: **points/poses in → points/poses out.** `conversion.py` = pose ↔ matrix + point projection +
an SVD point-cloud aligner; `triangulate.py` = multi-view 2D → 3D (DLT, with a RANSAC wrapper); `coordinate.py`
= static per-device frame-convention matrices (`DEVICE2WRIST`, `DEVICE2GLOBAL`). `__init__.py` is **empty**, so
callers always import by file path, e.g. `from paradex.transforms.conversion import SOLVE_XA_B`.

## File map
| File | What it is |
|------|-----------|
| `conversion.py` | `aa2mtx` (xArm axis-angle pose → 4×4), `to_homo`, `project` (points through a projection matrix), `SOLVE_XA_B` (SVD rigid fit `T @ A = B`). |
| `triangulate.py` | `_triangulate` (single DLT via SVD) + `triangulation` (RANSAC-filtered wrapper). Multi-view 2D corners → 3D point(s). |
| `coordinate.py` | Static dicts of hand-authored 4×4 frame transforms: `DEVICE2WRIST` (per arm/hand/xsens) and `DEVICE2GLOBAL` (arms only). No functions. |
| `__init__.py` | **Empty** — no re-exports; import each symbol by its file path. |

All paths relative to [`paradex/transforms/`](../../paradex/transforms/). Related: triangulation feeds
[`paradex/image/image_dict.py`](../../paradex/image/image_dict.py) (see `agent_docs/image/`).

---

## Who calls this
| Caller | Uses |
|--------|------|
| [`src/calibration/handeye/calculate.py`](../../src/calibration/handeye/calculate.py) | `SOLVE_XA_B` — camera-motion `M1 C1 C2 M2` for Tsai-Lenz hand-eye. |
| [`src/validate/calibration/compare_xarm_kinematic_calib.py`](../../src/validate/calibration/compare_xarm_kinematic_calib.py) | `SOLVE_XA_B` — relative motion between charuco captures. |
| `src/process/object_turntable/*.py` (`get_rotation`, `extract_charuco`, `extract_mask_sam3`, `check/*`, several `deprecated/*`) | `SOLVE_XA_B` — board/object pose in world from point correspondences. |
| `src/inference/{grasp_w_gui,bodex,grasp_eval/real,pringles_test}/{object6d,visualize,grasp,get_floor}.py` | `SOLVE_XA_B` — object 6D pose (`obj_T`) from A→B point fit. |
| [`src/capture/robot/xarm_teaching.py`](../../src/capture/robot/xarm_teaching.py) | `aa2mtx` — live xArm `get_position_aa` → 4×4 wrist pose to save. |
| [`paradex/image/projection.py`](../../paradex/image/projection.py) | `project` — `project_point` renders 3D verts into an image. |
| [`paradex/image/image_dict.py`](../../paradex/image/image_dict.py) | `triangulation` — multi-cam marker/keypoint 2D → 3D. |
| [`paradex/retargetor/unimanual.py`](../../paradex/retargetor/unimanual.py) | `DEVICE2WRIST`, `DEVICE2GLOBAL` — retarget human wrist into device frame. |
| [`src/util/robot/merge_urdf.py`](../../src/util/robot/merge_urdf.py) | `DEVICE2WRIST` — **but via a broken import** `from paradex.geometry.coordinate import DEVICE2WRIST` (see Gotchas). |

So `SOLVE_XA_B` is by far the most-used symbol (pose-from-correspondences everywhere); `triangulation`
and `project` are the geometry backbone of `paradex/image/`; the `coordinate.py` dicts feed retargeting
and URDF merging.

---

## `conversion.py`

### `aa2mtx(pos_aa)` — xArm 6-vector pose → 4×4
Input is `[x, y, z, rx, ry, rz]` (translation + axis-angle rotation vector), i.e. the exact output of xArm
`get_position_aa`. Builds a 4×4: translation is `pos_aa[:3] / 1000` (**millimetres → metres, hard-coded**),
rotation is `axangle2mat(axis, angle)` where `angle = norm(pos_aa[3:])` and `axis = pos_aa[3:] / angle`.

### `to_homo(x)` / `project(proj_mtx, x)`
`to_homo` appends a ones column to an `(N, D)` array. `project` sends `(N,3)` (or already-homogeneous)
points through `proj_mtx`, then divides by the third coordinate and returns `(N,2)` pixels. Designed for a
**3×4** projection matrix (the working path); the 4×4 branch is buggy — see Gotchas.

### `SOLVE_XA_B(A, B)` — SVD rigid alignment
Classic Kabsch/Umeyama (no scale): centroids → covariance `H = AAᵀ·BB` → SVD → `R = Vᵀ Uᵀ`, with the
`det(R) < 0` reflection fix, then `t = centroid_B − R·centroid_A`. Returns a 4×4 `T` such that
**`T @ A ≈ B`** for `Nx3` point sets. Asserts equal shapes and 3-D points. This is the workhorse pose
estimator across calibration, turntable processing, and grasp inference.

## `triangulate.py`

### `_triangulate(corners, projections)` — one DLT solve
`corners` `(N,2)` image points across `N` cameras, `projections` `(N,3,4)`. Builds the `(2N,4)` DLT matrix
(each camera contributes `y·P₂ − P₁` and `x·P₂ − P₀`), takes the smallest-singular-vector via SVD, and
returns the de-homogenised **3-vector** (shape `(3,)`, despite the docstring saying `(1,3)`). If `corners`
is 3-D `(N, P, 2)` it recurses per point and `vstack`s. Needs ≥ 2 views or returns `None`.

### `triangulation(corners, projections, threshold=1.5, iterations=100)` — RANSAC wrapper
Runs `iterations` rounds, each sampling `max(2, N//2)` cameras, triangulating, and counting reprojection
inliers under `threshold` **pixels**. Keeps the estimate with the most inliers. For 3-D `(N,P,2)` corners it
recurses per point (each point gets its own independent RANSAC) and `vstack`s; returns `None` if any point
or view-count check fails. Return shape depends on input, not the `(4,3)` the docstring claims.

## `coordinate.py`
Two dicts of hand-authored constant 4×4 matrices (see the linked Notion "Coordinate system" doc in the file
header). Keys are device-name strings.
- `DEVICE2WRIST` — `T^{predefined wrist}_{device wrist}` for `xarm`, `franka`, `allegro`, `inspire`,
  `xsens_left`, `xsens_right`. The `xarm` variant carries a `-0.01` m offset in the wrist column; a second
  (commented-out) xarm form with a √2 rotation is left in the source.
- `DEVICE2GLOBAL` — `T^{predefined global}_{device}` for **arms only** (`xarm`, `franka`), both identity.
  There is **no** hand/xsens entry here (retargetor works around this with `np.eye(4)` for hands).

---

## Gotchas for editors
- **`merge_urdf.py` imports from a module that doesn't exist.** `src/util/robot/merge_urdf.py` does
  `from paradex.geometry.coordinate import DEVICE2WRIST`, but there is **no `paradex/geometry/` package** — the
  dict lives in `paradex/transforms/coordinate.py`. That import is broken (should be `paradex.transforms.coordinate`).
  Every other caller uses the correct path.
- **`project`'s 4×4 branch is broken.** When `proj_mtx` is 4×4 (so `proj_x.shape[-1] == 4`), it does
  `proj_x = proj_x[:, 3]`, collapsing to a **1-D** array; the next line `proj_x[:, 2:]` then raises an
  IndexError. It almost certainly meant `proj_x[:, :3]`. Only the 3×4 path (the one all callers use) works.
- **`aa2mtx` assumes millimetres and a non-zero rotation.** Translation is unconditionally `/1000` (xArm mm →
  m); feeding metres gives a 1000× error. If `pos_aa[3:]` is all zeros, `angle = 0` → divide-by-zero in
  `axis = pos_aa[3:] / angle`.
- **`DEVICE2WRIST["franka"]` is not a valid rotation.** Its rows are `[[1,0,0],[0,0,1],[0,-1,1]]` — the third
  row `[0,-1,1]` is neither unit-length nor orthogonal to the second, so the top-left 3×3 is **non-orthonormal**
  (contrast the `xarm` block's clean `[0,-1,0]`). Looks like a typo (`1` where a `0` was intended). Only affects
  `franka`; the other devices' blocks are proper rotations.
- **Docstrings overstate return shapes.** `_triangulate` returns `(3,)` not `(1,3)`; `triangulation` returns a
  shape driven by the input (`(3,)` for one point, `(P,3)` for a stack), not the documented `(4,3)`.
- **`SOLVE_XA_B` solves `T @ A = B`, not `A = T @ B`.** The direction matters — callers rely on `T` mapping the
  first argument's frame onto the second's. It's rigid-only (rotation + translation, no scale/shear).
- **`triangulation`'s `threshold` is in pixels and `_triangulate`'s inputs are undistorted.** Callers
  (`image_dict.py`) must pass already-undistorted 2D corners and matching `(N,3,4)` projection matrices; there's
  no distortion handling here.
- **`__init__.py` is empty on purpose.** Don't add re-exports expecting `from paradex.transforms import X` to
  work elsewhere — every current caller imports by file path.
