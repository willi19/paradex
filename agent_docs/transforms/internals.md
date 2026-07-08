# paradex/transforms — Internals (for agents editing this module)

**You are here to change the math** (the SVD rigid fit, the DLT/RANSAC triangulator, the projection
divide) or to reason about a shape/direction/units trap. If you only want to *call* a function, read
[`usage.md`](usage.md).

Three tiny files (~130 lines total), no shared state, no I/O. Organized by file. The through-line:
**everything is `numpy` (+ `transforms3d`), evaluated eagerly, and the docstrings overstate the
return shapes.**

---

## 1. `conversion.py`

### `SOLVE_XA_B(A, B)` — Kabsch/Umeyama rigid fit
Centroids → covariance `H = AAᵀ·BB` → SVD → `R = Vᵀ Uᵀ`, with the `det(R) < 0` reflection fix
(negate `Vt[-1]`, recompute), then `t = centroid_B − R·centroid_A`. Returns 4×4 `T` such that
**`T @ A ≈ B`**. Rigid only — **no scale, no shear**. Asserts equal shapes and 3-D points.

- **Direction matters.** `T` maps the *first* argument's frame onto the *second*'s. Every caller
  relies on this; don't silently flip `A`/`B`.

### `aa2mtx(pos_aa)` — assumes mm + non-zero rotation
- Translation is **unconditionally** `pos_aa[:3] / 1000` (xArm mm → m). Feeding metres gives a 1000×
  error.
- `angle = norm(pos_aa[3:])`, `axis = pos_aa[3:] / angle`. If the rotation vector is all zeros,
  `angle = 0` → **divide-by-zero**. No guard.

### `project(proj_mtx, x)` — the 4×4 branch is broken
The working path is a **3×4** `proj_mtx`: `(proj_mtx @ x.T).T` → divide by the third column → take
first two → `(N,2)`. But when `proj_mtx` is 4×4 (so `proj_x.shape[-1] == 4`) it does
`proj_x = proj_x[:, 3]`, collapsing to a **1-D** array; the next line `proj_x[:, 2:]` then raises an
`IndexError`. It almost certainly meant `proj_x[:, :3]`. **Only the 3×4 path (all real callers) works.**

---

## 2. `triangulate.py`

### `_triangulate(corners, projections)` — one DLT
Builds the `(2N, 4)` DLT matrix — each camera contributes `y·P₂ − P₁` and `x·P₂ − P₀` — takes the
smallest-singular-vector via SVD, de-homogenises. Returns a **`(3,)`** vector (docstring says `(1,3)`
— wrong). If `corners` is 3-D `(N, P, 2)` it recurses per point and `vstack`s. Needs ≥ 2 views or
returns `None`.

### `triangulation(...)` — RANSAC wrapper
`iterations` rounds, each sampling `max(2, N // 2)` cameras, triangulating, counting reprojection
inliers under `threshold` **pixels**. Keeps the estimate with the most inliers. For 3-D corners it
recurses per point (independent RANSAC each) and `vstack`s.

- **Return shape is driven by input, not the docstring.** `(3,)` for one point, `(P,3)` for a stack —
  **not** the documented `(4,3)`.
- **The multi-point view guard differs from the single-point one.** The `(N,P,2)` branch bails on
  `numImg <= 2` (needs **≥ 3**), while the core solve only needs `numImg < 2` (**≥ 2**). Not obviously
  intentional; be aware if you touch the thresholds.
- **No distortion handling.** Callers must pass already-undistorted corners and matching `(N,3,4)`
  projection matrices. The only real caller is
  [`paradex/image/image_dict.py`](../../paradex/image/image_dict.py) (multi-cam marker/keypoint 2D → 3D).
- Uses `np.random.choice` — non-deterministic across runs unless the global seed is fixed.

---

## 3. `coordinate.py` — hand-authored constants

Two dicts of constant 4×4 matrices (see the Notion "Coordinate system" doc linked in the file header),
keyed by device-name string. No functions.

- `DEVICE2WRIST["xarm"]` carries a `-0.01` m offset in the wrist column; a second (commented-out) xarm
  form with a √2 rotation is left in the source — leave it or delete it deliberately, don't half-edit.
- **`DEVICE2WRIST["franka"]` is not a valid rotation.** Its rows are `[[1,0,0],[0,0,1],[0,-1,1]]` — the
  third row `[0,-1,1]` is neither unit-length nor orthogonal to the second, so the top-left 3×3 is
  **non-orthonormal** (contrast the clean `xarm` block `[0,-1,0]`). Looks like a `1`-where-`0`-intended
  typo. Only affects `franka`; every other device block is a proper rotation.
- `DEVICE2GLOBAL` has **arms only** (`xarm`, `franka`, both identity) — no hand/xsens entry.

---

## 4. Traps that look like bugs (and the ones that are)

**Real breakage — fix at the call site, not here:**
- **`src/util/robot/merge_urdf.py` imports from a module that doesn't exist:**
  `from paradex.geometry.coordinate import DEVICE2WRIST`. There is **no `paradex/geometry/` package** —
  the dict lives in `paradex.transforms.coordinate`. Every other caller uses the correct path. Don't
  create a `geometry` shim; fix the import.
- **`project`'s 4×4 branch** (`IndexError`) and **`aa2mtx`'s zero-rotation divide** (§1) — latent,
  callable bugs.
- **`DEVICE2WRIST["franka"]` non-orthonormal** (§3).

**Intentional / leave alone:**
- **`__init__.py` is empty on purpose.** Don't add re-exports expecting `from paradex.transforms import X`
  to work — every current caller imports by file path.
- Docstrings overstate return shapes (`_triangulate` `(3,)` not `(1,3)`; `triangulation` input-driven,
  not `(4,3)`). Fix the docstrings if you like, but callers already rely on the *actual* shapes.

**Blast radius:**
- `SOLVE_XA_B` is by far the most-used symbol (pose-from-correspondences across calibration, turntable
  processing, and grasp inference — see the caller table in [`README.md`](README.md)). `triangulation`
  and `project` are the geometry backbone of `paradex/image/`. A signature or return-shape change here
  ripples through `src/calibration/`, `src/process/`, and `src/inference/`.
