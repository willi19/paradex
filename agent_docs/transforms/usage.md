# paradex/transforms — How to Call Each Function

Recipes for the coordinate / rigid-transform math layer. Every symbol is imported **from its
submodule** (`paradex.transforms.conversion`, not `paradex.transforms` — the package `__init__.py`
is empty). Pure numpy in → numpy out; no I/O, no calibration loading, no device handles.

> Editing one of these functions, or reasoning about a shape/direction trap? Read [`internals.md`](internals.md).

---

## 1. Pose / point conversion — [`conversion.py`](../../paradex/transforms/conversion.py)

```python
from paradex.transforms.conversion import aa2mtx, to_homo, project, SOLVE_XA_B
```

### `aa2mtx(pos_aa)` — xArm 6-vector pose → 4×4
`pos_aa = [x, y, z, rx, ry, rz]` (translation in **millimetres** + axis-angle rotation vector), i.e.
the exact output of xArm `get_position_aa`. Returns a 4×4 with translation `pos_aa[:3] / 1000` (mm → m)
and rotation `axangle2mat`.

```python
T_wrist = aa2mtx(arm.get_position_aa()[0])   # 4×4 wrist pose in metres
```

### `SOLVE_XA_B(A, B)` — rigid fit `T @ A ≈ B` (the workhorse)
SVD Kabsch/Umeyama, rotation + translation, **no scale**. `A`, `B` are `(N,3)` point sets (same shape).
Returns a 4×4 `T` mapping the **first** argument's frame onto the second's.

```python
T = SOLVE_XA_B(pts_local, pts_world)   # object 6D pose from correspondences
```

### `to_homo(x)` / `project(proj_mtx, x)`
`to_homo((N,D))` → `(N,D+1)` (appends a ones column). `project(proj_mtx, x)` pushes `(N,3)` (or already
homogeneous) points through a **3×4** projection matrix and returns `(N,2)` pixels.

```python
px = project(P_3x4, verts_Nx3)   # 3D verts → image pixels
```

| Function | In | Out |
|----------|----|----|
| `aa2mtx(pos_aa)` | `(6,)` xArm pose (mm + axis-angle) | `(4,4)` in metres |
| `to_homo(x)` | `(N,D)` | `(N,D+1)` |
| `project(proj_mtx, x)` | `(3,4)` matrix, `(N,3)`/`(N,4)` pts | `(N,2)` pixels |
| `SOLVE_XA_B(A, B)` | two `(N,3)` sets | `(4,4)` s.t. `T @ A ≈ B` |

---

## 2. Multi-view triangulation — [`triangulate.py`](../../paradex/transforms/triangulate.py)

```python
from paradex.transforms.triangulate import triangulation, _triangulate
```

### `triangulation(corners, projections, threshold=1.5, iterations=100)` — RANSAC (use this)
Multi-view 2D → 3D with reprojection-inlier RANSAC.

```python
pt3d  = triangulation(corners_Nx2, proj_Nx3x4)          # one point  -> (3,)
pts3d = triangulation(corners_NxPx2, proj_Nx3x4)        # P points   -> (P,3)
```

- `corners`: `(N, 2)` image points across `N` cameras, **or** `(N, P, 2)` for `P` points (each point
  gets its own independent RANSAC).
- `projections`: `(N, 3, 4)` per-camera matrices, **index-aligned with `corners`**.
- `threshold` is in **pixels**; `corners` must already be **undistorted**.
- Returns `None` if fewer than 2 views (or any point fails in the multi-point path).

### `_triangulate(corners, projections)` — one DLT solve, no RANSAC
The bare linear solve. Same inputs; returns the raw `(3,)` estimate with no outlier rejection. Prefer
`triangulation` unless you know every view is an inlier.

| Function | In | Out |
|----------|----|----|
| `triangulation(corners, projections, threshold, iterations)` | `(N,2)`/`(N,P,2)` + `(N,3,4)` | `(3,)` or `(P,3)`, `None` on failure |
| `_triangulate(corners, projections)` | `(N,2)`/`(N,P,2)` + `(N,3,4)` | `(3,)` (or stacked), `None` if `<2` views |

---

## 3. Static frame conventions — [`coordinate.py`](../../paradex/transforms/coordinate.py)

No functions — two module-level dicts of hand-authored constant 4×4 matrices, keyed by device-name
string. Import and index directly.

```python
from paradex.transforms.coordinate import DEVICE2WRIST, DEVICE2GLOBAL

T_w = DEVICE2WRIST["xarm"]      # T^{predefined wrist}_{device wrist}
T_g = DEVICE2GLOBAL["franka"]   # T^{predefined global}_{device}  (arms only)
```

| Dict | Keys | Meaning |
|------|------|---------|
| `DEVICE2WRIST` | `xarm`, `franka`, `allegro`, `inspire`, `xsens_left`, `xsens_right` | device wrist frame → predefined wrist frame |
| `DEVICE2GLOBAL` | `xarm`, `franka` (**arms only**, both identity) | device frame → predefined global frame |

There is **no** hand/xsens entry in `DEVICE2GLOBAL` — retargeting works around this with `np.eye(4)`
for hands (see [`paradex/retargetor/unimanual.py`](../../paradex/retargetor/unimanual.py)).

> **Import by the right path.** It is `paradex.transforms.coordinate`, **not** `paradex.geometry.coordinate`
> (there is no `paradex/geometry/` package). `src/util/robot/merge_urdf.py` has the broken import — see
> [`internals.md`](internals.md).
