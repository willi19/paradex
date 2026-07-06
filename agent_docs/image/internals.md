# paradex/image — Internals (for editors)

You're changing `paradex/image/` itself. Seven files; the weight is in `image_dict.py`
(container + geometry) and `projection.py` (the nvdiffrast renderer). This layer *consumes*
calibration — it never solves it; param shapes are owned by `agent_docs/calibration/`.

> Just calling these? Read [`usage.md`](usage.md).

---

## image_dict.py — `ImageDict`

The multi-cam hub: `self.images` (`{serial: ndarray}`) + `self.intrinsic` + `self.extrinsic` +
optional `self.path`, plus a `self._cache` dict that memoizes derived state:
- `_cache['proj_mtx']` = `get_cammtx(intrinsic, extrinsic)` (3×4 per serial, built from
  `intrinsics_undistort`) — populated lazily by every triangulate/project method.
- `_cache['undistort_map']` = `{serial: precomute_undistort_map(...)}`.
- `_cache['render']` = a `BatchRenderer` (built on first `project_mesh`).

**The cache never invalidates.** If you mutate `intrinsic`/`extrinsic` after any geometry call,
`proj_mtx`/`render` are stale. Editing invalidation? Clear `_cache` in `set_camparam` /
`load_current_camparam` (currently they don't) — but check callers first: most code builds a
fresh `ImageDict` per capture, which is why the staleness has never bitten.

`from_path` resolution order: `<path>/images/` → `<path>/raw/images/` → `<path>` itself;
calibration loads only if `<path>/cam_param/` exists (`load_camparam`). Batch ops (`apply`,
`map_images`) take `use_camparam=` to also pass `intrinsic=`/`extrinsic=` into `func`, raising
if a serial lacks params.

Geometry methods all follow: ensure `_cache['proj_mtx']` → detect/collect 2D per id per view →
`triangulation(pts2d, proj_mtx)` (from `paradex.transforms.triangulate`). `project_pointcloud`
does the homogeneous `proj @ X` by hand; `project_mesh` routes through `BatchRenderer.render`
then `overlay_mask`.

### Known bugs in this file (don't copy, fix deliberately)
- `from_path` (~L67): `raise (f"No images found …")` **raises a str** (→ `TypeError`, not a clear
  error), and the bare `ValueError` on the next line is dead code. Intended: `raise ValueError(...)`.
- `__repr__` (~L174): interpolates `self.serials`, which doesn't exist (the property is
  `serial_list`) → `AttributeError` whenever repr runs. Fix to `self.serial_list`.
- `undistort()` builds the whole `undistort_map` cache inside a `for serial` loop guarded by
  `'undistort_map' not in self._cache` — works, but the shadowed inner `serial` loop is
  confusing; safe to hoist out of the loop.

These are latent because callers avoid the paths that hit them. If you touch the file, fixing
the two real bugs above is low-risk and worth it.

---

## aruco.py — marker/corner detection

> **Board *semantics* live in `agent_docs/calibration/`, not here.** That doc's "Charuco
> detection (paradex/image/aruco.py)" section already owns `charuco_info.json`, the
> `setLegacyPattern` gotcha, and board object-point scale. This section covers only the
> image-side detection surface — don't duplicate the calibration content.

`detect_aruco(img, dict_type)` → `(corners, ids)` (ids `(N,1)` int, or `[]`/`(0,1)` when none).
`detect_charuco(img)` → `{b_id: {checkerCorner (n,2), checkerIDs (n,)}}` per board;
`merge_charuco_detection` concatenates boards with per-board id offsets `Σ (numX-1)(numY-1)`.
Correspondence: `find_common_indices(ids1, ids2) -> (idx1, idx2)`.

Editor-relevant internals:
- Import-time side effect: `boardinfo_dict` and `_charuco_board_cache` are built when the module
  is imported (reads `config_dir/charuco_info.json`) — import **fails** if the config lacks it.
- Detectors (`_aruco_detector_cache`/`_charuco_detector_cache`) use **default**
  `DetectorParameters` — the tuning lever if detection is flaky.
- `get_board_cor()`'s `× 0.05` object-point scale is board-`"3"`-specific — the *why* (and the
  double-scale risk on the metric boards) is in the calibration doc; check it before generalizing.

---

## projection.py — `BatchRenderer` (nvdiffrast)

GPU mesh rasterizer, one instance per camera set. `import nvdiffrast` is wrapped in try/except,
so the module imports without it but any `BatchRenderer()` then dies — `project_mesh` needs
CUDA + nvdiffrast + a working `RasterizeCudaContext`.

Pipeline: `mesh_to_obj_dict` (trimesh/o3d → torch verts/faces/colors) → `transform_pos` to clip
space with `intr_opengl @ flip_z @ cam_extrs_t` → `dr.rasterize/interpolate/antialias`. Two
frames-of-reference conversions are load-bearing:
- `intr_opencv_to_opengl_proj` turns the OpenCV K into a GL projection; `flip_z = diag(1,-1,-1,1)`
  flips OpenCV (+Z fwd, +Y down) to GL (−Z fwd, +Y up).
- Outputs are `torch.flip(..., dims=[1])` (vertical flip) before returning, undoing the GL
  y-origin. Change one of these and renders come out mirrored/black.

Return-shape **inconsistency** to preserve or fix consciously:
- `render()` → `mask_dict` is **bool** (`[:,:,0].astype(np.bool_)`), color/depth are 2D per serial.
- `render_multi()` → `mask_dict` is **float** and keeps the trailing channel; also returns an
  extra `id_dict` (1-indexed instance ids, 0=bg). `project_mesh` uses `render` (bool mask).

`_denormalize_depth` inverts the rasterizer's inverse-depth encoding (`near`/`far` from the
ctor, default 0.01/2 m); background (triangle 0) is forced to `far`.

---

## undistort.py, merge.py, grid.py, overlay.py — leaf ops

- `undistort.py`: `precomute_undistort_map` (note the misspelling, it's public API) builds a
  `CV_16SC2` remap; the `_torch` pair builds a `grid_sample` grid for GPU video. `undistort_points`
  re-applies `intrinsics_undistort` after `cv2.undistortPoints`. `remap_corners` hardcodes
  `w,h = 2048,1536` and recomputes its own optimal matrix — legacy, prefer the intrinsic-driven path.
- `merge.py` `merge_image`: hardcodes a 2048×1536 canvas and auto-sizes a caption font; preview
  only. `grid.py` `make_image_grid` is the label-less tiling variant (resizes odd-sized tiles).
- `overlay.py` `overlay_mask`: **in-place** alpha blend, asserts `mask.ndim==2` and matching HxW;
  mutates the passed image (callers that need the original must `.copy()` first).

---

## Consumers (verify against these when you change signatures)
- `ImageDict`: `src/process/object_turntable/*`, `src/util/register_object/*`,
  `src/validate/calibration/`, `src/validate/robot/`, `src/process/miyungpa/visualizer.py`.
- `aruco` detection + `find_common_indices`/`merge_charuco_detection`/`get_board_cor`:
  `src/process/object_turntable/*` (charuco pipeline).
- `merge_image`: `src/validate/camera_system/*`, `src/util/camera_tuning/live_tuner.py`.
- `undistort` maps: `src/validate/upload_raw_video/`, video pipelines.

No test suite — validate by re-running a turntable charuco extract or a camera preview.
