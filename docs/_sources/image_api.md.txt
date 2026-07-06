# Image Layer — API

Method reference for the 2D image subsystem: parameters (input) and return values (output). For
the architecture and how these fit together, see the {doc}`overview <image>`.

Signatures are verified against the code. The subsystem lives in `paradex/image/`. Everything is
keyed by **camera serial string**.

Each entry is collapsed below — click to expand.

:::{dropdown} `ImageDict` (`paradex/image/image_dict.py`)
:open:

Multi-camera image container wrapping `{serial: image}` + `intrinsic`/`extrinsic`, with batch
operations. Geometry ops go through `get_cammtx` (3×4 per serial from `intrinsics_undistort`) —
feed them **undistorted** 2D.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `ImageDict(images, intrinsic=None, extrinsic=None, path=None)` | `images: {str: ndarray}`, `intrinsic: {str: dict}`, `extrinsic: {str: ndarray}`, `path` | instance | Store dict + calibration; empty `_cache`. |
| `ImageDict.from_path(path)` *(classmethod)* | `path: str\|Path` | `ImageDict` | Load `images/` → `raw/images/` → `path`; load calibration iff `cam_param/` exists. Raises if no images found. |
| `.undistort(save_path=None)` | `save_path` | new `ImageDict` | Per-serial `cv2.remap` via a cached undistort map; saves if `save_path` given. Needs `intrinsic`. |
| `.apply(func, use_camparam=False, **kw)` | `func(img, **kw)`, `use_camparam: bool` | `{serial: result}` | Run `func` on every image; `use_camparam=True` also passes `intrinsic=`/`extrinsic=` (raises if missing). |
| `.map_images(func, use_camparam=False, **kw)` | as above | new `ImageDict` | Like `apply`, rewrapping results as images. |
| `.triangulate_markers(dict_type='6X6_1000')` | `dict_type: str` | `(marker_2d, marker_3d)` | Detect ArUco per view, group by id, triangulate → `{id: (3,)}`. `marker_2d` holds per-id 2D + cammtx lists. |
| `.triangulate_charuco()` | — | `{board_id: {checkerIDs, checkerCorner}}` | Detect ChArUco per view, triangulate each corner id. |
| `.traingulate_points(points_2d)` | `points_2d: {serial: (N,2)}` | `(N,3)` | Triangulate matched 2D across views (method name misspelled in source). |
| `.project_pointcloud(points_3d)` | `points_3d: (N,3)` or `(3,)` | `{serial: (N,2)}` | Project world points into every camera. |
| `.project_mesh(object, color=(0,255,0), alpha=0.5)` | `object`: trimesh/o3d mesh, `color`, `alpha` | new `ImageDict` | Render mesh via `BatchRenderer`, overlay each mask. **Needs CUDA + nvdiffrast.** |
| `.merge(image_text={})` | `image_text: {serial: str}` | `np.ndarray` | Labeled grid preview (`merge_image`). |
| `.draw_keypoint(keypoints, color, radius=3, thickness=-1)` | `keypoints: {serial: (N,2)}` | new `ImageDict` | Draw dots per serial. |
| `.set_camparam(intrinsic, extrinsic)` / `.load_current_camparam()` | params / — | `None` | Attach calibration (explicit / latest global). |
| `.save(path=None)` / `.save_camparam(path=None)` | `path` | `None` | Write `images/*.png` / `cam_param/{intrinsics,extrinsics}.json`. |
| `.copy()` | — | `ImageDict` | Deep copy of images + calibration. |

**Dict interface**: `imgs[serial]`, `imgs[serial]=img`, `len(imgs)`, `for serial,img in imgs`,
`serial in imgs`, `.serial_list`, `.keys()/.values()/.items()`, `.update(dict)`, `.update_path(path)`.

:::{warning}
`_cache` (`proj_mtx`, `undistort_map`, `render`) is lazily filled and never invalidated — mutating
calibration afterward is unsafe; build a fresh `ImageDict`. `__repr__` references a nonexistent
`self.serials` (→ `AttributeError`); use `.serial_list`.
:::
:::

:::{dropdown} Detection — `aruco.py` (`paradex/image/aruco.py`)

Single-image detectors + helpers. Board *semantics* (config, scale, `setLegacyPattern`) are
documented with the calibration subsystem; this is the call surface only.

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `detect_aruco(img, dict_type='6X6_1000')` | `img`, `dict_type: str` | `(corners, ids)` | `ids` is `(N,1)` int; returns `([], (0,1))` when none. Cached detector per dict-type. |
| `detect_charuco(img)` | `img` | `{b_id: {checkerCorner (n,2), checkerIDs (n,)}}` | Runs every board's cached `CharucoDetector`; omits boards with no detection. |
| `merge_charuco_detection(detection)` | `detection: {b_id: {...}}` | `{checkerCorner, checkerIDs}` | Concat all boards with per-board id offset `Σ(numX-1)(numY-1)`; empty → `(0,2)`/`(0,1)`. |
| `find_common_indices(ids1, ids2)` | two id arrays | `(idx1, idx2)` or `(None, None)` | Indices that align corners by shared id (sorted). |
| `get_board_cor()` | — | `{b_id: {checkerIDs, checkerCorner}}` | Board object points (`getChessboardCorners()` × 0.05 — board-`"3"`-specific scale). |
| `get_adjecent_ids()` | — | `{id: [neighbor_ids]}` | Corner adjacency graph (per-board offsets); used by turntable outlier checks. |
| `get_aruco_detector(dict_type)` / `get_charuco_detector()` | `dict_type` / — | detector(s) | Cached detector accessors; charuco applies `setLegacyPattern` per board. |
| `draw_aruco` / `draw_charuco` / `draw_keypoint` | `img`, corners, … | `img` | Debug drawing (mutate/return the image). |

Import-time: `boardinfo_dict` + `_charuco_board_cache` are loaded from
`config_dir/charuco_info.json` — import fails if it's absent.
:::

:::{dropdown} Undistortion — `undistort.py` (`paradex/image/undistort.py`)

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `undistort_img(img, intrinsic)` | `img`, `intrinsic: dict` | `ndarray` | One-shot `cv2.undistort` with `original_intrinsics`/`dist_params` → `intrinsics_undistort`. |
| `undistort_points(pts, intrinsic)` | `pts: (N,1,2)`/`(N,2)`, `intrinsic` | `(N,2)` | `cv2.undistortPoints` then re-apply `intrinsics_undistort`. |
| `precomute_undistort_map(intrinsic)` | `intrinsic` | `(new_cammtx, mapx, mapy)` | Build a `CV_16SC2` remap (name misspelled, public API). |
| `apply_undistort_map(img, mapx, mapy)` | `img`, maps | `ndarray` | `cv2.remap` (fast, for many frames). |
| `precompute_undistort_map_torch(intrinsic, device='cuda')` | `intrinsic`, `device` | `(new_cammtx, grid_tensor)` | GPU `grid_sample` grid `(1,H,W,2)`. |
| `apply_undistort_torch(img_bgr, grid_tensor)` | `img_bgr: uint8 HxWx3`, `grid_tensor` | `uint8 HxWx3` | GPU undistort via `grid_sample`. |
| `remap_corners(corners, cammtx, dist_coef, sn, img=None)` | corners, K, dist, … | `(mapped_pixels, new_cammtx)` | Legacy; hardcodes `2048×1536` and its own optimal matrix. Prefer the intrinsic-driven path. |
:::

:::{dropdown} Mesh rendering — `projection.py` (`paradex/image/projection.py`)

Requires CUDA + `nvdiffrast` (imported in a `try/except`; `BatchRenderer()` fails without it).

| Method / function | Input | Output | Description |
|-------------------|-------|--------|-------------|
| `BatchRenderer(intrinsics, extrinsics, near=0.01, far=2)` | calibration dicts, clip planes | instance | Precompute GL projections, `flip_z`, stacked extrinsics; one batch = all serials (sorted). |
| `.render(mesh)` | trimesh/o3d mesh | `(color_dict, mask_dict, depth_dict)` | Single mesh; `mask` is **bool** per serial. |
| `.render_multi(mesh_list)` | `list[mesh]` | `(color, mask, depth, id_dict)` | Multi-mesh; `mask` is **float**, plus 1-indexed instance-id map (0=bg). |
| `mesh_to_obj_dict(mesh, device='cuda', texture_type='vertex_color')` | mesh, … | `dict` | Convert trimesh/o3d → torch verts/faces/colors (or UVs+texture). |
| `intr_opencv_to_opengl_proj(K, width, height, near, far)` | K, size, planes | `(4,4)` | OpenCV intrinsics → GL projection. |
| `project_point(verts, cammtx, image, color)` | verts, `(3,4)`, image | `image` | Draw projected 3D points as circles. |

Depth: `_denormalize_depth` inverts the rasterizer's inverse-depth encoding; background is forced
to `far`. Outputs are vertically flipped (`torch.flip`) to undo the GL y-origin.
:::

:::{dropdown} Preview & compositing — `merge.py` / `grid.py` / `overlay.py`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `merge_image(image_dict, image_text={}, put_text=True)` | `{serial: img}`, `{serial: str}`, `bool` | `np.ndarray` | Labeled near-square grid; auto-sizes caption font. Assumes ~2048×1536 sources. |
| `make_image_grid(images)` | `list[ndarray]` | `np.ndarray` | Bare tiling; resizes odd-sized tiles. Empty list → `(1,1,3)`. |
| `overlay_mask(image, mask, color, alpha=0.7)` | `image`, `mask: bool HxW`, `color`, `alpha` | `image` | **In-place** alpha blend; asserts `mask.ndim==2` and matching HxW. |
:::
