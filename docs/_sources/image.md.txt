# Image Layer

Overview of Paradex's 2D multi-camera image subsystem — the container (`ImageDict`) and the
operations it fans across every camera: undistortion, marker detection, multi-view triangulation,
and mesh↔image projection. Read this for the mental model; for method signatures see the
{doc}`API reference <image_api>`.

- Core library: `paradex/image/` (`image_dict.py`, `aruco.py`, `undistort.py`, `projection.py`, `merge.py`, `grid.py`, `overlay.py`)
- Calibration params it consumes: `paradex/calibration/` — see {doc}`camera_system <camera_system>` and the calibration guide
- Generated per-symbol API: {doc}`API Reference <autoapi/index>`

:::{note}
Everything is keyed by **camera serial string**: a multi-camera image set is
`Dict[str, np.ndarray]`. This convention is repo-wide.
:::

---

## 1. What the subsystem does

Once cameras are calibrated, most vision work is the same shape: *take one operation, run it on
every camera's image, and combine the per-view results using the calibration*. `ImageDict` is the
hub that makes that one line of code. It wraps the serial→image dict **plus** the calibration
(`intrinsic`/`extrinsic`) and exposes batch ops; the other files are the leaf operations it (and
direct callers) build on.

```{mermaid}
flowchart TB
    subgraph HUB["ImageDict — {serial: image} + intrinsic/extrinsic"]
      direction LR
      APPLY["apply / map_images<br/>(fan any func over all cams)"]
      GEO["triangulate_* / project_*<br/>(combine views via calibration)"]
    end
    ARUCO["aruco.py<br/>detect_aruco / detect_charuco"] --> APPLY
    UND["undistort.py<br/>maps + points"] --> HUB
    PROJ["projection.py<br/>BatchRenderer (nvdiffrast)"] --> GEO
    HUB --> MERGE["merge.py / grid.py / overlay.py<br/>(preview + compositing)"]
```

What `ImageDict` actually contributes is **fan-out + multi-view geometry**, *not* detection:
detection lives in `aruco.py` as standalone functions; `ImageDict` runs them across all cameras
(`apply`) and, for `triangulate_*`, lifts the per-view 2D into 3D with the calibration.

---

## 2. Core concepts

| Term | Meaning |
|------|---------|
| **ImageDict** | The container: `{serial: image}` + optional `intrinsic`/`extrinsic` + `path`. Dict-like (`imgs[serial]`, iterate, `len`, `in`). |
| **Batch op** | A method that runs over every serial: `apply` (→ dict of results), `map_images`/`undistort`/`project_*` (→ new `ImageDict` or per-serial arrays). |
| **Projection matrix** | `get_cammtx(intrinsic, extrinsic)` → 3×4 per serial, built from **`intrinsics_undistort`**. Cached on first geometry call; every triangulate/project uses it. |
| **Distorted vs undistorted** | Raw frames are lens-distorted; `intrinsics_undistort` applies only after undistortion. `get_cammtx` (hence all geometry) assumes **undistorted** 2D — mixing the two is the classic silent bug. |
| **Detection** | `aruco.py` functions on a single image; ImageDict fans them out. Board *semantics* (which board, scale, `setLegacyPattern`) belong to calibration, not here. |

---

## 3. ImageDict — the hub

```python
from paradex.image.image_dict import ImageDict

imgs = ImageDict.from_path(path)     # loads images/ (or raw/images/); calibration if cam_param/ exists
imgs.load_current_camparam()         # or attach the latest global calibration
```

`from_path` resolution: `<path>/images/` → `<path>/raw/images/` → `<path>` itself; calibration
loads only when `<path>/cam_param/` is present.

| Op | Returns | Purpose |
|----|---------|---------|
| `undistort(save_path=None)` | new `ImageDict` | per-serial `cv2.remap`; caches the map. |
| `apply(func, use_camparam=False, **kw)` | `{serial: result}` | run any `func(img, **kw)` on all cams; `use_camparam` also passes `intrinsic=`/`extrinsic=`. |
| `map_images(func, …)` | new `ImageDict` | like `apply`, rewrapping images. |
| `triangulate_markers(dict_type='6X6_1000')` | `(marker_2d, marker_3d)` | detect ArUco in every view → triangulate per id. |
| `triangulate_charuco()` | `{board_id: {checkerIDs, checkerCorner}}` | same for ChArUco corners. |
| `traingulate_points(pts2d)` | `(N,3)` | triangulate `{serial: (N,2)}` (method name is misspelled). |
| `project_pointcloud(pts3d)` | `{serial: (N,2)}` | project world points into every camera. |
| `project_mesh(obj, color, alpha)` | new `ImageDict` | GPU-render a mesh + overlay its mask (needs CUDA + nvdiffrast). |
| `merge(image_text={})` | `np.ndarray` | labeled grid preview. |
| `save` / `save_camparam` | — | write `images/*.png` / `cam_param/*.json`. |

:::{warning}
The `_cache` (proj matrices, undistort maps, renderer) is populated lazily and **never
invalidated**. If you mutate `intrinsic`/`extrinsic` after a geometry call, build a fresh
`ImageDict`. Two latent bugs: `ImageDict.__repr__` references a nonexistent attribute (don't rely
on `repr`/`print`), and `from_path` on an empty directory raises the wrong way. See the agent
internals note (`agent_docs/image/internals.md`).
:::

---

## 4. Detection (aruco.py)

`detect_aruco(img, dict_type)` and `detect_charuco(img)` are standalone single-image functions;
run them across a set with `imgs.apply(...)` or let `triangulate_*` call them for you.

```python
corners, ids = detect_aruco(img, dict_type='6X6_1000')
det = detect_charuco(img)                       # {board_id: {checkerCorner, checkerIDs}}
flat = merge_charuco_detection(det)             # concat boards with per-board id offsets
idx1, idx2 = find_common_indices(ids1, ids2)    # match two views by id
```

Detectors run with **default** OpenCV parameters (the first lever if detection is flaky) and are
cached per dict-type/board.

:::{note}
**Board configuration is a calibration concern, not an image one.** Which board is used,
`charuco_info.json`, object-point scale, and the `setLegacyPattern` gotcha are documented with
the calibration subsystem. This page covers only the detection call surface.
:::

---

## 5. Undistortion (undistort.py)

```python
from paradex.image.undistort import undistort_img, undistort_points
und_img = undistort_img(img, intrinsic[serial])        # one-shot cv2.undistort
und_pts = undistort_points(pts, intrinsic[serial])     # distorted → undistorted pixels
```

For many frames from one camera, precompute the remap once (`precomute_undistort_map` +
`apply_undistort_map`, CPU) or the GPU grid (`precompute_undistort_map_torch` +
`apply_undistort_torch`, used by the video pipeline). After undistorting, use `intrinsics_undistort`
with **no** distortion coefficients.

---

## 6. Mesh rendering (projection.py)

`BatchRenderer` is an nvdiffrast GPU rasterizer, one instance per camera set, that renders a mesh
into every calibrated view (color + soft mask + depth). It powers `ImageDict.project_mesh` (used
by miyungpa overlay and 6D-pose validation).

```{mermaid}
flowchart LR
    MESH["trimesh / open3d mesh"] --> OBJ["mesh_to_obj_dict<br/>(torch verts/faces/colors)"]
    OBJ --> CLIP["transform_pos<br/>intr_opengl · flip_z · extrinsic"]
    CLIP --> RAST["dr.rasterize / interpolate / antialias"]
    RAST --> OUT["{serial: color, mask, depth}<br/>(+ id map for render_multi)"]
```

Two frame conversions are load-bearing: `intr_opencv_to_opengl_proj` (OpenCV K → GL projection)
and `flip_z = diag(1,-1,-1,1)` (OpenCV +Z-fwd/+Y-down → GL), with a final vertical `torch.flip`
to undo the GL y-origin. `render()` returns a **bool** mask; `render_multi()` returns a **float**
mask plus an instance-id map — a deliberate inconsistency to preserve.

---

## 7. Preview & compositing (merge / grid / overlay)

| Function | Location | Purpose |
|----------|----------|---------|
| `merge_image(image_dict, image_text={})` | `merge.py` | Labeled near-square grid (serials as captions). Assumes ~2048×1536 sources. Live previews / sync checks. |
| `make_image_grid(images)` | `grid.py` | Bare tiling of a **list** of images, no labels. |
| `overlay_mask(image, mask, color, alpha)` | `overlay.py` | **In-place** alpha-blend of a bool mask; asserts matching HxW. Callers needing the original must `.copy()` first. |

---

## 8. Component & file reference

| Component | File | Responsibility |
|-----------|------|----------------|
| `ImageDict` | `paradex/image/image_dict.py` | Multi-cam container + batch undistort/detect/triangulate/project. |
| `detect_aruco` / `detect_charuco` / `merge_charuco_detection` / `find_common_indices` / `get_board_cor` | `paradex/image/aruco.py` | ArUco/ChArUco detection, id-merge, correspondence, board object points. |
| undistort maps + `undistort_img` / `undistort_points` | `paradex/image/undistort.py` | CPU/GPU undistortion. |
| `BatchRenderer` / `mesh_to_obj_dict` | `paradex/image/projection.py` | nvdiffrast mesh rasterizer (color/mask/depth/id). |
| `merge_image` | `paradex/image/merge.py` | Labeled grid preview. |
| `make_image_grid` | `paradex/image/grid.py` | Label-less tiling. |
| `overlay_mask` | `paradex/image/overlay.py` | In-place mask alpha blend. |
| Triangulation solver | `paradex/transforms/triangulate.py` | `triangulation(pts2d, proj_mtx)` backing the `triangulate_*` methods. |

---

## 9. Downstream consumers

| Consumer | Uses |
|----------|------|
| `src/process/object_turntable/*` | `ImageDict`, charuco detection, `find_common_indices`, `get_board_cor` for the turntable reconstruction pipeline. |
| `src/util/register_object/*` | `ImageDict` for object marker registration. |
| `src/validate/calibration/`, `src/validate/robot/` | `ImageDict` for calibration/overlay validation. |
| `src/process/miyungpa/visualizer.py` | `ImageDict` + `merge_image`. |
| `src/validate/camera_system/*`, `src/util/camera_tuning/live_tuner.py` | `merge_image` for live previews. |

Method-by-method API (parameters / returns): {doc}`Image Layer — API <image_api>`.
