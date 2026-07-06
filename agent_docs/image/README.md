# agent_docs/image — agent orientation

Docs for **AI agents working on `paradex/image/`** — the multi-camera 2D-image layer: an
image container (`ImageDict`), marker detection (`aruco`), undistortion, mesh↔image projection,
and grid/overlay compositing. Routed here from the repo-root `AGENTS.md`. Read **one** file
for your task; don't scan everything.

Mental model: everything is keyed by **camera serial string** (`Dict[serial, np.ndarray]`).
`ImageDict` is the hub — it wraps that dict *plus* the calibration (`intrinsic`/`extrinsic`
from `paradex/calibration/`) and exposes batch ops (undistort, detect, triangulate, project).
The other files are the leaf operations `ImageDict` (and callers) build on.

| Your task | Read |
|-----------|------|
| **Use** `ImageDict` / detect markers / triangulate / project a mesh into cameras | [`usage.md`](usage.md) |
| Undistort images or 2D points (CPU or GPU) | [`usage.md`](usage.md) §undistort |
| Merge many cameras into one labeled preview | [`usage.md`](usage.md) §merge |
| **Edit** `ImageDict`, the detectors, the nvdiffrast renderer, undistort maths | [`internals.md`](internals.md) |
| How calibration params (`intrinsic`/`extrinsic`, distorted vs undistorted) are shaped | `agent_docs/calibration/README.md` (source of truth — this layer consumes it) |

Rule of thumb: **calling** these modules → `usage.md`; **editing** them → `internals.md`.
`ImageDict` also appears in the calibration doc as the calibration *consumer*; that doc owns the
param format, this one owns the image ops.

## File map
| File | What it is |
|------|-----------|
| `image_dict.py` | `ImageDict` — the multi-cam container + all batch geometry ops |
| `aruco.py` | ArUco / ChArUco **detection**, board caches, id-merge, drawing helpers |
| `undistort.py` | undistort **maps** (cpu `remap` + torch `grid_sample`), `undistort_img`/`undistort_points` |
| `projection.py` | `BatchRenderer` — nvdiffrast GPU mesh rasterizer (color/mask/depth) + OpenCV→OpenGL proj |
| `merge.py` | `merge_image` — labeled square grid preview (assumes 2048×1536 sources) |
| `grid.py` | `make_image_grid` — bare tiling, no labels |
| `overlay.py` | `overlay_mask` — alpha-blend a boolean mask onto an image |

All paths relative to [`paradex/image/`](../../paradex/image/). Related: calibration params
(`agent_docs/calibration/`), triangulation solver (`paradex/transforms/triangulate.py`).
