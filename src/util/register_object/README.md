# Object Registration (marker offsets)

Computes the rigid offset between ArUco markers attached to an object and the object's canonical mesh frame, so the object's 6D pose can later be recovered from detected markers alone.

## Scripts
| File | Purpose |
|------|---------|
| `box.py` | Registers a parametric box object. Mesh is created on the fly via `trimesh.creation.box` with hard-coded extents (23.4 x 5.5 x 21.0 cm). |
| `pringles.py` | Registers the "pringles" object. Loads a raw `.obj` mesh from disk instead of a primitive. |

Both scripts share the same pipeline; they differ only in mesh source and the z-offset used to place the object origin relative to the template plane.

## Usage
```bash
python src/util/register_object/box.py
python src/util/register_object/pringles.py
```
No CLI args. Each iterates over every capture index found under `~/shared_data/obj_register/<object>/`.

Pipeline per index:
1. Load multi-view images via `ImageDict.from_path`; undistort if no `image/` subdir exists yet.
2. Triangulate the object's markers (default dict) and the floor/template markers (`4X4_50`).
3. Build the object frame from template markers 0-3 (x/y from marker midpoints, z = cross product), origin pushed down by the object's half-depth / fixed offset.
4. Express each object marker in the object frame -> `marker_offset` dict.
5. Save the offsets and write a debug image overlay (markers + projected mesh).

## Inputs & Outputs
- Reads: `~/shared_data/obj_register/<object>/<index>/` capture (multi-view images + camera params consumed by `ImageDict`). `pringles.py` also reads `~/shared_data/RSS2026_Mingi/object/paradex/pringles/raw_mesh/pringles.obj`.
- Writes:
  - `~/shared_data/RSS2026_Mingi/marker_offset/<object>/<index>.npy` — dict `{marker_id: (N,3) points in object frame}`.
  - `~/shared_data/obj_register/<object>/<index>/debug/` — debug images with reprojected markers and mesh.

## Related
- [`paradex/image/image_dict.py`](../../../paradex/image/image_dict.py) — `ImageDict` (undistort, `triangulate_markers`, `project_pointcloud`, `draw_keypoint`, `project_mesh`, `save`).
- [`paradex/utils/path.py`](../../../paradex/utils/path.py) — `shared_dir`.
- Downstream consumers: 6D pose estimation in [`src/object6d/`](../../object6d) / [`src/inference/`](../../inference).

> Note: the large commented-out block at the bottom of each file is an older renderer-based debug path (BatchRenderer / nvdiffrast) kept for reference; it is not executed.
