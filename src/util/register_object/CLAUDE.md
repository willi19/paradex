# CLAUDE.md — src/util/register_object

## Purpose
Build per-object `marker_offset` tables: the 3D positions of each attached ArUco marker expressed in the object's canonical mesh frame. These let downstream pose estimators recover object 6D pose from marker detections.

## Files
- `box.py` — object = `trimesh.creation.box(extents=[0.234, 0.055, 0.210])`; origin offset = `depth/2` along z.
- `pringles.py` — object = `trimesh.load(.../RSS2026_Mingi/object/paradex/pringles/raw_mesh/pringles.obj)`; origin offset = fixed `0.096` along z.

Shared flow (both files, identical except mesh + offset):
1. `ImageDict.from_path(obj_register/<obj>/<index>)`; if no `image/` subdir, call `.undistort()` then reload.
2. `obj_marker_3d = img_dict.triangulate_markers()` (object markers, default dict).
3. `template_marker_3d = img_dict.triangulate_markers("4X4_50")` (floor/template markers).
4. Frame from template markers 0..3: `x_dir=(p1+p2)-(p3+p4)`, `y_dir=(p2+p4)-(p1+p3)`, `z=cross(x,y)`, origin = mean(p1..p4) - offset*z. Build `obj_T`, invert to `obj_cor`.
5. `marker_offset[mid] = (obj_cor @ obj_marker_3d[mid])` -> save `.npy`.
6. `mesh.apply_transform(obj_T)`, `project_mesh`, `save` debug overlay.

## paradex modules used
- `paradex.image.image_dict.ImageDict`
- `paradex.utils.path.shared_dir`

## Data flow & IO
- In: `{shared_dir}/obj_register/<obj>/<index>/` (multi-view capture). pringles also reads raw mesh under `RSS2026_Mingi/object/paradex/pringles/raw_mesh/`.
- Out: `{shared_dir}/RSS2026_Mingi/marker_offset/<obj>/<index>.npy` and `obj_register/<obj>/<index>/debug/`.

## When working here
- To register a new object, copy one of these and change `obj_name`/extents/mesh path and the z-offset (object origin placement relative to the template plane).
- These run over ALL indices in the object's `obj_register` dir; no arg filtering.

## Gotchas
- Object frame depends on template markers being IDs 0..3 present in `4X4_50` triangulation; missing ones break the `np.mean(template_marker_3d[i])` lookups.
- The trailing commented block (BatchRenderer / nvdiffrast mask projection, "book" object) is dead reference code — do not assume it runs.
- Hard-coded `RSS2026_Mingi` output namespace; not generic.
