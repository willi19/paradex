"""
Fingertip STL editor — load, modify, preview, save.

Usage:
    python src/util/edit_fingertip_mesh.py

Workflow:
    1. Set INPUT_PATH / OUTPUT_PATH at the top.
    2. Uncomment / edit only the operations you want under "EDIT OPS".
    3. Run. A trimesh viewer pops up before and after each operation if PREVIEW=True.
    4. Output STL gets written to OUTPUT_PATH.

All operations are in-place on `mesh`. Inspect `mesh.vertices` (N,3) and
`mesh.faces` (M,3) anytime to know what you're working with.
"""
import os
import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MESHES_DIR = "/home/temp_id/paradex/rsc/robot/allegro_v5/meshes"
INPUT_PATH = os.path.join(MESHES_DIR, "finger_tip.stl")
# Save to a new file first so you don't clobber the original until you're happy.
OUTPUT_PATH = os.path.join(MESHES_DIR, "finger_tip_edited.stl")

PREVIEW = True  # set False to skip the popup viewer between steps


def show(m, title=""):
    if not PREVIEW:
        return
    print(f"[{title}] vertices={len(m.vertices)} faces={len(m.faces)} "
          f"bounds={np.round(m.bounds, 4).tolist()} "
          f"extents={np.round(m.extents, 4).tolist()}")
    m.show()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
mesh = trimesh.load_mesh(INPUT_PATH, process=False)
assert isinstance(mesh, trimesh.Trimesh), f"Loaded {type(mesh)}; expected Trimesh."
print(f"Loaded: {INPUT_PATH}")
show(mesh, "loaded")


# ===========================================================================
# EDIT OPS — uncomment / edit the ones you need
# ===========================================================================

# --- 1) Whole-mesh affine transform -----------------------------------------
# Translate (mm — STL scale 0.001 → URDF is meters, but STL itself is mm here).
# tx, ty, tz = 0.0, 0.0, -2.0
# mesh.apply_translation([tx, ty, tz])
# show(mesh, "after translate")

# Rotate around axis through origin (radians).
# from trimesh.transformations import rotation_matrix
# mesh.apply_transform(rotation_matrix(angle=np.deg2rad(15), direction=[0, 1, 0]))
# show(mesh, "after rotate")

# Uniform / non-uniform scale.
# mesh.apply_scale([1.0, 1.0, 0.9])         # squish along Z
# show(mesh, "after scale")


# --- 2) Crop / cut --------------------------------------------------------
# Slice with a plane: keep half on +normal side. Plane is point + normal.
# mesh = mesh.slice_plane(plane_origin=[0, 0, 5], plane_normal=[0, 0, 1])
# show(mesh, "after slice")

# Crop by bounding box (axis-aligned). Anything outside the box is removed.
# bbox_min = np.array([-30, -10, -5])
# bbox_max = np.array([ 30,  10, 30])
# keep = np.all((mesh.vertices >= bbox_min) & (mesh.vertices <= bbox_max), axis=1)
# face_keep = keep[mesh.faces].all(axis=1)
# mesh.update_faces(face_keep)
# mesh.remove_unreferenced_vertices()
# show(mesh, "after bbox crop")


# --- 3) Vertex-level surgery ----------------------------------------------
# Remove vertices matching a condition (e.g. Z > 30).
# bad = mesh.vertices[:, 2] > 30.0
# mesh.update_vertices(~bad)                 # also drops dependent faces
# show(mesh, "after vertex filter")

# Move a vertex subset (e.g. all vertices with Z > 25 by -2 mm along Z).
# sel = mesh.vertices[:, 2] > 25.0
# mesh.vertices[sel] += np.array([0, 0, -2.0])
# show(mesh, "after vertex shift")


# --- 4) Smoothing / topology -----------------------------------------------
# Laplacian smoothing (iterations: more = smoother but shrinks).
# mesh = trimesh.smoothing.filter_humphrey(mesh, iterations=5)
# show(mesh, "after smooth")

# Decimate (reduce face count). Needs `pip install fast-simplification` or
# trimesh's built-in (uses open3d if available).
# mesh = mesh.simplify_quadric_decimation(face_count=len(mesh.faces) // 2)
# show(mesh, "after decimate")

# Subdivide (split each face into 4).
# mesh = mesh.subdivide()
# show(mesh, "after subdivide")


# --- 5) Repair --------------------------------------------------------------
# mesh.remove_duplicate_faces()
# mesh.remove_unreferenced_vertices()
# mesh.fill_holes()
# mesh.fix_normals()


# ===========================================================================
# Save
# ===========================================================================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
mesh.export(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")
print(f"Final: vertices={len(mesh.vertices)} faces={len(mesh.faces)} "
      f"watertight={mesh.is_watertight}")
