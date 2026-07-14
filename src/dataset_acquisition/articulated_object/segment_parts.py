#!/usr/bin/env python3
"""Static/movable part segmentation from cross-state multiview mask consistency.

Stage A of part decomposition (REDESIGN_PLAN.md). It consumes the registration
output of register_states.py plus the SAM3 masks and the known mesh, and labels
every mesh vertex as body (static) or movable, then splits the mesh into a body
submesh and one submesh per connected movable part.

The lever: within a placement group the body does not move, only the articulated
part does, and register_states already gave each state a (group-shared) body pose.
Placing the known mesh at each state's body pose and reprojecting every vertex into
that state's masks, a BODY vertex lands inside the object mask in every state, while
a MOVABLE vertex lands inside only in the state(s) whose physical part matches the
mesh rest configuration and outside where the part has swung away. So the
discriminator is per-vertex support consistency across states: body = supported
everywhere; movable = supported in some states, absent in others. This is the
discrete, known-mesh form of PARIS's static/movable split, done by a multiview
mask vote (no neural field), and it only trusts states whose body registration fit
is good (fit_iou), so a weak pose cannot corrupt the labels.

Stage B adds a diagnostic part-observation layer.  It combines a conservative
``object mask - full rest mesh`` residual with within-placement RGB change maps,
then independently fits each Stage-A movable candidate to that positive-only
motion evidence.  A broad Stage-A candidate is subsequently split into local mesh
patches and each patch is tested at the body pose versus the fitted moving pose.
Patch certification is a JOINT explaining-away selection, not a set of independent
per-patch tests: a static surface that the true mover occludes or reveals shows
real image change at its own footprint, so "this patch moved" and "this patch's
occluder moved" predict the same pixels and cannot be separated one patch at a
time.  The depth-composite renderer already predicts that disocclusion when the
mover alone is displaced, so a greedy marginal-gain subset selection keeps the
mover and rejects revealed static surfaces whose change it already explains.  The
pool is cross-candidate: each canonical patch competes under every moving-candidate
SE(3) track, may borrow a reliable hinge track, and is exclusive to one track. The
selected track components are then refit and passed to Stage C rather than returning
to their broad source candidates.
This keeps SAM3 false negatives and occluded views as *unknown*, rather than
treating them as evidence that a real part is static.

Stage C turns a moving candidate into a closed solid part. Stage A/B only ever
score the camera-visible shell, so the handle back and interior stay 'body' and the
exported part is an open shell. The discriminator is an EXPLAINING-AWAY test (the
discrete form of PARIS's compositional split), not plain object-mask inclusion:
a flat part folded against the body never leaves the object silhouette at either
pose (its rest footprint is a subset of the body's), so 'inside the object mask'
carries zero bits there. Instead, for every vertex and every fitted state, movable
evidence = landing in the beyond-body RESIDUAL (pixels the registered body render
cannot explain) at the part pose, minus the same measure at the body pose (cancels
registration-rim noise); static evidence = EXITING the object mask at the part pose
(the part pose throws a true body vertex into free space). A state only contributes
motion evidence where the vertex does not exit in that same state (hits without a
misses-check let accidental residual grazes seed the whole mesh). Stage C trusts only
body/part registrations and residuals that pass per-state quality gates, then requires
motion evidence to recur across independent placement groups. The Stage-B
motion-refined shell is a SOFT seed: it can fill an unobservable back surface but can
never override definitely-static evidence. All connectivity runs in welded index space
(unwelded-OBJ safety); distance-based bridges are off by default because a nearby but
disconnected frame/hinge/metal piece is not topological evidence. The union of
definitely-static regions is walled off as the body, and the confident residual-backed
seed grows through the remaining ambiguous band to close the solid part.

Known limits (to fix later, not concept failures): the joint-adjacent part base is
ambiguous from mask consistency (it stays covered near the hinge) so recall of the
part is partial there; and a very coarse mesh can fragment one part. Precision is
strong (no body leakage). Later refinement can grow the part toward the joint using
the per-state beyond-body mask residual (Stage B) instead of consistency alone.

Self-contained apart from reusing the validated helpers in register_states.py
(the new pipeline base, not the legacy calc_states). numpy + cv2 + trimesh + PIL.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit as _njit
except ImportError:  # Optional acceleration; the NumPy/Python fallback remains correct.
    _njit = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_states as rs  # noqa: E402  (new-pipeline base; shared pose/mask/mesh helpers)

# Distinct colors for movable parts in the labeled mesh / overlays (RGB).
_PART_COLORS = [
    (220, 50, 50), (50, 120, 220), (60, 180, 75), (245, 165, 35),
    (145, 30, 180), (70, 190, 190), (240, 90, 190), (160, 110, 60),
]
_BODY_COLOR = (170, 170, 170)


# ===========================================================================
# Registration + mesh loading
# ===========================================================================


def _load_registration(output_dir: str) -> dict:
    """Per-state registration records written by register_states.py."""
    manifest_path = os.path.join(output_dir, "registration_manifest.json")
    states = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            states = (json.load(f).get("states") or {})
    if states:
        return states
    reg_root = os.path.join(output_dir, "registration")
    if os.path.isdir(reg_root):
        for sid in sorted(os.listdir(reg_root)):
            path = os.path.join(reg_root, sid, "registration.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    states[sid] = json.load(f)
    return states


# ===========================================================================
# Stage A: per-vertex support and labeling
# ===========================================================================


def _state_vertex_support(world_vertices, cameras, masks, dilate_px, min_obs_cams):
    """Per-vertex (support, observed) for one state.

    ``support`` = fraction of cameras (that see the vertex) whose mask covers it;
    ``observed`` = the vertex was in front of and inside at least ``min_obs_cams``
    camera frames. The mask is dilated a few pixels so registration-boundary jitter
    does not flip a body vertex to 'outside'.
    """
    import cv2

    n = world_vertices.shape[0]
    hits = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=np.float64)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for serial, cam in cameras.items():
        mask = masks.get(serial)
        if mask is None:
            continue
        height, width = mask.shape[:2]
        mdil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=max(0, int(dilate_px))) > 0 if dilate_px > 0 else mask
        pixels = rs._project_points(cam["projection"], world_vertices)
        depths = rs._depths_in_camera(cam["cam_from_world"], world_vertices)
        u = np.rint(pixels[:, 0])
        v = np.rint(pixels[:, 1])
        onimg = (
            np.isfinite(u) & np.isfinite(v) & np.isfinite(depths) & (depths > 0.0)
            & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        )
        idx = np.nonzero(onimg)[0]
        if idx.size == 0:
            continue
        seen[idx] += 1.0
        vi = v[idx].astype(np.int64)
        ui = u[idx].astype(np.int64)
        inside = mdil[vi, ui]
        hits[idx[inside]] += 1.0
    support = np.where(seen > 0, hits / np.maximum(seen, 1.0), 0.0)
    observed = seen >= float(min_obs_cams)
    return support, observed


def _grow_movable(seed, eligible, faces, max_iters=100, extra_edges=None):
    """Region-grow the confident movable seed along mesh edges into eligible vertices.

    Body vertices have ~0 support drop so they are not eligible; growth therefore
    stops cleanly at the joint, recovering the joint-adjacent part base that the
    strict seed threshold misses without leaking into the body. ``extra_edges``
    (component bridges) let the grow cross between disconnected mesh pieces.
    """
    if faces is None or faces.shape[0] == 0:
        if extra_edges is None or len(extra_edges) == 0:
            return seed
        edges = np.asarray(extra_edges, dtype=np.int64).reshape(-1, 2)
    else:
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
        if extra_edges is not None and len(extra_edges):
            edges = np.vstack([edges, np.asarray(extra_edges, dtype=edges.dtype).reshape(-1, 2)])
    a, b = edges[:, 0], edges[:, 1]
    movable = seed.copy()
    for _ in range(int(max_iters)):
        new_b = movable[a] & eligible[b] & ~movable[b]
        new_a = movable[b] & eligible[a] & ~movable[a]
        if not (new_b.any() or new_a.any()):
            break
        movable[b[new_b]] = True
        movable[a[new_a]] = True
    return movable


def _close_label_holes(movable, faces, close_frac, max_iters=5):
    """Fill interior label holes: flip a non-movable vertex to movable when most of
    its mesh neighbors are movable. Interior holes (surrounded by the part) fill;
    the body boundary does not (it has body neighbors), so the part becomes solid
    and connected without leaking into the body.
    """
    if faces is None or faces.shape[0] == 0:
        return movable
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    ue = np.unique(np.sort(edges, axis=1), axis=0)
    a, b = ue[:, 0], ue[:, 1]
    deg = np.zeros(movable.shape[0], dtype=np.float64)
    np.add.at(deg, a, 1.0)
    np.add.at(deg, b, 1.0)
    deg = np.maximum(deg, 1.0)
    movable = movable.copy()
    for _ in range(int(max_iters)):
        mn = np.zeros(movable.shape[0], dtype=np.float64)
        np.add.at(mn, a, movable[b].astype(np.float64))
        np.add.at(mn, b, movable[a].astype(np.float64))
        flip = (~movable) & ((mn / deg) >= float(close_frac))
        if not flip.any():
            break
        movable[flip] = True
    return movable


def _label_vertices(vertices, faces, states, reg, args):
    """Return per-vertex label array (0 = body, 1 = movable) + diagnostics.

    Pools per-state support at each state's own (group-shared) body pose over the
    states with a good registration fit; a vertex is movable when it is present in
    some states and its support drops across states (the part swung out of its rest
    footprint), then the confident seed is region-grown toward the joint.
    """
    n = vertices.shape[0]
    min_sup = np.full(n, np.inf, dtype=np.float64)
    max_sup = np.full(n, -np.inf, dtype=np.float64)
    obs_count = np.zeros(n, dtype=np.float64)
    used_states = []
    for st in states:
        sid = st["state_id"]
        record = reg.get(sid) or {}
        T = record.get("T_world_object")
        if T is None or record.get("status") == "skipped":
            continue
        fit = record.get("fit_iou")
        if fit is not None and float(fit) < float(args.min_fit_iou):
            continue
        cameras = rs._load_state_cameras(st["state_dir"])
        masks = rs._load_state_masks(st["mask_dir"])
        if not cameras or not masks:
            continue
        world_vertices = rs._apply_transform(vertices, np.asarray(T, dtype=np.float64))
        support, observed = _state_vertex_support(
            world_vertices, cameras, masks, int(args.mask_dilate), int(args.min_observed_cameras)
        )
        min_sup = np.where(observed, np.minimum(min_sup, support), min_sup)
        max_sup = np.where(observed, np.maximum(max_sup, support), max_sup)
        obs_count += observed.astype(np.float64)
        used_states.append(sid)

    labels = np.zeros(n, dtype=np.int64)
    enough = obs_count >= float(args.min_observed_states)
    valid_min = np.where(np.isfinite(min_sup), min_sup, 1.0)
    valid_max = np.where(np.isfinite(max_sup), max_sup, 0.0)
    drop = valid_max - valid_min
    present = enough & (valid_max >= float(args.present_support))
    # Confident seed: support DROPS strongly across states (the part swung out of its
    # rest footprint). The drop is robust to multiview averaging, which keeps a
    # joint-adjacent vertex's minimum well above 0 even though it clearly vacates in
    # the views that see the swing.
    seed = present & (drop >= float(args.support_drop))
    # Grow into the transition zone toward the joint (weaker drop); body vertices have
    # ~0 drop so growth stops at the joint.
    eligible = present & (drop >= float(args.grow_support_drop))
    movable = _grow_movable(seed, eligible, faces)
    grown = int(np.count_nonzero(movable))
    movable = _close_label_holes(movable, faces, float(args.close_holes_frac))
    labels[movable] = 1
    diag = {
        "used_states": used_states,
        "observed_vertices": int(np.count_nonzero(enough)),
        "movable_seed": int(np.count_nonzero(seed)),
        "movable_grown": grown,
        "movable_vertices_raw": int(np.count_nonzero(movable)),
    }
    return labels, valid_min, valid_max, diag


# ===========================================================================
# Faces -> parts (connected components)
# ===========================================================================


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _min_component_distance(verts_a, verts_b, cap=200):
    """Min Euclidean distance between two vertex sets (subsampled for speed)."""
    a = verts_a if verts_a.shape[0] <= cap else verts_a[np.linspace(0, verts_a.shape[0] - 1, cap).astype(np.int64)]
    b = verts_b if verts_b.shape[0] <= cap else verts_b[np.linspace(0, verts_b.shape[0] - 1, cap).astype(np.int64)]
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min()))


def _merge_close_components(components, vertices, merge_dist):
    """Merge movable components whose vertices come within ``merge_dist`` (one
    physical part fragmented by label holes)."""
    comps = [set(c) for c in components]
    changed = True
    while changed and len(comps) > 1:
        changed = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                if _min_component_distance(vertices[list(comps[i])], vertices[list(comps[j])]) <= merge_dist:
                    comps[i] |= comps[j]
                    comps.pop(j)
                    changed = True
                    break
            if changed:
                break
    return comps


def _movable_parts(vertices, faces, vertex_labels, min_part_vertices, merge_dist):
    """Group movable vertices into connected parts.

    Connectivity unions movable vertices that share a face with >= 2 movable
    vertices (bridges small label holes), then spatially close components are merged
    (one physical part split by noise). Returns (face_part, part_vertex_sets):
    ``face_part[f]`` is -1 for body faces else a part index; components smaller than
    ``min_part_vertices`` after merging drop back to the body.
    """
    movable = vertex_labels.astype(bool)
    uf = _UnionFind(vertices.shape[0])
    for face in faces:
        mv = [int(x) for x in face if movable[int(x)]]
        for x in mv[1:]:
            uf.union(mv[0], x)

    comp_vertices: Dict[int, set] = {}
    for v in np.nonzero(movable)[0]:
        comp_vertices.setdefault(uf.find(int(v)), set()).add(int(v))
    merged = _merge_close_components(list(comp_vertices.values()), vertices, merge_dist)
    kept = [c for c in merged if len(c) >= int(min_part_vertices)]
    kept.sort(key=lambda c: -len(c))

    vert_to_part = {}
    for i, comp in enumerate(kept):
        for v in comp:
            vert_to_part[v] = i
    face_part = np.full(faces.shape[0], -1, dtype=np.int64)
    for fi, face in enumerate(faces):
        hits = [vert_to_part[int(x)] for x in face if int(x) in vert_to_part]
        if len(hits) >= 2:
            face_part[fi] = max(set(hits), key=hits.count)
    return face_part, [c for c in kept]


# ===========================================================================
# Stage B: beyond-body residual part observations
# ===========================================================================


def _local_submesh(vertices, faces_subset):
    """Return a compact vertex/face array for a face subset."""
    if faces_subset is None or faces_subset.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    used = np.unique(faces_subset)
    remap = {int(v): i for i, v in enumerate(used)}
    local_faces = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces_subset], dtype=np.int64)
    return np.asarray(vertices[used], dtype=np.float64), local_faces


def _clean_binary_mask(mask, min_area):
    """Drop tiny residual components; keep shape otherwise untouched."""
    import cv2

    arr = np.asarray(mask, dtype=bool)
    min_area = int(min_area)
    if min_area <= 1 or not arr.any():
        return arr
    count, comps, stats, _cent = cv2.connectedComponentsWithStats(arr.astype(np.uint8), 8)
    out = np.zeros_like(arr, dtype=bool)
    for label in range(1, int(count)):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[comps == label] = True
    return out


def _write_stage_b_overlay(path, image_bgr, object_mask, body_mask, residual_mask):
    import cv2

    height, width = object_mask.shape[:2]
    canvas = image_bgr.copy() if image_bgr is not None else np.zeros((height, width, 3), dtype=np.uint8)
    overlay = canvas.copy()
    overlay[body_mask.astype(bool)] = (180, 180, 180)
    overlay[residual_mask.astype(bool)] = (255, 0, 255)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    def draw(mask_bool, color, thickness):
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, thickness)

    draw(object_mask, (0, 255, 255), 1)   # object mask = yellow
    draw(body_mask, (180, 180, 180), 1)   # rendered body = grey
    draw(residual_mask, (255, 0, 255), 2) # beyond-body residual = magenta
    cv2.imwrite(path, canvas)


def _stage_b_fit_args(args):
    """Small argparse-like object for register_states._register_one_state."""
    return argparse.Namespace(
        min_cameras=int(args.stage_b_min_cameras),
        score_points=int(args.stage_b_score_points),
        normal_dirs=int(args.stage_b_normal_dirs),
        inplane_steps=int(args.stage_b_inplane_steps),
        coarse_cameras=int(args.stage_b_coarse_cameras),
        coarse_downscale=int(args.stage_b_coarse_downscale),
        refine_cameras=int(args.stage_b_refine_cameras),
        refine_downscale=int(args.stage_b_refine_downscale),
        refine_seeds=int(args.stage_b_refine_seeds),
        refine_rounds=int(args.stage_b_refine_rounds),
        precision_weight=float(args.stage_b_precision_weight),
        recall_weight=float(args.stage_b_recall_weight),
        trim_fraction=float(args.stage_b_trim_fraction),
    )


def _rotation_angle(rotation):
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    value = (float(np.trace(r)) - 1.0) * 0.5
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def _pose_spread(relative_poses):
    """Max pairwise translation/rotation span for body-relative part poses."""
    poses = [np.asarray(p, dtype=np.float64) for p in relative_poses if p is not None]
    if len(poses) < 2:
        return 0.0, 0.0
    max_translation = 0.0
    max_rotation = 0.0
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            max_translation = max(max_translation, float(np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3])))
            max_rotation = max(max_rotation, _rotation_angle(poses[i][:3, :3].T @ poses[j][:3, :3]))
    return max_translation, max_rotation


def _build_stage_b_residuals(states, reg, vertices, render_faces, args, output_dir, overlay_dirname, source_name):
    """Build ``object mask - rendered mesh`` residuals for one Stage-B source.

    ``full_mesh`` is conservative: it explains every known rest-mesh surface and
    leaves only geometry exposed outside that silhouette. ``stage_a_body`` is useful
    after a reliable split, but using it first is circular: a static face mistakenly
    removed by Stage A becomes residual evidence for itself.
    """
    residual_root = os.path.join(output_dir, "segmentation", "stage_b", overlay_dirname)
    os.makedirs(residual_root, exist_ok=True)
    residuals = {}
    reports = {}
    for st in states:
        sid = st["state_id"]
        record = reg.get(sid) or {}
        T = record.get("T_world_object")
        if T is None or record.get("status") == "skipped":
            continue
        cameras = rs._load_state_cameras(st["state_dir"])
        masks = rs._load_state_masks(st["mask_dir"])
        if not cameras or not masks:
            continue
        T = np.asarray(T, dtype=np.float64)
        state_residuals = {}
        state_body_masks = {}
        image_fractions = []
        object_fractions = []
        explained_object_fractions = []
        body_precisions = []
        usable_cameras = 0
        for serial in sorted(masks):
            if serial not in cameras:
                continue
            obj = np.asarray(masks[serial], dtype=bool)
            if not obj.any():
                continue
            usable_cameras += 1
            rendered = rs._render_mesh_silhouette_mask(
                vertices, render_faces, T, cameras[serial], obj.shape, int(args.stage_b_body_dilate)
            )
            raw_residual = obj & ~rendered
            residual = _clean_binary_mask(raw_residual, int(args.stage_b_min_component_area))
            object_area = float(max(np.count_nonzero(obj), 1))
            rendered_area = float(max(np.count_nonzero(rendered), 1))
            overlap = float(np.count_nonzero(obj & rendered))
            image_fraction = float(np.count_nonzero(residual)) / float(max(obj.size, 1))
            object_fraction = float(np.count_nonzero(raw_residual)) / object_area
            image_fractions.append(image_fraction)
            object_fractions.append(object_fraction)
            explained_object_fractions.append(overlap / object_area)
            body_precisions.append(overlap / rendered_area)
            if residual.any() and image_fraction >= float(args.stage_b_min_residual_fraction):
                state_residuals[serial] = residual
                state_body_masks[serial] = rendered
        reports[sid] = {
            "residual_source": source_name,
            "placement_group": str(record.get("placement_group") or "default"),
            "registration_fit_iou": record.get("fit_iou"),
            "object_cameras": int(usable_cameras),
            "residual_cameras": int(len(state_residuals)),
            # These legacy image-area fractions remain for comparison. Quality
            # gates must use object-area fractions, otherwise object scale leaks
            # into the threshold.
            "mean_residual_fraction": float(np.mean(image_fractions)) if image_fractions else 0.0,
            "max_residual_fraction": float(np.max(image_fractions)) if image_fractions else 0.0,
            "mean_residual_object_fraction": float(np.mean(object_fractions)) if object_fractions else 0.0,
            "max_residual_object_fraction": float(np.max(object_fractions)) if object_fractions else 0.0,
            "mean_explained_object_fraction": float(np.mean(explained_object_fractions)) if explained_object_fractions else 0.0,
            "min_explained_object_fraction": float(np.min(explained_object_fractions)) if explained_object_fractions else 0.0,
            "mean_render_precision": float(np.mean(body_precisions)) if body_precisions else 0.0,
        }
        if state_residuals:
            residuals[sid] = state_residuals
            for serial in list(sorted(state_residuals))[: max(0, int(args.stage_b_overlay_cameras))]:
                image = rs._load_image_bgr(st["state_dir"], serial, args.image_dirname)
                _write_stage_b_overlay(
                    os.path.join(residual_root, f"{sid}_{serial}.jpg"),
                    image,
                    masks[serial],
                    state_body_masks[serial],
                    state_residuals[serial],
                )
    return residuals, reports


def _write_stage_b_image_motion_overlay(path, image_bgr, object_mask, motion_mask, rejected_motion=None):
    """Write one readable image-difference diagnostic.

    Cyan is RGB motion inside the dilated *group* object support and is usable for
    fitting. Red is raw RGB change rejected as background (typically table shadows
    or exposure artifacts). The yellow contour remains the current state's SAM mask;
    the accepted support is the union across this placement group so a handle may be
    present in a different joint state without being rejected here.
    """
    import cv2

    height, width = motion_mask.shape[:2]
    canvas = image_bgr.copy() if image_bgr is not None else np.zeros((height, width, 3), dtype=np.uint8)
    overlay = canvas.copy()
    if rejected_motion is not None:
        overlay[np.asarray(rejected_motion, dtype=bool)] = (0, 0, 255)  # red, BGR
    overlay[np.asarray(motion_mask, dtype=bool)] = (255, 255, 0)  # cyan, BGR
    canvas = cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0)
    if object_mask is not None:
        contours, _ = cv2.findContours(np.asarray(object_mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 255), 1)  # yellow, BGR
    contours, _ = cv2.findContours(np.asarray(motion_mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)
    cv2.imwrite(path, canvas)


def _gate_stage_b_image_motion(raw_motion, group_masks, object_band_px):
    """Keep RGB motion near the union object support for one group/camera.

    A within-placement image difference also sees cast shadows and the table.  The
    SAM masks are not trusted as an exact per-state part boundary, so the gate uses
    their *group union* plus a pixel band rather than the current state's mask.
    This preserves a handle that moves between the three states while rejecting
    change that is clearly detached from the object.
    """
    import cv2

    raw = np.asarray(raw_motion, dtype=bool)
    support = np.zeros(raw.shape, dtype=bool)
    for mask in group_masks:
        if mask is not None and np.asarray(mask).shape == raw.shape:
            support |= np.asarray(mask, dtype=bool)
    if not np.any(support):
        return np.zeros_like(raw), raw.copy(), support
    radius = max(0, int(object_band_px))
    if radius > 0:
        support = cv2.dilate(support.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=radius) > 0
    accepted = raw & support
    return accepted, raw & ~support, support


def _body_edge_ring(silhouette, band_px):
    """Band of pixels within ``band_px`` of the silhouette boundary (both sides)."""
    import cv2

    band = max(0, int(band_px))
    if band <= 0 or silhouette is None:
        return None
    sil = np.asarray(silhouette, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(sil, kernel, iterations=band) > 0
    eroded = cv2.erode(sil, kernel, iterations=band) > 0
    return dilated & ~eroded


def _build_stage_b_image_motion(states, reg, vertices, faces, args, output_dir):
    """Create per-state positive RGB-change maps inside each placement group.

    The cameras are static within a placement group, so the median image is a
    practical static-scene reference. Raw motion is restricted to the dilated union
    of object masks from the group, which removes table/shadow changes without
    requiring every individual SAM mask to contain the handle. A per-image
    median-brightness correction prevents a small exposure drift from turning the
    whole frame into motion evidence.

    RGB change within ``--stage-b-image-motion-body-edge-band`` pixels of the
    REGISTERED body silhouette boundary is suppressed: a dark body on light cloth
    flickers at its high-contrast edges (micro-shake, shadow, demosaic noise) even
    though it is fixed within the group, and those false thin bands both feed the
    part-pose fit and hand rim-shaped frame patches their "explained" pixels. A
    real moving part sweeps an AREA, so losing a thin rim of it is cheap.
    """
    import cv2

    root = os.path.join(output_dir, "segmentation", "stage_b", "image_motion_overlays")
    os.makedirs(root, exist_ok=True)
    by_group: Dict[str, list] = {}
    for st in states:
        sid = st["state_id"]
        record = reg.get(sid) or {}
        if record.get("T_world_object") is None or record.get("status") == "skipped":
            continue
        fit = record.get("fit_iou")
        if fit is not None and float(fit) < float(args.min_fit_iou):
            continue
        by_group.setdefault(str(record.get("placement_group") or "default"), []).append(st)

    evidence = {}
    reports = {}
    threshold = float(args.stage_b_image_motion_threshold)
    min_area = int(args.stage_b_image_motion_min_component_area)
    object_band = int(args.stage_b_image_motion_object_band)
    edge_band = int(args.stage_b_image_motion_body_edge_band)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for group, group_states in sorted(by_group.items()):
        if len(group_states) < 2:
            for st in group_states:
                reports.setdefault(st["state_id"], {
                    "placement_group": group,
                    "status": "insufficient_group_states",
                    "motion_cameras": 0,
                    "mean_motion_fraction": 0.0,
                })
            continue
        state_masks = {st["state_id"]: rs._load_state_masks(st["mask_dir"]) for st in group_states}
        serial_rows: Dict[str, list] = {}
        cams_by_state: Dict[str, dict] = {}
        for st in group_states:
            cameras = rs._load_state_cameras(st["state_dir"])
            cams_by_state[st["state_id"]] = cameras
            for serial in sorted(cameras):
                image = rs._load_image_bgr(st["state_dir"], serial, args.image_dirname)
                if image is not None:
                    serial_rows.setdefault(serial, []).append((st, image, state_masks[st["state_id"]].get(serial)))

        for serial, rows in serial_rows.items():
            if len(rows) < 2:
                continue
            grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) for _st, image, _mask in rows]
            reference = np.median(np.stack(grays, axis=0), axis=0)
            group_masks = [mask for _st, _image, mask in rows]
            for (st, image, object_mask), gray in zip(rows, grays):
                # Most background pixels are static.  Correct only the global offset
                # before differencing, leaving local part motion untouched.
                offset = float(np.median(reference - gray))
                diff = np.abs(np.clip(gray + offset, 0.0, 255.0) - reference)
                raw_motion = diff >= threshold
                if min_area > 1:
                    raw_motion = _clean_binary_mask(raw_motion, min_area)
                # Fill thin edges of a moved support without growing the region.
                raw_motion = cv2.morphologyEx(raw_motion.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
                motion, rejected_motion, group_support = _gate_stage_b_image_motion(
                    raw_motion, group_masks, object_band
                )
                sid = st["state_id"]
                edge_suppressed = 0
                if edge_band > 0 and np.any(motion):
                    record = reg.get(sid) or {}
                    body_T = record.get("T_world_object")
                    cam = (cams_by_state.get(sid) or {}).get(serial)
                    if body_T is not None and cam is not None:
                        silhouette = rs._render_mesh_silhouette_mask(
                            vertices, faces, np.asarray(body_T, dtype=np.float64), cam, motion.shape, 0
                        )
                        ring = _body_edge_ring(silhouette, edge_band)
                        if ring is not None:
                            edge_removed = motion & ring
                            edge_suppressed = int(np.count_nonzero(edge_removed))
                            if edge_suppressed:
                                motion = motion & ~ring
                                rejected_motion = rejected_motion | edge_removed
                if np.any(motion):
                    evidence.setdefault(sid, {})[serial] = motion
                row = reports.setdefault(sid, {
                    "placement_group": group,
                    "status": "ok",
                    "motion_cameras": 0,
                    "motion_fractions": [],
                    "raw_motion_fractions": [],
                    "rejected_motion_fractions": [],
                    "group_support_fractions": [],
                    "edge_suppressed_fractions": [],
                })
                row["motion_cameras"] += 1
                row["motion_fractions"].append(float(np.count_nonzero(motion)) / float(max(motion.size, 1)))
                row["raw_motion_fractions"].append(float(np.count_nonzero(raw_motion)) / float(max(raw_motion.size, 1)))
                row["rejected_motion_fractions"].append(float(np.count_nonzero(rejected_motion)) / float(max(rejected_motion.size, 1)))
                row["group_support_fractions"].append(float(np.count_nonzero(group_support)) / float(max(group_support.size, 1)))
                row["edge_suppressed_fractions"].append(float(edge_suppressed) / float(max(motion.size, 1)))
                _write_stage_b_image_motion_overlay(
                    os.path.join(root, f"{sid}_{serial}.jpg"), image, object_mask, motion, rejected_motion
                )

    for sid, row in reports.items():
        for source, target in (
            ("motion_fractions", "motion"),
            ("raw_motion_fractions", "raw_motion"),
            ("rejected_motion_fractions", "rejected_motion"),
            ("group_support_fractions", "group_support"),
            ("edge_suppressed_fractions", "edge_suppressed"),
        ):
            fractions = row.pop(source, [])
            row[f"mean_{target}_fraction"] = float(np.mean(fractions)) if fractions else 0.0
            row[f"max_{target}_fraction"] = float(np.max(fractions)) if fractions else 0.0
    return evidence, reports, root


def _combine_stage_b_evidence(residuals, image_motion, args):
    """Choose the positive evidence used for Stage-B fitting.

    ``hybrid`` retains image change outright and admits a residual pixel only near
    an image-change region.  This prevents a broad, imperfect rendered residual
    from dominating the fit while retaining silhouette-only motion when the RGB
    signal is locally weak.  If a state has no RGB comparison at all, it falls back
    to its residual instead of silently losing the state.
    """
    import cv2

    source = str(args.stage_b_evidence_source)
    out = {}
    rgb_only = {}
    radius = max(0, int(args.stage_b_hybrid_residual_dilate))
    kernel = np.ones((3, 3), dtype=np.uint8)
    for sid in sorted(set(residuals) | set(image_motion)):
        rows = {}
        for serial in sorted(set((residuals.get(sid) or {})) | set((image_motion.get(sid) or {}))):
            residual = (residuals.get(sid) or {}).get(serial)
            motion = (image_motion.get(sid) or {}).get(serial)
            if motion is not None and np.any(motion):
                rgb_only.setdefault(sid, {})[serial] = np.asarray(motion, dtype=bool)
            if source == "residual":
                chosen = residual
            elif source == "image_motion":
                chosen = motion
            elif motion is None or not np.any(motion):
                chosen = residual
            elif residual is None or not np.any(residual):
                chosen = motion
            else:
                near_motion = cv2.dilate(motion.astype(np.uint8), kernel, iterations=radius) > 0 if radius > 0 else motion
                chosen = np.asarray(motion, dtype=bool) | (np.asarray(residual, dtype=bool) & near_motion)
            if chosen is not None and np.any(chosen):
                rows[serial] = np.asarray(chosen, dtype=bool)
        if rows:
            out[sid] = rows
    return out, rgb_only


def _stage_b_residual_vertex_scores(states, reg, residuals, vertices, args):
    """Vertex support variation against beyond-body residual masks at the body pose."""
    n = vertices.shape[0]
    min_sup = np.full(n, np.inf, dtype=np.float64)
    max_sup = np.full(n, -np.inf, dtype=np.float64)
    obs_count = np.zeros(n, dtype=np.float64)
    used_states = []
    for st in states:
        sid = st["state_id"]
        if sid not in residuals:
            continue
        record = reg.get(sid) or {}
        T = record.get("T_world_object")
        if T is None or record.get("status") == "skipped":
            continue
        cameras = rs._load_state_cameras(st["state_dir"])
        if not cameras:
            continue
        world_vertices = rs._apply_transform(vertices, np.asarray(T, dtype=np.float64))
        support, observed = _state_vertex_support(
            world_vertices,
            cameras,
            residuals[sid],
            int(args.stage_b_refine_mask_dilate),
            int(args.stage_b_refine_min_observed_cameras),
        )
        min_sup = np.where(observed, np.minimum(min_sup, support), min_sup)
        max_sup = np.where(observed, np.maximum(max_sup, support), max_sup)
        obs_count += observed.astype(np.float64)
        used_states.append(sid)
    valid_min = np.where(np.isfinite(min_sup), min_sup, 1.0)
    valid_max = np.where(np.isfinite(max_sup), max_sup, 0.0)
    return valid_min, valid_max, valid_max - valid_min, obs_count, used_states


def _refine_face_part_by_residual(states, reg, residuals, vertices, faces, face_part, args):
    """Trim Stage-A movable labels using residual support variation inside each candidate.

    Static false splits that were removed from the body produce residual support at
    the same body-frame location in every state: high residual support but low drop.
    A truly moving surface is present in the body-frame residual in some states and
    absent in others: high residual drop. This produces a stricter Stage-B part mesh
    for later pose/joint fitting.
    """
    if bool(args.skip_stage_b_refine):
        return face_part, None
    min_sup, max_sup, drop, obs_count, used_states = _stage_b_residual_vertex_scores(
        states, reg, residuals, vertices, args
    )
    original = np.zeros(vertices.shape[0], dtype=bool)
    part_faces = face_part >= 0
    if np.any(part_faces):
        original[np.unique(faces[part_faces])] = True
    enough = obs_count >= float(args.stage_b_refine_min_observed_states)
    present = enough & (max_sup >= float(args.stage_b_refine_present_support))
    seed = original & present & (drop >= float(args.stage_b_refine_seed_drop))
    eligible = original & present & (drop >= float(args.stage_b_refine_grow_drop))
    refined = _grow_movable(seed, eligible, faces, max_iters=int(args.stage_b_refine_grow_iters))
    refined = _close_label_holes(refined, faces, float(args.stage_b_refine_close_holes_frac))
    mesh_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    refined_face_part, refined_sets = _movable_parts(
        vertices,
        faces,
        refined.astype(np.int64),
        int(args.stage_b_refine_min_part_vertices),
        float(args.stage_b_refine_merge_dist_frac) * mesh_diag,
    )
    diagnostics = {
        "used_states": used_states,
        "original_movable_vertices": int(np.count_nonzero(original)),
        "residual_observed_vertices": int(np.count_nonzero(enough)),
        "residual_seed_vertices": int(np.count_nonzero(seed)),
        "residual_eligible_vertices": int(np.count_nonzero(eligible)),
        "refined_movable_vertices": int(np.count_nonzero(refined)),
        "refined_part_count": int(len(refined_sets)),
        "params": {
            "seed_drop": args.stage_b_refine_seed_drop,
            "grow_drop": args.stage_b_refine_grow_drop,
            "present_support": args.stage_b_refine_present_support,
        },
    }
    return refined_face_part, diagnostics


def _write_stage_b_refined_outputs(output_dir, vertices, faces, refined_face_part, diagnostics):
    refined_dir = os.path.join(output_dir, "segmentation", "stage_b", "refined")
    parts_dir = os.path.join(refined_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    body_faces = faces[refined_face_part < 0]
    _write_obj_submesh(os.path.join(parts_dir, "body.obj"), vertices, body_faces)
    part_infos = [{"part_id": "body", "faces": int(body_faces.shape[0]), "mesh_path": os.path.join(parts_dir, "body.obj")}]
    vertex_colors = np.array([_BODY_COLOR] * vertices.shape[0], dtype=np.int64)
    for part_id in sorted(int(x) for x in np.unique(refined_face_part) if int(x) >= 0):
        pf = faces[refined_face_part == part_id]
        path = os.path.join(parts_dir, f"part_{part_id:02d}.obj")
        _write_obj_submesh(path, vertices, pf)
        color = _PART_COLORS[part_id % len(_PART_COLORS)]
        vertex_colors[np.unique(pf)] = color
        part_infos.append({"part_id": int(part_id), "faces": int(pf.shape[0]), "mesh_path": path})
    labeled_path = os.path.join(refined_dir, "mesh_labeled_refined.ply")
    _write_colored_ply(labeled_path, vertices, faces, vertex_colors)
    payload = dict(diagnostics or {})
    payload["mesh_labeled_path"] = labeled_path
    payload["parts"] = part_infos
    with open(os.path.join(refined_dir, "refinement.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _write_stage_b_motion_refined_outputs(output_dir, vertices, faces, motion_face_part, diagnostics, preview_max_faces=12000,
                                          dirname="motion_refined", mesh_filename="mesh_labeled_motion_refined.ply",
                                          report_filename="motion_refinement.json"):
    refined_dir = os.path.join(output_dir, "segmentation", "stage_b", dirname)
    parts_dir = os.path.join(refined_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    body_faces = faces[motion_face_part < 0]
    body_path = os.path.join(parts_dir, "body.obj")
    _write_obj_submesh(body_path, vertices, body_faces)
    part_infos = [{"part_id": "body", "faces": int(body_faces.shape[0]), "mesh_path": body_path}]
    vertex_colors = np.array([_BODY_COLOR] * vertices.shape[0], dtype=np.int64)
    for part_id in sorted(int(x) for x in np.unique(motion_face_part) if int(x) >= 0):
        pf = faces[motion_face_part == part_id]
        path = os.path.join(parts_dir, f"part_{part_id:02d}.obj")
        _write_obj_submesh(path, vertices, pf)
        color = _PART_COLORS[part_id % len(_PART_COLORS)]
        vertex_colors[np.unique(pf)] = color
        part_infos.append({"part_id": int(part_id), "faces": int(pf.shape[0]), "mesh_path": path})
    labeled_path = os.path.join(refined_dir, mesh_filename)
    _write_colored_ply(labeled_path, vertices, faces, vertex_colors)
    payload = dict(diagnostics or {})
    payload["mesh_labeled_path"] = labeled_path
    payload["parts"] = part_infos
    with open(os.path.join(refined_dir, report_filename), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    payload["preview_mesh"] = _labeled_preview_geometry(vertices, faces, motion_face_part, int(preview_max_faces))
    return payload


def _write_stage_b_geometry_refined_outputs(output_dir, vertices, faces, geometry_face_part, diagnostics, preview_max_faces=12000):
    refined_dir = os.path.join(output_dir, "segmentation", "stage_b", "geometry_refined")
    parts_dir = os.path.join(refined_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    body_faces = faces[geometry_face_part < 0]
    body_path = os.path.join(parts_dir, "body.obj")
    _write_obj_submesh(body_path, vertices, body_faces)
    part_infos = [{"part_id": "body", "faces": int(body_faces.shape[0]), "mesh_path": body_path}]
    vertex_colors = np.array([_BODY_COLOR] * vertices.shape[0], dtype=np.int64)
    for part_id in sorted(int(x) for x in np.unique(geometry_face_part) if int(x) >= 0):
        pf = faces[geometry_face_part == part_id]
        path = os.path.join(parts_dir, f"part_{part_id:02d}.obj")
        _write_obj_submesh(path, vertices, pf)
        color = _PART_COLORS[part_id % len(_PART_COLORS)]
        vertex_colors[np.unique(pf)] = color
        part_infos.append({"part_id": int(part_id), "faces": int(pf.shape[0]), "mesh_path": path})
    labeled_path = os.path.join(refined_dir, "mesh_labeled_geometry_refined.ply")
    _write_colored_ply(labeled_path, vertices, faces, vertex_colors)
    payload = dict(diagnostics or {})
    payload["mesh_labeled_path"] = labeled_path
    payload["parts"] = part_infos
    with open(os.path.join(refined_dir, "geometry_refinement.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    payload["preview_mesh"] = _labeled_preview_geometry(vertices, faces, geometry_face_part, int(preview_max_faces))
    return payload


def _geometry_refine_face_part(vertices, faces, face_part, part_summaries, args, output_dir):
    """Keep high-fit candidate surfaces that protrude outside the static body slab.

    This adds a mesh prior that the frame body is a broad, flat slab while the
    handle is a thin connected protrusion. It is deliberately applied after
    motion/residual scoring: low-fit metal fragments are skipped first; then the
    high-fit candidate is trimmed by out-of-slab geometry so oak border faces that
    live on the same slab as the body are merged back into body.
    """
    if not bool(args.enable_stage_b_geometry_refine):
        return None, None
    body_faces = face_part < 0
    if not np.any(body_faces):
        return None, None
    body_vertices_idx = np.unique(faces[body_faces])
    if body_vertices_idx.size < 8:
        return None, None
    body_vertices = vertices[body_vertices_idx]
    centroid, axes, extents = rs._pca_axes(body_vertices)
    normal = axes[:, 2]
    signed = (vertices - centroid.reshape(1, 3)) @ normal
    body_signed = signed[body_vertices_idx]
    body_center = float(np.median(body_signed))
    body_abs = np.abs(body_signed - body_center)
    slab_percentile = float(args.stage_b_geometry_slab_percentile)
    slab_half = float(np.percentile(body_abs, np.clip(slab_percentile, 50.0, 99.9)))
    mesh_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    seed_threshold = slab_half + float(args.stage_b_geometry_seed_margin_frac) * mesh_diag
    grow_threshold = slab_half + float(args.stage_b_geometry_grow_margin_frac) * mesh_diag

    fit_by_part = {int(p.get("part_id")): p for p in part_summaries}
    candidate_vertices = np.zeros(vertices.shape[0], dtype=bool)
    skipped_parts = []
    used_parts = []
    for part_id in sorted(int(x) for x in np.unique(face_part) if int(x) >= 0):
        summary = fit_by_part.get(part_id, {})
        fit = summary.get("median_fit_iou")
        status = str(summary.get("status", ""))
        if fit is None or float(fit) < float(args.stage_b_geometry_min_fit_iou):
            skipped_parts.append({"part_id": part_id, "reason": "low_fit", "median_fit_iou": fit, "status": status})
            continue
        if bool(args.stage_b_geometry_require_moving_candidate) and status != "moving_candidate":
            skipped_parts.append({"part_id": part_id, "reason": "not_moving_candidate", "median_fit_iou": fit, "status": status})
            continue
        pf = face_part == part_id
        if not np.any(pf):
            continue
        verts = np.unique(faces[pf])
        candidate_vertices[verts] = True
        used_parts.append({"part_id": part_id, "median_fit_iou": fit, "status": status, "source_vertices": int(verts.size)})

    if not np.any(candidate_vertices):
        diagnostics = {
            "status": "no_high_fit_candidates",
            "used_parts": used_parts,
            "skipped_parts": skipped_parts,
            "body_slab": {
                "normal": normal.tolist(),
                "center": body_center,
                "half_thickness": slab_half,
                "seed_threshold": seed_threshold,
                "grow_threshold": grow_threshold,
            },
        }
        return None, diagnostics

    out_of_slab = np.abs(signed - body_center)
    seed = candidate_vertices & (out_of_slab >= seed_threshold)
    eligible = candidate_vertices & (out_of_slab >= grow_threshold)
    keep = _grow_movable(seed, eligible, faces, max_iters=int(args.stage_b_geometry_grow_iters))
    keep = _close_label_holes(keep, faces, float(args.stage_b_geometry_close_holes_frac))
    keep &= candidate_vertices
    geometry_face_part, geometry_sets = _movable_parts(
        vertices,
        faces,
        keep.astype(np.int64),
        int(args.stage_b_geometry_min_part_vertices),
        float(args.stage_b_geometry_merge_dist_frac) * mesh_diag,
    )
    diagnostics = {
        "status": "ok",
        "used_parts": used_parts,
        "skipped_parts": skipped_parts,
        "source_candidate_vertices": int(np.count_nonzero(candidate_vertices)),
        "seed_vertices": int(np.count_nonzero(seed)),
        "eligible_vertices": int(np.count_nonzero(eligible)),
        "kept_vertices": int(np.count_nonzero(keep)),
        "part_count": int(len(geometry_sets)),
        "body_slab": {
            "normal": normal.tolist(),
            "center": body_center,
            "half_thickness": slab_half,
            "seed_threshold": seed_threshold,
            "grow_threshold": grow_threshold,
            "pca_extents": extents.tolist(),
        },
        "params": {
            "min_fit_iou": args.stage_b_geometry_min_fit_iou,
            "slab_percentile": args.stage_b_geometry_slab_percentile,
            "seed_margin_frac": args.stage_b_geometry_seed_margin_frac,
            "grow_margin_frac": args.stage_b_geometry_grow_margin_frac,
        },
    }
    payload = _write_stage_b_geometry_refined_outputs(
        output_dir, vertices, faces, geometry_face_part, diagnostics, int(args.viewer_max_faces)
    )
    return geometry_face_part, payload


def _motion_refine_face_part_from_tracks(states, reg, residuals, vertices, faces, face_part, tracks, part_summaries, args, output_dir):
    """Keep only surfaces whose fitted moving pose explains residuals better than body pose.

    This is stricter than residual support/drop. A static wood or metal false split
    can appear in residuals if Stage A removed it from the body, but it should still
    be explained at the body pose. A true moving part surface should explain residual
    masks better under the fitted per-state part pose.
    """
    if bool(args.skip_stage_b_motion_refine):
        return None, None
    summary_by_part = {int(p.get("part_id")): p for p in part_summaries}
    by_state = {st["state_id"]: st for st in states}
    original_movable = np.zeros(vertices.shape[0], dtype=bool)
    if np.any(face_part >= 0):
        original_movable[np.unique(faces[face_part >= 0])] = True
    keep_vertices = np.zeros(vertices.shape[0], dtype=bool)
    per_part = []

    for part_id in sorted(int(x) for x in np.unique(face_part) if int(x) >= 0):
        summary = summary_by_part.get(part_id, {})
        median_fit = summary.get("median_fit_iou")
        if median_fit is None or float(median_fit) < float(args.stage_b_motion_refine_min_fit_iou):
            per_part.append(
                {
                    "source_part_id": part_id,
                    "status": "skipped_low_fit",
                    "median_fit_iou": median_fit,
                    "kept_vertices": 0,
                }
            )
            continue
        candidate_vertices = np.zeros(vertices.shape[0], dtype=bool)
        candidate_faces = face_part == part_id
        if np.any(candidate_faces):
            candidate_vertices[np.unique(faces[candidate_faces])] = True
        if not np.any(candidate_vertices):
            continue

        max_adv = np.full(vertices.shape[0], -np.inf, dtype=np.float64)
        max_moving = np.zeros(vertices.shape[0], dtype=np.float64)
        sum_adv = np.zeros(vertices.shape[0], dtype=np.float64)
        obs_count = np.zeros(vertices.shape[0], dtype=np.float64)
        used_states = []
        for obs in tracks.get(f"part_{part_id:02d}", []):
            if obs.get("status") != "fit":
                continue
            sid = obs.get("state_id")
            if sid not in residuals or sid not in by_state:
                continue
            body_T = (reg.get(sid) or {}).get("T_world_object")
            part_T = obs.get("T_world_part")
            if body_T is None or part_T is None:
                continue
            cameras = rs._load_state_cameras(by_state[sid]["state_dir"])
            if not cameras:
                continue
            body_world = rs._apply_transform(vertices, np.asarray(body_T, dtype=np.float64))
            part_world = rs._apply_transform(vertices, np.asarray(part_T, dtype=np.float64))
            body_sup, body_obs = _state_vertex_support(
                body_world,
                cameras,
                residuals[sid],
                int(args.stage_b_motion_refine_mask_dilate),
                int(args.stage_b_motion_refine_min_observed_cameras),
            )
            moving_sup, moving_obs = _state_vertex_support(
                part_world,
                cameras,
                residuals[sid],
                int(args.stage_b_motion_refine_mask_dilate),
                int(args.stage_b_motion_refine_min_observed_cameras),
            )
            valid = candidate_vertices & (body_obs | moving_obs)
            if not np.any(valid):
                continue
            adv = moving_sup - body_sup
            max_adv[valid] = np.maximum(max_adv[valid], adv[valid])
            max_moving[valid] = np.maximum(max_moving[valid], moving_sup[valid])
            sum_adv[valid] += adv[valid]
            obs_count[valid] += 1.0
            used_states.append(sid)

        mean_adv = np.zeros(vertices.shape[0], dtype=np.float64)
        valid_count = obs_count > 0
        mean_adv[valid_count] = sum_adv[valid_count] / obs_count[valid_count]
        local_keep = (
            candidate_vertices
            & (obs_count >= float(args.stage_b_motion_refine_min_states))
            & (max_moving >= float(args.stage_b_motion_refine_min_moving_support))
            & (max_adv >= float(args.stage_b_motion_refine_min_advantage))
            & (mean_adv >= float(args.stage_b_motion_refine_min_mean_advantage))
        )
        keep_vertices |= local_keep
        per_part.append(
            {
                "source_part_id": part_id,
                "status": "scored",
                "median_fit_iou": median_fit,
                "used_states": sorted(set(used_states)),
                "source_vertices": int(np.count_nonzero(candidate_vertices)),
                "kept_vertices": int(np.count_nonzero(local_keep)),
                "max_advantage": float(np.max(max_adv[candidate_vertices & np.isfinite(max_adv)]))
                if np.any(candidate_vertices & np.isfinite(max_adv)) else 0.0,
                "max_moving_support": float(np.max(max_moving[candidate_vertices])) if np.any(candidate_vertices) else 0.0,
            }
        )

    keep_vertices &= original_movable
    keep_vertices = _close_label_holes(keep_vertices, faces, float(args.stage_b_motion_refine_close_holes_frac))
    keep_vertices &= original_movable
    mesh_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    motion_face_part, motion_sets = _movable_parts(
        vertices,
        faces,
        keep_vertices.astype(np.int64),
        int(args.stage_b_motion_refine_min_part_vertices),
        float(args.stage_b_motion_refine_merge_dist_frac) * mesh_diag,
    )
    diagnostics = {
        "source_movable_vertices": int(np.count_nonzero(original_movable)),
        "kept_vertices": int(np.count_nonzero(keep_vertices)),
        "part_count": int(len(motion_sets)),
        "source_parts": per_part,
        "params": {
            "min_fit_iou": args.stage_b_motion_refine_min_fit_iou,
            "min_moving_support": args.stage_b_motion_refine_min_moving_support,
            "min_advantage": args.stage_b_motion_refine_min_advantage,
            "min_mean_advantage": args.stage_b_motion_refine_min_mean_advantage,
        },
    }
    payload = _write_stage_b_motion_refined_outputs(
        output_dir, vertices, faces, motion_face_part, diagnostics, int(args.viewer_max_faces)
    )
    return motion_face_part, payload


def _face_normals(vertices, faces):
    """Unit triangle normals; degenerate triangles receive a zero normal."""
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, np.maximum(length, 1.0e-12), out=np.zeros_like(normals), where=length > 1.0e-12)


def _candidate_face_patches(vertices, faces, candidate_face_indices, args):
    """Split one broad candidate into welded, locally smooth face patches.

    This is intentionally a *surface* partition, not a semantic part predictor.
    The later pose/evidence test decides which patches move.  Welding is only used
    to recover adjacency from unwelded OBJ triangles; it never modifies the mesh.
    """
    selected = np.asarray(candidate_face_indices, dtype=np.int64)
    if selected.size == 0:
        return []
    mesh_diag = float(np.linalg.norm(np.asarray(vertices).max(axis=0) - np.asarray(vertices).min(axis=0)))
    tol = float(args.stage_b_patch_weld_tol_frac) * (mesh_diag if mesh_diag > 0.0 else 1.0)
    wid, _count = _weld_ids(vertices, tol)
    normals = _face_normals(vertices, faces)
    local_of = {int(fi): i for i, fi in enumerate(selected)}
    uf = _UnionFind(selected.size)
    edge_owner = {}
    cos_limit = float(np.cos(np.deg2rad(float(args.stage_b_patch_normal_angle_deg))))
    for li, fi in enumerate(selected):
        wf = wid[np.asarray(faces[int(fi)], dtype=np.int64)]
        for a, b in ((int(wf[0]), int(wf[1])), (int(wf[1]), int(wf[2])), (int(wf[0]), int(wf[2]))):
            edge = (a, b) if a < b else (b, a)
            other = edge_owner.get(edge)
            if other is None:
                edge_owner[edge] = li
                continue
            dot = float(np.dot(normals[int(fi)], normals[int(selected[other])]))
            if dot >= cos_limit:
                uf.union(li, other)

    groups: Dict[int, list] = {}
    for li, fi in enumerate(selected):
        groups.setdefault(uf.find(li), []).append(int(fi))
    patches = [np.asarray(group, dtype=np.int64) for group in groups.values()]
    patches.sort(key=lambda arr: -arr.size)
    return patches


def _camera_center_from_world_to_camera(cam_from_world):
    """Return the optical center for the camera bundle's world-to-camera matrix."""
    matrix = np.asarray(cam_from_world, dtype=np.float64)
    if matrix.shape == (3, 4):
        return -matrix[:, :3].T @ matrix[:, 3]
    if matrix.shape == (4, 4):
        return np.linalg.inv(matrix)[:3, 3]
    raise ValueError(f"unsupported camera extrinsic shape: {matrix.shape}")


def _state_patch_centroid_support(world_vertices, faces, patch_faces, cameras, evidence_masks, min_facing_cos):
    """Legacy positive-hit score for a front/projected patch face centroid.

    The new raster-change score is the default because a sparse set of centroids can
    let a large static patch score well by landing on a few RGB-change pixels.  This
    remains available for ablations and comparison with older Stage-B runs.
    """
    face_idx = np.asarray(patch_faces, dtype=np.int64)
    if face_idx.size == 0:
        return 0.0, 0
    tri = np.asarray(world_vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)[face_idx]]
    centers = np.mean(tri, axis=1)
    normals = _face_normals(world_vertices, np.asarray(faces, dtype=np.int64)[face_idx])
    support_sum = 0.0
    observed_cameras = 0
    for serial, mask in (evidence_masks or {}).items():
        cam = cameras.get(serial)
        if cam is None or mask is None:
            continue
        height, width = mask.shape[:2]
        pixels = rs._project_points(cam["projection"], centers)
        depths = rs._depths_in_camera(cam["cam_from_world"], centers)
        camera_center = _camera_center_from_world_to_camera(cam["cam_from_world"])
        view = camera_center.reshape(1, 3) - centers
        view_norm = np.linalg.norm(view, axis=1)
        facing = np.sum(normals * view, axis=1) / np.maximum(view_norm, 1.0e-12)
        u = np.rint(pixels[:, 0])
        v = np.rint(pixels[:, 1])
        visible = (
            np.isfinite(u) & np.isfinite(v) & np.isfinite(depths) & (depths > 0.0)
            & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            & (facing >= float(min_facing_cos))
        )
        idx = np.nonzero(visible)[0]
        if idx.size == 0:
            continue
        observed_cameras += 1
        support_sum += float(np.mean(np.asarray(mask, dtype=bool)[v[idx].astype(np.int64), u[idx].astype(np.int64)]))
    if observed_cameras == 0:
        return 0.0, 0
    return support_sum / float(observed_cameras), observed_cameras


def _depth_layers_kernel_python(triangles, triangle_depths, face_ids, height, width):
    """Software rasterizer for the nearest three depth layers.

    The same implementation is JIT-compiled below when numba is available.  Keeping
    this small fallback avoids a hard runtime dependency for an offline pipeline.
    """
    first_depth = np.full((height, width), np.inf, dtype=np.float32)
    second_depth = np.full((height, width), np.inf, dtype=np.float32)
    third_depth = np.full((height, width), np.inf, dtype=np.float32)
    first_owner = np.full((height, width), -1, dtype=np.int32)
    second_owner = np.full((height, width), -1, dtype=np.int32)
    third_owner = np.full((height, width), -1, dtype=np.int32)
    for face_index in range(triangles.shape[0]):
        tri = triangles[face_index]
        z = triangle_depths[face_index]
        x0, y0 = float(tri[0, 0]), float(tri[0, 1])
        x1, y1 = float(tri[1, 0]), float(tri[1, 1])
        x2, y2 = float(tri[2, 0]), float(tri[2, 1])
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1.0e-12:
            continue
        min_x = max(0, int(math.floor(min(x0, x1, x2))))
        max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
        min_y = max(0, int(math.floor(min(y0, y1, y2))))
        max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
        if min_x > max_x or min_y > max_y:
            continue
        inv_z0, inv_z1, inv_z2 = 1.0 / float(z[0]), 1.0 / float(z[1]), 1.0 / float(z[2])
        for y in range(min_y, max_y + 1):
            py = float(y) + 0.5
            for x in range(min_x, max_x + 1):
                px = float(x) + 0.5
                w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
                w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 < -1.0e-7 or w1 < -1.0e-7 or w2 < -1.0e-7:
                    continue
                inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2
                if inv_z <= 1.0e-12:
                    continue
                depth = 1.0 / inv_z
                if depth < first_depth[y, x]:
                    third_depth[y, x] = second_depth[y, x]
                    third_owner[y, x] = second_owner[y, x]
                    second_depth[y, x] = first_depth[y, x]
                    second_owner[y, x] = first_owner[y, x]
                    first_depth[y, x] = depth
                    first_owner[y, x] = face_ids[face_index]
                elif depth < second_depth[y, x]:
                    third_depth[y, x] = second_depth[y, x]
                    third_owner[y, x] = second_owner[y, x]
                    second_depth[y, x] = depth
                    second_owner[y, x] = face_ids[face_index]
                elif depth < third_depth[y, x]:
                    third_depth[y, x] = depth
                    third_owner[y, x] = face_ids[face_index]
    return first_depth, first_owner, second_depth, second_owner, third_depth, third_owner


_depth_layers_kernel = (
    _njit(nogil=True)(_depth_layers_kernel_python)
    if _njit is not None
    else _depth_layers_kernel_python
)
_DEPTH_RASTER_BACKEND = "numba" if _njit is not None else "python"


def _rasterize_mesh_depth_layers(world_vertices, faces, face_indices, camera, image_shape):
    """Return the nearest three visible mesh layers as depth and original face ids."""
    height, width = int(image_shape[0]), int(image_shape[1])
    empty_depth = np.full((height, width), np.inf, dtype=np.float32)
    empty_owner = np.full((height, width), -1, dtype=np.int32)
    selected = np.asarray(face_indices, dtype=np.int64)
    if height <= 0 or width <= 0 or selected.size == 0:
        return (
            empty_depth,
            empty_owner,
            empty_depth.copy(),
            empty_owner.copy(),
            empty_depth.copy(),
            empty_owner.copy(),
        )
    faces_arr = np.asarray(faces, dtype=np.int64)
    world_vertices = np.asarray(world_vertices, dtype=np.float64)
    pixels = rs._project_points(camera["projection"], world_vertices)
    depths = rs._depths_in_camera(camera["cam_from_world"], world_vertices)
    finite = np.isfinite(pixels).all(axis=1) & np.isfinite(depths) & (depths > 1.0e-9)
    finite &= np.max(np.abs(pixels), axis=1) < 1.0e6
    selected_faces = faces_arr[selected]
    face_valid = finite[selected_faces].all(axis=1)
    if not np.any(face_valid):
        return (
            empty_depth,
            empty_owner,
            empty_depth.copy(),
            empty_owner.copy(),
            empty_depth.copy(),
            empty_owner.copy(),
        )
    valid_indices = selected[face_valid].astype(np.int32)
    triangles = pixels[selected_faces[face_valid]].astype(np.float64, copy=False)
    triangle_depths = depths[selected_faces[face_valid]].astype(np.float64, copy=False)
    return _depth_layers_kernel(triangles, triangle_depths, valid_indices, height, width)


def _composite_patch_owner_change(full_layers, rest_patch_layers, moved_patch_layers, patch_face_mask):
    """Compose fixed non-patch geometry with rest/moved patch depth layers.

    ``full_layers`` carries the nearest three full-mesh face ids.  When the nearest
    owner belongs to this patch, the first deeper non-patch layer becomes the fixed
    body behind it.  This handles a thin patch's front/back triangles before looking
    for the separate static surface behind them.
    """
    full_depths = full_layers[0::2]
    full_owners = full_layers[1::2]
    patch_lookup = np.asarray(patch_face_mask, dtype=bool)
    static_depth = np.full(full_depths[0].shape, np.inf, dtype=np.float32)
    static_owner = np.full(full_owners[0].shape, -1, dtype=np.int32)
    unresolved = np.ones(static_depth.shape, dtype=bool)
    for depth, owner in zip(full_depths, full_owners):
        is_patch = (owner >= 0) & patch_lookup[np.maximum(owner, 0)]
        take = unresolved & (owner >= 0) & ~is_patch
        static_depth[take] = depth[take]
        static_owner[take] = owner[take]
        unresolved &= ~take

    def compose(patch_layers):
        patch_depth = patch_layers[0]
        patch_visible = patch_depth < static_depth
        depth = np.where(patch_visible, patch_depth, static_depth)
        owner = np.where(patch_visible, -2, static_owner).astype(np.int32)
        return depth, owner

    _rest_depth, rest_owner = compose(rest_patch_layers)
    _moved_depth, moved_owner = compose(moved_patch_layers)
    predicted = rest_owner != moved_owner
    return predicted, rest_owner == -2, moved_owner == -2


def _rasterize_patch_mask(world_vertices, faces, patch_faces, camera, image_shape, min_facing_cos):
    """Rasterize visible/front-facing patch triangles for one camera.

    This deliberately scores a patch's complete projected footprint, rather than
    individual face centroids.  It is a conservative first compositional check: we
    do not yet z-buffer against the static body, so an occluded patch can only lose
    support, never create a false positive by overwriting the body render.
    """
    import cv2

    height, width = image_shape[:2]
    raster = np.zeros((height, width), dtype=np.uint8)
    face_idx = np.asarray(patch_faces, dtype=np.int64)
    if face_idx.size == 0:
        return raster.astype(bool)
    tri = np.asarray(world_vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)[face_idx]]
    centers = np.mean(tri, axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(normal_length, 1.0e-12), out=np.zeros_like(normals), where=normal_length > 1.0e-12)
    camera_center = _camera_center_from_world_to_camera(camera["cam_from_world"])
    view = camera_center.reshape(1, 3) - centers
    facing = np.sum(normals * view, axis=1) / np.maximum(np.linalg.norm(view, axis=1), 1.0e-12)

    flat_tri = tri.reshape(-1, 3)
    pixels = rs._project_points(camera["projection"], flat_tri).reshape(-1, 3, 2)
    depths = rs._depths_in_camera(camera["cam_from_world"], flat_tri).reshape(-1, 3)
    for index, poly in enumerate(pixels):
        if facing[index] < float(min_facing_cos) or not np.all(np.isfinite(poly)) or not np.all(depths[index] > 0.0):
            continue
        if np.max(poly[:, 0]) < 0.0 or np.min(poly[:, 0]) >= width or np.max(poly[:, 1]) < 0.0 or np.min(poly[:, 1]) >= height:
            continue
        polygon = np.rint(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(raster, [polygon], 1)
    return raster.astype(bool)


def _state_patch_change_precision(
    body_world,
    moving_world,
    faces,
    patch_faces,
    cameras,
    evidence_masks,
    min_facing_cos,
    evidence_dilate_px,
    min_change_pixels,
):
    """Score whether a patch's predicted rest-vs-moving image change is observed.

    Precision is intentionally asymmetric: a broad static frame patch must explain
    most of the footprint it predicts, while an occluded or weakly textured true
    handle is allowed to be absent in individual views/groups.  The caller enforces
    repeated group support before retaining a patch.
    """
    import cv2

    precision_sum = 0.0
    observed_cameras = 0
    predicted_pixels = 0
    overlap_pixels = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    for serial, evidence in (evidence_masks or {}).items():
        camera = cameras.get(serial)
        if camera is None or evidence is None:
            continue
        evidence = np.asarray(evidence, dtype=bool)
        rest = _rasterize_patch_mask(body_world, faces, patch_faces, camera, evidence.shape, min_facing_cos)
        moved = _rasterize_patch_mask(moving_world, faces, patch_faces, camera, evidence.shape, min_facing_cos)
        predicted = np.logical_xor(rest, moved)
        count = int(np.count_nonzero(predicted))
        if count < int(min_change_pixels):
            continue
        observed = evidence
        radius = max(0, int(evidence_dilate_px))
        if radius:
            observed = cv2.dilate(observed.astype(np.uint8), kernel, iterations=radius) > 0
        overlap = int(np.count_nonzero(predicted & observed))
        precision_sum += float(overlap) / float(count)
        observed_cameras += 1
        predicted_pixels += count
        overlap_pixels += overlap
    if observed_cameras == 0:
        return 0.0, 0, 0, 0
    return precision_sum / float(observed_cameras), observed_cameras, predicted_pixels, overlap_pixels


def _pooled_change_precision(predicted_pixels, overlap_pixels):
    """Return the pixel-weighted evidence precision across all scored cameras."""
    predicted = int(predicted_pixels)
    if predicted <= 0:
        return 0.0
    return float(overlap_pixels) / float(predicted)


def _build_patch_composite_views(body_world, faces, cameras, evidence_masks, args):
    """Cache full-scene depth layers for the strongest RGB-evidence cameras.

    The body pose is shared by every patch in a state.  Rendering it once here is
    the difference between a compositional score that is practical and one that
    rerenders the whole frame for every local patch.
    """
    rows = []
    for serial, evidence in (evidence_masks or {}).items():
        camera = cameras.get(serial)
        if camera is None or evidence is None:
            continue
        evidence = np.asarray(evidence, dtype=bool)
        if not np.any(evidence):
            continue
        rows.append((int(np.count_nonzero(evidence)), str(serial), evidence, camera))
    rows.sort(key=lambda row: (-row[0], row[1]))
    max_cameras = int(args.stage_b_patch_composite_max_cameras)
    if max_cameras > 0:
        rows = rows[:max_cameras]
    all_faces = np.arange(np.asarray(faces).shape[0], dtype=np.int64)
    downscale = max(1, int(args.stage_b_patch_composite_downscale))
    views = []
    for _area, serial, evidence, camera in rows:
        small_evidence, small_camera, shape = rs._scaled_mask_and_cam(evidence, camera, downscale)
        full_layers = _rasterize_mesh_depth_layers(body_world, faces, all_faces, small_camera, shape)
        views.append({
            "serial": serial,
            "camera": small_camera,
            "evidence": small_evidence,
            "full_layers": full_layers,
        })
    return views


def _state_patch_composite_change_precision(
    body_world,
    moving_world,
    faces,
    patch_faces,
    composite_views,
    evidence_dilate_px,
    min_change_pixels,
    composite_min_change_pixels,
    downscale,
):
    """Score visible full-scene ownership changes caused by moving one patch.

    Unlike a patch-only silhouette XOR, a rear patch hidden behind fixed geometry
    owns no pixels in either composite render and cannot borrow the handle's RGB
    motion.  Only front-most owner changes can become predicted evidence.
    """
    import cv2

    patch_faces = np.asarray(patch_faces, dtype=np.int64)
    patch_lookup = np.zeros(np.asarray(faces).shape[0], dtype=bool)
    patch_lookup[patch_faces] = True
    scale = max(1, int(downscale))
    scaled_min_pixels = max(
        int(composite_min_change_pixels),
        int(math.ceil(float(min_change_pixels) / float(scale * scale))),
    )
    scaled_dilate = max(0, int(math.ceil(float(max(0, evidence_dilate_px)) / float(scale))))
    kernel = np.ones((3, 3), dtype=np.uint8)
    precision_sum = 0.0
    observed_cameras = 0
    predicted_pixels = 0
    overlap_pixels = 0
    for view in composite_views:
        evidence = np.asarray(view["evidence"], dtype=bool)
        rest_layers = _rasterize_mesh_depth_layers(body_world, faces, patch_faces, view["camera"], evidence.shape)
        moved_layers = _rasterize_mesh_depth_layers(moving_world, faces, patch_faces, view["camera"], evidence.shape)
        predicted, _rest_visible, _moved_visible = _composite_patch_owner_change(
            view["full_layers"], rest_layers, moved_layers, patch_lookup
        )
        count = int(np.count_nonzero(predicted))
        if count < scaled_min_pixels:
            continue
        observed = evidence
        if scaled_dilate:
            observed = cv2.dilate(observed.astype(np.uint8), kernel, iterations=scaled_dilate) > 0
        overlap = int(np.count_nonzero(predicted & observed))
        precision_sum += float(overlap) / float(count)
        observed_cameras += 1
        predicted_pixels += count
        overlap_pixels += overlap
    if observed_cameras == 0:
        return 0.0, 0, 0, 0
    return precision_sum / float(observed_cameras), observed_cameras, predicted_pixels, overlap_pixels


def _sparse_top_depth(world_vertices, faces, patch_faces, camera, image_shape):
    """Nearest-depth footprint of one patch as (flat pixel indices, depths)."""
    layers = _rasterize_mesh_depth_layers(world_vertices, faces, patch_faces, camera, image_shape)
    top = layers[0].reshape(-1)
    idx = np.nonzero(np.isfinite(top))[0].astype(np.int64)
    return idx, top[idx].astype(np.float32)


def _joint_static_base(full_layers, candidate_lookup):
    """Depth of the nearest full-scene surface that is not any scored patch.

    Candidate patches are handled separately (rest depths of the non-moved ones
    are min-merged back in per subset), so the same base serves every subset.
    """
    depths = full_layers[0::2]
    owners = full_layers[1::2]
    lookup = np.asarray(candidate_lookup, dtype=bool)
    static = np.full(depths[0].shape, np.inf, dtype=np.float32)
    unresolved = np.ones(static.shape, dtype=bool)
    for depth, owner in zip(depths, owners):
        take = unresolved & (owner >= 0) & ~lookup[np.maximum(owner, 0)]
        static[take] = depth[take]
        unresolved &= ~take
    return static.reshape(-1)


def _scatter_min(dense, sparse):
    """Min-merge one sparse (indices, depths) footprint into a flat depth map."""
    idx, depth = sparse
    if idx.size:
        dense[idx] = np.minimum(dense[idx], depth)


def _joint_predicted_change(view, subset_ids):
    """Predicted visible-change pixels when ``subset_ids`` move together.

    Composes the never-candidate scene, the REST surfaces of the candidates not
    in the subset, and the subset at its rest/moved poses.  A pixel is predicted
    to change exactly when the subset's front-most visibility flips, which
    includes the disocclusion the subset causes behind itself: a revealed static
    surface is thereby already explained by the mover and cannot earn its own
    selection gain.
    """
    static = view["static_base"].copy()
    rest_top = np.full(static.shape, np.inf, dtype=np.float32)
    moved_top = np.full(static.shape, np.inf, dtype=np.float32)
    for pid, sparse in view["rest"].items():
        if pid in subset_ids:
            _scatter_min(rest_top, sparse)
        else:
            _scatter_min(static, sparse)
    for pid in subset_ids:
        sparse = view["moved"].get(pid)
        if sparse is not None:
            _scatter_min(moved_top, sparse)
    return (rest_top < static) != (moved_top < static)


def _joint_objective(views, subset_ids, floor_pixels, spurious_weight):
    """(explained, spurious, score) of one subset across every cached view."""
    explained = 0
    spurious = 0
    for view in views:
        predicted = _joint_predicted_change(view, subset_ids)
        count = int(np.count_nonzero(predicted))
        if count < int(floor_pixels):
            continue
        overlap = int(np.count_nonzero(predicted & view["evidence"]))
        explained += overlap
        spurious += count - overlap
    return explained, spurious, float(explained) - float(spurious_weight) * float(spurious)


def _joint_select_patches(views, patch_ids, floor_pixels, spurious_weight, min_gain_pixels, min_gain_frac, max_selected):
    """Greedy marginal-gain subset selection (the explaining-away competition).

    Each round adds the patch whose joint prediction newly explains the most RGB
    evidence net of the spurious change it would add.  Pixels the current subset
    already explains contribute no gain, so a static surface whose only change is
    caused by an already-selected mover collapses to its spurious cost and drops
    out without an explicit duplicate-claim penalty.
    """
    selected: List[int] = []
    trace = []
    current = (0, 0, 0.0)
    first_gain = None
    remaining = [int(pid) for pid in patch_ids]
    while remaining and (int(max_selected) <= 0 or len(selected) < int(max_selected)):
        best = None
        for pid in remaining:
            explained, spurious, score = _joint_objective(views, selected + [pid], floor_pixels, spurious_weight)
            gain = score - current[2]
            if best is None or gain > best[0]:
                best = (gain, pid, explained, spurious, score)
        gain, pid, explained, spurious, score = best
        floor_gain = float(min_gain_pixels)
        if first_gain is not None:
            floor_gain = max(floor_gain, float(min_gain_frac) * float(first_gain))
        if gain < floor_gain:
            trace.append({"patch_id": int(pid), "gain": float(gain), "accepted": False,
                          "reason": "gain_below_floor", "gain_floor": float(floor_gain)})
            break
        selected.append(pid)
        remaining.remove(pid)
        if first_gain is None:
            first_gain = gain
        current = (explained, spurious, score)
        trace.append({"patch_id": int(pid), "gain": float(gain), "accepted": True,
                      "explained_pixels": int(explained), "spurious_pixels": int(spurious), "score": float(score)})
    return selected, trace, {"explained_pixels": int(current[0]), "spurious_pixels": int(current[1]), "score": float(current[2])}


def _joint_patch_stats(views, patch_ids, floor_pixels, args):
    """Independent per-patch recurrence statistics under the joint scene model.

    These do not decide selection (that is the greedy pass); they feed the same
    certification gates as ``composite_depth`` so a selected patch still has to
    recur across states and placement groups with adequate precision.
    """
    stats = {
        int(pid): {
            "state_count": 0,
            "groups": set(),
            "max_change_precision": 0.0,
            "sum_change_precision": 0.0,
            "predicted_change_pixels": 0,
            "overlap_change_pixels": 0,
            "used_states": [],
        }
        for pid in patch_ids
    }
    by_state: Dict[str, list] = {}
    for view in views:
        by_state.setdefault(view["state_id"], []).append(view)
    min_cams = int(args.stage_b_patch_min_observed_cameras)
    for pid in patch_ids:
        subset = [int(pid)]
        for sid, state_views in by_state.items():
            precision_sum = 0.0
            change_cams = 0
            predicted_pixels = 0
            overlap_pixels = 0
            for view in state_views:
                predicted = _joint_predicted_change(view, subset)
                count = int(np.count_nonzero(predicted))
                if count < int(floor_pixels):
                    continue
                overlap = int(np.count_nonzero(predicted & view["evidence"]))
                precision_sum += float(overlap) / float(count)
                change_cams += 1
                predicted_pixels += count
                overlap_pixels += overlap
            if change_cams < min_cams:
                continue
            precision = precision_sum / float(change_cams)
            row = stats[int(pid)]
            row["state_count"] += 1
            row["sum_change_precision"] += precision
            row["max_change_precision"] = max(row["max_change_precision"], precision)
            row["predicted_change_pixels"] += predicted_pixels
            row["overlap_change_pixels"] += overlap_pixels
            row["used_states"].append(sid)
            if precision >= float(args.stage_b_patch_min_change_precision):
                row["groups"].add(state_views[0]["group"])
    return stats


def _joint_gates_pass(stats, args):
    """The composite_depth certification gates, applied to one selected patch."""
    state_count = int(stats["state_count"])
    if state_count <= 0:
        return False
    mean_change_precision = stats["sum_change_precision"] / float(state_count)
    pooled = _pooled_change_precision(stats["predicted_change_pixels"], stats["overlap_change_pixels"])
    return (
        state_count >= int(args.stage_b_patch_min_states)
        and len(stats["groups"]) >= int(args.stage_b_patch_min_groups)
        and stats["max_change_precision"] >= float(args.stage_b_patch_min_change_precision)
        and mean_change_precision >= float(args.stage_b_patch_min_mean_change_precision)
        and pooled >= float(args.stage_b_patch_min_pooled_change_precision)
    )


def _build_joint_views(states, reg, scoring_evidence_masks, vertices, faces, patch_faces_of, candidate_lookup, track, args):
    """Per-camera joint scene caches for every fitted state of one candidate.

    Each view keeps the nearest never-candidate depth plus sparse nearest-depth
    footprints of every scored patch at rest and moved, so the greedy pass can
    compose any subset without re-rendering the full scene.
    """
    import cv2

    by_state = {st["state_id"]: st for st in states}
    scale = max(1, int(args.stage_b_patch_composite_downscale))
    scaled_dilate = max(0, int(math.ceil(float(max(0, int(args.stage_b_patch_change_dilate))) / float(scale))))
    kernel = np.ones((3, 3), dtype=np.uint8)
    views = []
    for obs in track:
        if obs.get("status") != "fit":
            continue
        sid = obs.get("state_id")
        st = by_state.get(sid)
        record = reg.get(sid) or {}
        body_T = record.get("T_world_object")
        part_T = obs.get("T_world_part")
        masks = scoring_evidence_masks.get(sid)
        if st is None or body_T is None or part_T is None or not masks:
            continue
        cameras = rs._load_state_cameras(st["state_dir"])
        if not cameras:
            continue
        body_world = rs._apply_transform(vertices, np.asarray(body_T, dtype=np.float64))
        moving_world = rs._apply_transform(vertices, np.asarray(part_T, dtype=np.float64))
        composite_views = _build_patch_composite_views(body_world, faces, cameras, masks, args)
        if len(composite_views) < int(args.stage_b_patch_min_observed_cameras):
            continue
        group = str(record.get("placement_group") or obs.get("placement_group") or "default")
        for view in composite_views:
            evidence = np.asarray(view["evidence"], dtype=bool)
            shape = (int(evidence.shape[0]), int(evidence.shape[1]))
            static_base = _joint_static_base(view["full_layers"], candidate_lookup)
            if scaled_dilate:
                evidence = cv2.dilate(evidence.astype(np.uint8), kernel, iterations=scaled_dilate) > 0
            rest = {}
            moved = {}
            for pid, pf in patch_faces_of.items():
                rest[pid] = _sparse_top_depth(body_world, faces, pf, view["camera"], shape)
                moved[pid] = _sparse_top_depth(moving_world, faces, pf, view["camera"], shape)
            views.append({
                "state_id": sid,
                "serial": view["serial"],
                "group": group,
                "shape": shape,
                "static_base": static_base,
                "evidence": evidence.reshape(-1),
                "rest": rest,
                "moved": moved,
            })
    return views


def _export_candidate_patches(patches_dir, vertices, faces, part_id, patch_faces_of):
    """Write each scored patch as an OBJ plus one patch-colored PLY per candidate."""
    os.makedirs(patches_dir, exist_ok=True)
    vertex_colors = np.array([_BODY_COLOR] * vertices.shape[0], dtype=np.int64)
    for pid in sorted(patch_faces_of):
        pf = patch_faces_of[pid]
        _write_obj_submesh(os.path.join(patches_dir, f"src{part_id:02d}_patch{pid:02d}.obj"), vertices, faces[pf])
        color = _PART_COLORS[pid % len(_PART_COLORS)]
        vertex_colors[np.unique(faces[pf])] = color
    _write_colored_ply(os.path.join(patches_dir, f"mesh_patches_src{part_id:02d}.ply"), vertices, faces, vertex_colors)


def _write_joint_selection_overlays(views, kept_ids, by_state, args, overlay_dir, part_id):
    """Explained / spurious / unexplained maps for the selected subset.

    Green = predicted change confirmed by RGB evidence, red = predicted change
    where nothing changed, blue = evidence the subset does not explain.  The
    evidence shown is the dilated map the scoring actually saw.
    """
    import cv2

    max_states = int(args.stage_b_patch_joint_overlay_states)
    max_cams = int(args.stage_b_patch_joint_overlay_cameras)
    if max_states <= 0 or max_cams <= 0 or not views:
        return []
    area_by_state: Dict[str, int] = {}
    for view in views:
        area_by_state[view["state_id"]] = area_by_state.get(view["state_id"], 0) + int(np.count_nonzero(view["evidence"]))
    chosen_states = [sid for sid, _area in sorted(area_by_state.items(), key=lambda kv: -kv[1])[:max_states]]
    os.makedirs(overlay_dir, exist_ok=True)
    written = []
    kept = [int(x) for x in kept_ids]
    for sid in chosen_states:
        state_views = [v for v in views if v["state_id"] == sid][:max_cams]
        st = by_state.get(sid)
        for view in state_views:
            h, w = view["shape"]
            predicted = _joint_predicted_change(view, kept).reshape(h, w)
            evidence = view["evidence"].reshape(h, w)
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            image = rs._load_image_bgr(st["state_dir"], view["serial"], args.image_dirname) if st else None
            if image is not None:
                canvas = (cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA) * 0.45).astype(np.uint8)
            canvas[evidence & ~predicted] = (230, 120, 0)
            canvas[predicted & evidence] = (60, 200, 60)
            canvas[predicted & ~evidence] = (40, 40, 230)
            path = os.path.join(overlay_dir, f"src{part_id:02d}_{sid}_{view['serial']}.jpg")
            cv2.imwrite(path, canvas)
            written.append(path)
    return written


def _fit_part_track(states, reg, fit_evidence, part_vertices, part_faces, args):
    """Register one candidate submesh to each state's positive fit evidence.

    Shared by the initial broad-candidate proposal fit and the joint-subset refit
    so both produce identical observation records.
    """
    fit_args = _stage_b_fit_args(args)
    by_state = {st["state_id"]: st for st in states}
    part_points = part_vertices
    part_centroid = np.mean(part_vertices, axis=0)
    part_diag = float(np.linalg.norm(part_vertices.max(axis=0) - part_vertices.min(axis=0)))
    if part_diag < 1.0e-9:
        part_diag = 1.0
    observations = []
    for sid in sorted(fit_evidence):
        st = by_state.get(sid)
        record = reg.get(sid) or {}
        body_T = record.get("T_world_object")
        if st is None or body_T is None:
            continue
        cameras = rs._load_state_cameras(st["state_dir"])
        result = rs._register_one_state(
            cameras, fit_evidence[sid], part_vertices, part_points, part_centroid, part_diag, fit_args
        )
        if result is None:
            observations.append({"state_id": sid, "status": "skipped"})
            continue
        body_T_arr = np.asarray(body_T, dtype=np.float64)
        part_T = np.asarray(result["T"], dtype=np.float64)
        rel_T = np.linalg.inv(body_T_arr) @ part_T
        fit_iou = rs._fit_iou(
            part_vertices, part_faces, part_T, cameras, fit_evidence[sid], result["serials"], float(args.stage_b_trim_fraction)
        )
        observations.append(
            {
                "state_id": sid,
                "status": "fit",
                "placement_group": str(record.get("placement_group") or "default"),
                "fit_iou": fit_iou,
                "cost": float(result["cost"]),
                "mask_cameras": int(result["mask_cameras"]),
                "T_world_part": part_T.tolist(),
                "T_body_part": rel_T.tolist(),
            }
        )
    return observations


def _summarize_part_track(part_id, observations, args):
    """Motion/fit summary for one candidate's observation track."""
    successful = [obs for obs in observations if obs.get("status") == "fit"]
    all_rel = [np.asarray(obs["T_body_part"], dtype=np.float64) for obs in successful]
    t_span, r_span = _pose_spread(all_rel)
    relative_by_group: Dict[str, list] = {}
    for obs in successful:
        relative_by_group.setdefault(str(obs.get("placement_group") or "default"), []).append(
            np.asarray(obs["T_body_part"], dtype=np.float64)
        )
    group_spans = {}
    for group, poses in relative_by_group.items():
        gt, gr = _pose_spread(poses)
        group_spans[group] = {"translation_span": gt, "rotation_span_rad": gr, "rotation_span_deg": float(np.rad2deg(gr))}
    median_fit = float(np.median([obs["fit_iou"] for obs in successful if obs.get("fit_iou") is not None])) if successful else None
    has_fit_quality = median_fit is not None and median_fit >= float(args.stage_b_min_fit_iou)
    has_motion = (
        len(successful) >= int(args.stage_b_min_motion_states)
        and (
            r_span >= np.deg2rad(float(args.stage_b_motion_min_rotation_deg))
            or t_span >= float(args.stage_b_motion_min_translation)
        )
    )
    if has_fit_quality and has_motion:
        status = "moving_candidate"
    elif not has_fit_quality:
        status = "low_fit_candidate"
    else:
        status = "static_or_noise_candidate"
    return {
        "part_id": int(part_id),
        "status": status,
        "successful_states": int(len(successful)),
        "median_fit_iou": median_fit,
        "translation_span": float(t_span),
        "rotation_span_rad": float(r_span),
        "rotation_span_deg": float(np.rad2deg(r_span)),
        "group_spans": group_spans,
    }


def _joint_round_is_better(best, kept, score):
    """Adopt a later selection round only when it actually improves the objective.

    The joint score (explained minus weighted spurious pixels) is comparable
    across rounds because the evidence is fixed; subset fit_iou is NOT (a smaller
    submesh scores a lower IoU against the same evidence even at a perfect pose),
    so it must never decide the round.
    """
    if best is None:
        return True
    if not kept:
        return False
    if not best.get("kept"):
        return True
    return float(score) > float((best.get("totals") or {}).get("score", -np.inf))


def _joint_refine_candidate(
    states,
    reg,
    proposal_evidence_masks,
    scoring_evidence_masks,
    vertices,
    faces,
    part_id,
    patches,
    track,
    summary,
    args,
    output_dir,
):
    """Greedy explaining-away patch selection for one moving candidate.

    Independent per-patch testing cannot reject a static surface that the true
    mover occludes or reveals: the disocclusion is real image change at that
    surface's own footprint, and hinge-adjacent geometry sweeps into the mover's
    change region under the shared part transform.  Selecting patches jointly
    under the composite scene resolves both: the mover alone already predicts the
    revealed surfaces' change, so their marginal gain collapses to the spurious
    change they would add elsewhere.  The part pose (proposed by the broad,
    partially static candidate) is then refit on the selected subset and the
    selection repeated once under the better pose; a refit round that keeps
    nothing falls back to the previous round.
    """
    refined_dir = os.path.join(output_dir, "segmentation", "stage_b", "patch_refined")
    n_faces = int(np.asarray(faces).shape[0])
    scale = max(1, int(args.stage_b_patch_composite_downscale))
    floor_pixels = max(
        int(args.stage_b_patch_composite_min_change_pixels),
        int(math.ceil(float(args.stage_b_patch_min_change_pixels) / float(scale * scale))),
    )

    rows = []
    scored = []
    for patch_id, patch_faces in enumerate(patches):
        if patch_faces.size < int(args.stage_b_patch_min_faces):
            rows.append({"source_part_id": part_id, "patch_id": patch_id, "status": "skipped_too_small", "faces": int(patch_faces.size)})
            continue
        scored.append((int(patch_id), patch_faces))
    max_patches = int(args.stage_b_patch_joint_max_patches)
    if max_patches > 0 and len(scored) > max_patches:
        by_size = sorted(scored, key=lambda row: -row[1].size)
        for patch_id, patch_faces in by_size[max_patches:]:
            rows.append({"source_part_id": part_id, "patch_id": patch_id, "status": "skipped_joint_patch_cap", "faces": int(patch_faces.size)})
        keep_ids = {pid for pid, _pf in by_size[:max_patches]}
        scored = [(pid, pf) for pid, pf in scored if pid in keep_ids]

    info = {"part_id": int(part_id), "scored_patches": [pid for pid, _pf in scored], "rounds": []}
    empty = np.zeros(0, dtype=np.int64)
    if not scored:
        info["status"] = "no_scored_patches"
        return {"rows": rows, "kept_faces": empty, "info": info, "track_update": None, "summary_update": None}

    patch_faces_of = {pid: pf for pid, pf in scored}
    patch_ids = sorted(patch_faces_of)
    candidate_lookup = np.zeros(n_faces, dtype=bool)
    for pf in patch_faces_of.values():
        candidate_lookup[pf] = True

    patches_dir = os.path.join(refined_dir, "patches")
    info["patches_dir"] = patches_dir
    _export_candidate_patches(patches_dir, vertices, faces, part_id, patch_faces_of)

    refit_rounds = max(0, int(args.stage_b_patch_joint_refit_rounds))
    current_track = track
    pending_summary = None
    best = None
    for round_index in range(refit_rounds + 1):
        views = _build_joint_views(
            states, reg, scoring_evidence_masks, vertices, faces,
            patch_faces_of, candidate_lookup, current_track, args,
        )
        if not views:
            info["rounds"].append({"round": round_index, "status": "no_views"})
            break
        patch_stats = _joint_patch_stats(views, patch_ids, floor_pixels, args)
        selected, trace, totals = _joint_select_patches(
            views, patch_ids, floor_pixels,
            float(args.stage_b_patch_joint_spurious_weight),
            float(args.stage_b_patch_joint_min_gain_pixels),
            float(args.stage_b_patch_joint_min_gain_frac),
            int(args.stage_b_patch_joint_max_selected),
        )
        kept = [pid for pid in selected if _joint_gates_pass(patch_stats[pid], args)]
        round_summary = summary if round_index == 0 else pending_summary
        info["rounds"].append({
            "round": round_index,
            "selected": [int(x) for x in selected],
            "kept": [int(x) for x in kept],
            "explained_pixels": int(totals["explained_pixels"]),
            "spurious_pixels": int(totals["spurious_pixels"]),
            "score": float(totals["score"]),
            "median_fit_iou": (round_summary or {}).get("median_fit_iou"),
            "median_fit_iou_subset": (round_summary or {}).get("median_fit_iou_subset"),
            "trace": trace,
        })
        result = {
            "round": round_index,
            "views": views,
            "patch_stats": patch_stats,
            "selected": selected,
            "kept": kept,
            "trace": trace,
            "totals": totals,
            "track": current_track,
            "summary_update": None if round_index == 0 else pending_summary,
        }
        if _joint_round_is_better(best, kept, totals["score"]):
            best = result
        if round_index >= refit_rounds or not kept:
            break
        subset_faces = np.concatenate([patch_faces_of[pid] for pid in kept])
        sub_vertices, sub_faces = _local_submesh(vertices, faces[subset_faces])
        if sub_vertices.shape[0] < 3 or sub_faces.shape[0] == 0:
            info["refit_status"] = "skipped_small_subset"
            break
        refit_obs = _fit_part_track(states, reg, proposal_evidence_masks, sub_vertices, sub_faces, args)
        refit_summary = _summarize_part_track(part_id, refit_obs, args)
        # The subset-vs-evidence IoU is size-biased (a smaller submesh can only
        # lose IoU against the same evidence), so it is reported separately and
        # must neither downgrade the candidate's moving status nor overwrite the
        # comparable broad-candidate fit value (a downgraded status would silently
        # remove the candidate from Stage C).
        refit_summary["median_fit_iou_subset"] = refit_summary["median_fit_iou"]
        refit_summary["median_fit_iou"] = (summary or {}).get("median_fit_iou")
        refit_summary["status"] = (summary or {}).get("status") or refit_summary["status"]
        refit_summary["refit"] = True
        pending_summary = refit_summary
        current_track = refit_obs

    if best is None:
        info["status"] = "no_views"
        return {"rows": rows, "kept_faces": empty, "info": info, "track_update": None, "summary_update": None}

    info["used_round"] = int(best["round"])
    info["refit_rejected"] = bool(len(info["rounds"]) > 1 and best["round"] == 0)
    selected_set = {int(x) for x in best["selected"]}
    kept_set = {int(x) for x in best["kept"]}
    info["kept_patches"] = sorted(kept_set)
    info["stop_reason"] = next((t.get("reason") for t in best["trace"] if not t.get("accepted")), "max_selected_or_exhausted")
    gain_of = {int(t["patch_id"]): float(t["gain"]) for t in best["trace"] if t.get("accepted")}
    kept_faces_list = []
    for pid in patch_ids:
        stats = best["patch_stats"][pid]
        state_count = int(stats["state_count"])
        mean_change_precision = stats["sum_change_precision"] / float(state_count) if state_count else 0.0
        pooled = _pooled_change_precision(stats["predicted_change_pixels"], stats["overlap_change_pixels"])
        if pid in kept_set:
            status = "kept"
            kept_faces_list.append(patch_faces_of[pid])
        elif pid in selected_set:
            status = "selected_gate_failed"
        else:
            status = "rejected_not_selected"
        rows.append({
            "source_part_id": part_id,
            "patch_id": int(pid),
            "status": status,
            "faces": int(patch_faces_of[pid].size),
            "observed_states": state_count,
            "motion_groups": int(len(stats["groups"])),
            "used_states": sorted(set(stats["used_states"])),
            "max_advantage": 0.0,
            "mean_advantage": 0.0,
            "max_moving_support": 0.0,
            "max_change_precision": float(stats["max_change_precision"]),
            "mean_change_precision": float(mean_change_precision),
            "pooled_change_precision": float(pooled),
            "predicted_change_pixels": int(stats["predicted_change_pixels"]),
            "overlap_change_pixels": int(stats["overlap_change_pixels"]),
            "selection_gain": gain_of.get(int(pid)),
            "obj_path": os.path.join(patches_dir, f"src{part_id:02d}_patch{pid:02d}.obj"),
        })
    kept_faces = np.concatenate(kept_faces_list) if kept_faces_list else empty

    if kept_set:
        overlay_dir = os.path.join(refined_dir, "joint_overlays")
        by_state = {st["state_id"]: st for st in states}
        _write_joint_selection_overlays(best["views"], sorted(kept_set), by_state, args, overlay_dir, part_id)
        info["overlay_dir"] = overlay_dir

    track_update = best["track"] if best["round"] > 0 else None
    summary_update = best["summary_update"] if best["round"] > 0 else None
    return {"rows": rows, "kept_faces": kept_faces, "info": info, "track_update": track_update, "summary_update": summary_update}


def _cross_patch_key(source_part_id, patch_id):
    return f"src{int(source_part_id):02d}_patch{int(patch_id):02d}"


def _cross_hypothesis_key(patch_key, track_part_id):
    return f"{patch_key}__track{int(track_part_id):02d}"


def _track_pose_lookup(track):
    """Successful state poses keyed by state id for one candidate track."""
    out = {}
    for obs in track or ():
        if obs.get("status") != "fit" or obs.get("T_world_part") is None:
            continue
        sid = obs.get("state_id")
        if sid is not None:
            out[str(sid)] = np.asarray(obs["T_world_part"], dtype=np.float64)
    return out


def _cross_track_groups(track, reg):
    groups = set()
    for obs in track or ():
        if obs.get("status") != "fit":
            continue
        sid = obs.get("state_id")
        record = reg.get(sid) or {}
        groups.add(str(record.get("placement_group") or obs.get("placement_group") or "default"))
    return groups


def _cross_source_ids(part_summaries, args):
    """Return broad Stage-B candidates that may contribute source patches."""
    summary_by_part = {int(row.get("part_id")): row for row in part_summaries}
    source_ids = [
        int(row["part_id"])
        for row in part_summaries
        if row.get("status") == "moving_candidate"
        and row.get("median_fit_iou") is not None
        and float(row["median_fit_iou"]) >= float(args.stage_b_patch_min_fit_iou)
    ]
    source_ids.sort(key=lambda pid: -(summary_by_part[pid].get("median_fit_iou") or 0.0))
    return source_ids


def _cross_track_ids(source_ids, track_summaries, tracks, reg, args):
    """Admit only reliable pose tracks for cross-candidate composition."""
    ranked = sorted(
        (int(pid) for pid in source_ids),
        key=lambda pid: -(track_summaries.get(pid, {}).get("median_fit_iou") or 0.0),
    )
    min_fit = max(0.0, float(args.stage_b_cross_min_track_fit_iou))

    def track_for(pid):
        # Initial Stage-B tracks are keyed as ``part_00``; probe replacements are
        # held by integer part id until they are committed to the Stage-B payload.
        return tracks.get(pid, tracks.get(f"part_{pid:02d}", []))

    track_ids = [
        pid
        for pid in ranked
        if float(track_summaries.get(pid, {}).get("median_fit_iou") or 0.0) >= min_fit
        and len(_cross_track_groups(track_for(pid), reg)) >= int(args.stage_b_patch_min_groups)
    ]
    if track_ids:
        best_fit = float(track_summaries[track_ids[0]].get("median_fit_iou") or 0.0)
        fit_ratio = max(0.0, float(args.stage_b_cross_track_min_fit_ratio))
        if fit_ratio > 0.0 and best_fit > 0.0:
            track_ids = [
                pid
                for pid in track_ids
                if float(track_summaries[pid].get("median_fit_iou") or 0.0) >= fit_ratio * best_fit
            ]
    max_tracks = int(args.stage_b_cross_max_tracks)
    if max_tracks > 0:
        track_ids = track_ids[:max_tracks]
    return track_ids


def _cross_source_and_track_ids(part_summaries, tracks, reg, args):
    """Keep broad patch sources separate from reliable pose-track anchors.

    A low-fit Stage-B candidate can contain a true handle fragment, but its own
    pose is too ambiguous to become a second articulated output.  Such a source
    remains eligible to borrow a reliable track from the same global pool.
    """
    summary_by_part = {int(row.get("part_id")): row for row in part_summaries}
    source_ids = _cross_source_ids(part_summaries, args)
    return source_ids, _cross_track_ids(source_ids, summary_by_part, tracks, reg, args)


def _cross_patch_area(vertices, faces, patch_faces):
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)[patch_faces]]
    return float(np.sum(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)) * 0.5)


def _cross_track_fit_diagnostics(track):
    """Summarize the pose observations that actually drive one cross round."""
    fitted = [obs for obs in track or () if obs.get("status") == "fit" and obs.get("fit_iou") is not None]
    fits = [float(obs["fit_iou"]) for obs in fitted]
    return {
        "successful_states": int(len(fitted)),
        "median_fit_iou": float(np.median(fits)) if fits else None,
    }


def _cross_anchor_drift(anchor_track, candidate_track, mesh_diag, args):
    """Measure whether a refit stays on the independently fitted anchor pose.

    Cross selection may improve a shape subset, but it must not turn an independently
    fitted handle-patch trajectory into a different articulated trajectory.  Compare
    the body-relative pose at every state shared by the anchor and refit tracks.
    """
    anchor_by_state = {
        str(obs.get("state_id")): np.asarray(obs["T_body_part"], dtype=np.float64)
        for obs in anchor_track or ()
        if obs.get("status") == "fit" and obs.get("state_id") is not None and obs.get("T_body_part") is not None
    }
    candidate_by_state = {
        str(obs.get("state_id")): np.asarray(obs["T_body_part"], dtype=np.float64)
        for obs in candidate_track or ()
        if obs.get("status") == "fit" and obs.get("state_id") is not None and obs.get("T_body_part") is not None
    }
    shared = sorted(set(anchor_by_state) & set(candidate_by_state))
    translation = []
    rotation_deg = []
    for sid in shared:
        delta = np.linalg.inv(anchor_by_state[sid]) @ candidate_by_state[sid]
        translation.append(float(np.linalg.norm(delta[:3, 3])))
        rotation_deg.append(float(np.rad2deg(_rotation_angle(delta[:3, :3]))))
    scale = max(float(mesh_diag), 1.0e-9)
    max_translation = max(translation) if translation else None
    max_rotation = max(rotation_deg) if rotation_deg else None
    translation_limit = max(0.0, float(getattr(args, "stage_b_cross_refit_max_anchor_translation_frac", 0.03))) * scale
    rotation_limit = max(0.0, float(getattr(args, "stage_b_cross_refit_max_anchor_rotation_deg", 15.0)))
    passes = bool(
        shared
        and max_translation is not None
        and max_rotation is not None
        and max_translation <= translation_limit
        and max_rotation <= rotation_limit
    )
    if not shared:
        reason = "no_shared_anchor_states"
    elif max_rotation is not None and max_rotation > rotation_limit:
        reason = "anchor_rotation_drift"
    elif max_translation is not None and max_translation > translation_limit:
        reason = "anchor_translation_drift"
    else:
        reason = "locked"
    return {
        "shared_states": int(len(shared)),
        "max_translation": max_translation,
        "max_translation_fraction": (max_translation / scale) if max_translation is not None else None,
        "max_rotation_deg": max_rotation,
        "max_translation_limit": float(translation_limit),
        "max_translation_limit_fraction": float(translation_limit / scale),
        "max_rotation_limit_deg": float(rotation_limit),
        "passes": bool(passes),
        "reason": reason,
    }


def _cross_refit_faces(kept, hypotheses, patch_entries, track_part_id, anchor_patch_key, anchor_locked):
    """Faces used to refit one track, optionally retaining its probe anchor."""
    keys = [
        hypotheses[hypothesis]["patch_key"]
        for hypothesis in kept
        if int(hypotheses[hypothesis]["track_part_id"]) == int(track_part_id)
    ]
    if anchor_locked and anchor_patch_key is not None:
        keys.append(str(anchor_patch_key))
    keys = list(dict.fromkeys(keys))
    face_sets = [np.asarray(patch_entries[key]["faces"], dtype=np.int64) for key in keys if key in patch_entries]
    if not face_sets:
        return np.zeros(0, dtype=np.int64), keys
    return np.unique(np.concatenate(face_sets)), keys


def _cross_round_track_diagnostics(current_tracks, anchor_tracks, mesh_diag, args):
    """Expose the actual track quality and anchor drift for a scored round."""
    rows = []
    for track_part_id, track in sorted(current_tracks.items()):
        row = {"track_part_id": int(track_part_id), **_cross_track_fit_diagnostics(track)}
        anchor_track = anchor_tracks.get(int(track_part_id))
        if anchor_track is not None:
            row["anchor_drift"] = _cross_anchor_drift(anchor_track, track, mesh_diag, args)
        rows.append(row)
    return rows


def _cross_probe_patch_tracks(
    states,
    reg,
    scoring_evidence_masks,
    vertices,
    faces,
    patch_entries,
    track_summaries,
    args,
):
    """Optionally recover a pose anchor from a local patch before cross selection.

    The broad Stage-A candidate can contain the whole frame and the true handle.
    In that case its fitted pose is not a valid test of whether the handle patch can
    explain RGB motion.  Probing the largest local patches supplies that missing,
    independently fitted hypothesis without silently treating every tiny sliver as
    an anchor.
    """
    max_patches = max(0, int(args.stage_b_cross_probe_max_patches))
    if max_patches == 0:
        return {}, []
    by_source: Dict[int, list] = {}
    for key, entry in patch_entries.items():
        enriched = dict(entry)
        enriched["patch_key"] = key
        enriched["area"] = _cross_patch_area(vertices, faces, entry["faces"])
        by_source.setdefault(int(entry["source_part_id"]), []).append(enriched)

    anchor_updates = {}
    probe_rows = []
    min_fit = max(0.0, float(args.stage_b_cross_min_track_fit_iou))
    min_gain = max(0.0, float(args.stage_b_cross_probe_min_fit_gain))
    for source_part_id, entries in sorted(by_source.items()):
        original = dict(track_summaries.get(source_part_id, {}))
        original_fit = float(original.get("median_fit_iou") or 0.0)
        best = None
        for entry in sorted(entries, key=lambda row: -row["area"])[:max_patches]:
            sub_vertices, sub_faces = _local_submesh(vertices, faces[entry["faces"]])
            if sub_vertices.shape[0] < 3 or sub_faces.shape[0] == 0:
                continue
            observations = _fit_part_track(
                states, reg, scoring_evidence_masks, sub_vertices, sub_faces, args
            )
            summary = _summarize_part_track(source_part_id, observations, args)
            fit = float(summary.get("median_fit_iou") or 0.0)
            eligible = summary.get("status") == "moving_candidate" and fit >= min_fit
            row = {
                "source_part_id": int(source_part_id),
                "patch_id": int(entry["patch_id"]),
                "patch_key": entry["patch_key"],
                "faces": int(entry["faces"].size),
                "area": float(entry["area"]),
                "median_fit_iou": summary.get("median_fit_iou"),
                "successful_states": int(summary.get("successful_states", 0)),
                "status": str(summary.get("status", "unknown")),
                "eligible": bool(eligible),
                "selected_as_anchor": False,
            }
            probe_rows.append(row)
            if eligible and fit >= original_fit + min_gain and (best is None or fit > best[0]):
                best = (fit, observations, summary, row)
        if best is None:
            continue
        _fit, observations, summary, row = best
        replacement = dict(original)
        replacement.update(summary)
        replacement["broad_candidate_median_fit_iou"] = original.get("median_fit_iou")
        replacement["pose_source"] = "cross_patch_probe"
        replacement["pose_anchor_patch_id"] = int(row["patch_id"])
        row["selected_as_anchor"] = True
        anchor_updates[source_part_id] = {
            "observations": observations,
            "summary": replacement,
            "patch_key": str(row["patch_key"]),
        }
    return anchor_updates, probe_rows


def _build_cross_candidate_views(
    states,
    reg,
    scoring_evidence_masks,
    vertices,
    faces,
    patch_entries,
    candidate_lookup,
    tracks_by_part,
    args,
):
    """Cache one depth-composite scene for every patch x track hypothesis.

    Every entry is a patch in canonical mesh coordinates.  Each track is also an
    absolute canonical-to-world transform, so applying a hinge track to a handle
    patch is valid whenever they are one rigid physical part.  Rest footprints are
    shared by all hypotheses; only moved footprints vary by track.
    """
    import cv2

    by_state = {st["state_id"]: st for st in states}
    pose_by_track = {int(pid): _track_pose_lookup(track) for pid, track in tracks_by_part.items()}
    scale = max(1, int(args.stage_b_patch_composite_downscale))
    scaled_dilate = max(0, int(math.ceil(float(max(0, int(args.stage_b_patch_change_dilate))) / float(scale))))
    kernel = np.ones((3, 3), dtype=np.uint8)
    views = []
    for sid, masks in sorted((scoring_evidence_masks or {}).items()):
        st = by_state.get(sid)
        record = reg.get(sid) or {}
        body_T = record.get("T_world_object")
        if st is None or body_T is None or not masks:
            continue
        active_tracks = {
            int(pid): poses[str(sid)]
            for pid, poses in pose_by_track.items()
            if str(sid) in poses
        }
        if not active_tracks:
            continue
        cameras = rs._load_state_cameras(st["state_dir"])
        if not cameras:
            continue
        body_world = rs._apply_transform(vertices, np.asarray(body_T, dtype=np.float64))
        composite_views = _build_patch_composite_views(body_world, faces, cameras, masks, args)
        if len(composite_views) < int(args.stage_b_patch_min_observed_cameras):
            continue
        moving_world_by_track = {
            pid: rs._apply_transform(vertices, T_world_part)
            for pid, T_world_part in active_tracks.items()
        }
        group = str(record.get("placement_group") or "default")
        for base_view in composite_views:
            evidence = np.asarray(base_view["evidence"], dtype=bool)
            shape = (int(evidence.shape[0]), int(evidence.shape[1]))
            if scaled_dilate:
                evidence = cv2.dilate(evidence.astype(np.uint8), kernel, iterations=scaled_dilate) > 0
            rest = {
                key: _sparse_top_depth(body_world, faces, entry["faces"], base_view["camera"], shape)
                for key, entry in patch_entries.items()
            }
            moved = {}
            for key, entry in patch_entries.items():
                for track_part_id, moved_world in moving_world_by_track.items():
                    hypothesis = _cross_hypothesis_key(key, track_part_id)
                    moved[hypothesis] = _sparse_top_depth(
                        moved_world, faces, entry["faces"], base_view["camera"], shape
                    )
            views.append({
                "state_id": str(sid),
                "serial": base_view["serial"],
                "group": group,
                "shape": shape,
                "static_base": _joint_static_base(base_view["full_layers"], candidate_lookup),
                "evidence": evidence.reshape(-1),
                "rest": rest,
                "moved": moved,
            })
    return views


def _cross_predicted_change(view, selected_hypotheses, hypotheses):
    """Visible ownership change for a selected patch x track subset.

    If a track is not fitted for this state, its patch stays at rest for that view.
    A patch can appear at most once in ``selected_hypotheses``; the selector enforces
    this exclusivity before calling here.
    """
    selected_by_patch = {}
    for hypothesis in selected_hypotheses:
        spec = hypotheses.get(hypothesis)
        if spec is None or hypothesis not in view["moved"]:
            continue
        selected_by_patch[spec["patch_key"]] = hypothesis
    static = view["static_base"].copy()
    rest_top = np.full(static.shape, np.inf, dtype=np.float32)
    moved_top = np.full(static.shape, np.inf, dtype=np.float32)
    for patch_key, sparse in view["rest"].items():
        if patch_key in selected_by_patch:
            _scatter_min(rest_top, sparse)
        else:
            _scatter_min(static, sparse)
    for hypothesis in selected_by_patch.values():
        _scatter_min(moved_top, view["moved"][hypothesis])
    return (rest_top < static) != (moved_top < static)


def _cross_joint_objective(views, selected_hypotheses, hypotheses, floor_pixels, spurious_weight):
    """Group-normalized explaining-away objective plus raw audit counts.

    Raw pixel totals would prefer a track that happened to be fitted in more views.
    We instead average its evidence-normalized score within each placement group and
    sum the group means.  The raw explained/spurious counts remain in the report.
    """
    explained = 0
    spurious = 0
    evidence_pixels = 0
    unexplained_pixels = 0
    group_scores: Dict[str, list] = {}
    used_views = 0
    for view in views:
        predicted = _cross_predicted_change(view, selected_hypotheses, hypotheses)
        count = int(np.count_nonzero(predicted))
        observed = int(np.count_nonzero(view["evidence"]))
        if count < int(floor_pixels):
            count = 0
            overlap = 0
            outside = 0
        else:
            overlap = int(np.count_nonzero(predicted & view["evidence"]))
            outside = count - overlap
        explained += overlap
        spurious += outside
        evidence_pixels += observed
        unexplained_pixels += observed - overlap
        denom = float(max(observed, 1))
        group_scores.setdefault(str(view["group"]), []).append(
            (float(overlap) - float(spurious_weight) * float(outside)) / denom
        )
        used_views += 1
    score = float(sum(float(np.mean(values)) for values in group_scores.values()))
    return {
        "explained_pixels": int(explained),
        "spurious_pixels": int(spurious),
        "evidence_pixels": int(evidence_pixels),
        "unexplained_pixels": int(unexplained_pixels),
        "coverage": float(explained) / float(evidence_pixels) if evidence_pixels else 0.0,
        "score": score,
        "groups": int(len(group_scores)),
        "used_views": int(used_views),
    }


def _cross_coverage_passes(totals, args):
    return float(totals.get("coverage", 0.0)) >= float(args.stage_b_cross_min_coverage)


def _cross_select_hypotheses(views, hypotheses, floor_pixels, args):
    """Greedily select mutually exclusive patch x track hypotheses."""
    selected = []
    selected_patches = set()
    trace = []
    current = _cross_joint_objective(
        views, selected, hypotheses, floor_pixels, float(args.stage_b_patch_joint_spurious_weight)
    )
    first_gain = None
    remaining = list(sorted(hypotheses))
    max_selected = int(args.stage_b_cross_max_selected)
    while remaining and (max_selected <= 0 or len(selected) < max_selected):
        best = None
        for hypothesis in remaining:
            spec = hypotheses[hypothesis]
            if spec["patch_key"] in selected_patches:
                continue
            proposal = _cross_joint_objective(
                views,
                selected + [hypothesis],
                hypotheses,
                floor_pixels,
                float(args.stage_b_patch_joint_spurious_weight),
            )
            gain = float(proposal["score"] - current["score"])
            if best is None or gain > best[0]:
                best = (gain, hypothesis, proposal)
        if best is None:
            break
        gain, hypothesis, proposal = best
        floor_gain = float(args.stage_b_cross_min_gain)
        if first_gain is not None:
            floor_gain = max(floor_gain, float(args.stage_b_cross_min_gain_frac) * float(first_gain))
        if gain < floor_gain:
            trace.append({
                "hypothesis": hypothesis,
                "gain": float(gain),
                "accepted": False,
                "reason": "gain_below_floor",
                "gain_floor": float(floor_gain),
            })
            break
        selected.append(hypothesis)
        selected_patches.add(hypotheses[hypothesis]["patch_key"])
        remaining.remove(hypothesis)
        if first_gain is None:
            first_gain = gain
        current = proposal
        trace.append({
            "hypothesis": hypothesis,
            "patch_key": hypotheses[hypothesis]["patch_key"],
            "track_part_id": int(hypotheses[hypothesis]["track_part_id"]),
            "gain": float(gain),
            "accepted": True,
            **proposal,
        })
    return selected, trace, current


def _cross_hypothesis_stats(views, hypotheses, hypothesis_ids, floor_pixels, args):
    """Independent recurrence gates for selected cross-candidate hypotheses."""
    stats = {}
    by_state: Dict[str, list] = {}
    for view in views:
        by_state.setdefault(str(view["state_id"]), []).append(view)
    for hypothesis in hypothesis_ids:
        row = {
            "state_count": 0,
            "groups": set(),
            "max_change_precision": 0.0,
            "sum_change_precision": 0.0,
            "predicted_change_pixels": 0,
            "overlap_change_pixels": 0,
            "used_states": [],
        }
        for sid, state_views in by_state.items():
            precision_sum = 0.0
            change_cams = 0
            predicted_pixels = 0
            overlap_pixels = 0
            for view in state_views:
                predicted = _cross_predicted_change(view, [hypothesis], hypotheses)
                count = int(np.count_nonzero(predicted))
                if count < int(floor_pixels):
                    continue
                overlap = int(np.count_nonzero(predicted & view["evidence"]))
                precision_sum += float(overlap) / float(count)
                change_cams += 1
                predicted_pixels += count
                overlap_pixels += overlap
            if change_cams < int(args.stage_b_patch_min_observed_cameras):
                continue
            precision = precision_sum / float(change_cams)
            row["state_count"] += 1
            row["sum_change_precision"] += precision
            row["max_change_precision"] = max(row["max_change_precision"], precision)
            row["predicted_change_pixels"] += predicted_pixels
            row["overlap_change_pixels"] += overlap_pixels
            row["used_states"].append(sid)
            if precision >= float(args.stage_b_patch_min_change_precision):
                row["groups"].add(str(state_views[0]["group"]))
        stats[hypothesis] = row
    return stats


def _cross_hypothesis_passes(stats, args):
    return _joint_gates_pass(stats, args)


def _write_cross_joint_overlays(views, hypotheses, kept, by_state, args, overlay_dir):
    """Write the same green/red/blue audit maps for the global selected subset."""
    import cv2

    max_states = int(args.stage_b_patch_joint_overlay_states)
    max_cameras = int(args.stage_b_patch_joint_overlay_cameras)
    if max_states <= 0 or max_cameras <= 0 or not views:
        return []
    area_by_state: Dict[str, int] = {}
    for view in views:
        area_by_state[view["state_id"]] = area_by_state.get(view["state_id"], 0) + int(np.count_nonzero(view["evidence"]))
    chosen_states = [sid for sid, _area in sorted(area_by_state.items(), key=lambda item: -item[1])[:max_states]]
    os.makedirs(overlay_dir, exist_ok=True)
    written = []
    for sid in chosen_states:
        state_views = [view for view in views if view["state_id"] == sid][:max_cameras]
        st = by_state.get(sid)
        for view in state_views:
            height, width = view["shape"]
            predicted = _cross_predicted_change(view, kept, hypotheses).reshape(height, width)
            evidence = view["evidence"].reshape(height, width)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            image = rs._load_image_bgr(st["state_dir"], view["serial"], args.image_dirname) if st else None
            if image is not None:
                canvas = (cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) * 0.45).astype(np.uint8)
            canvas[evidence & ~predicted] = (230, 120, 0)
            canvas[predicted & evidence] = (60, 200, 60)
            canvas[predicted & ~evidence] = (40, 40, 230)
            path = os.path.join(overlay_dir, f"cross_{sid}_{view['serial']}.jpg")
            cv2.imwrite(path, canvas)
            written.append(path)
    return written


def _cross_candidate_patch_refine(
    states,
    reg,
    proposal_evidence_masks,
    scoring_evidence_masks,
    vertices,
    faces,
    face_part,
    tracks,
    part_summaries,
    args,
    output_dir,
):
    """Jointly assign every candidate patch to one competing candidate track.

    This is deliberately not a 1-DOF assumption.  An articulated handle can borrow
    its hinge's reliable SE(3) track, while truly separable components retain distinct
    tracks and compete in the same explaining-away scene.
    """
    refined_dir = os.path.join(output_dir, "segmentation", "stage_b", "patch_refined")
    summary_by_part = {int(row.get("part_id")): row for row in part_summaries}
    source_ids = _cross_source_ids(part_summaries, args)
    track_summaries = {pid: dict(summary_by_part[pid]) for pid in source_ids}
    candidate_tracks = {pid: tracks.get(f"part_{pid:02d}", []) for pid in source_ids}
    rows = []
    patch_entries = {}
    source_faces = 0
    patches_dir = os.path.join(refined_dir, "patches")
    for source_part_id in source_ids:
        candidate_faces = np.nonzero(face_part == source_part_id)[0]
        source_faces += int(candidate_faces.size)
        patches = _candidate_face_patches(vertices, faces, candidate_faces, args)
        scored = [(patch_id, patch_faces) for patch_id, patch_faces in enumerate(patches)
                  if patch_faces.size >= int(args.stage_b_patch_min_faces)]
        cap = int(args.stage_b_patch_joint_max_patches)
        if cap > 0 and len(scored) > cap:
            largest = sorted(scored, key=lambda row: -row[1].size)
            kept_patch_ids = {patch_id for patch_id, _patch_faces in largest[:cap]}
            for patch_id, patch_faces in largest[cap:]:
                rows.append({"source_part_id": source_part_id, "patch_id": patch_id,
                             "status": "skipped_cross_patch_cap", "faces": int(patch_faces.size)})
            scored = [(patch_id, patch_faces) for patch_id, patch_faces in scored if patch_id in kept_patch_ids]
        for patch_id, patch_faces in scored:
            key = _cross_patch_key(source_part_id, patch_id)
            patch_entries[key] = {
                "source_part_id": source_part_id,
                "patch_id": patch_id,
                "faces": np.asarray(patch_faces, dtype=np.int64),
            }
        if scored:
            _export_candidate_patches(
                patches_dir,
                vertices,
                faces,
                source_part_id,
                {patch_id: patch_faces for patch_id, patch_faces in scored},
            )

    probe_updates, probe_rows = _cross_probe_patch_tracks(
        states, reg, scoring_evidence_masks, vertices, faces, patch_entries,
        track_summaries, args,
    )
    anchor_tracks = {
        int(part_id): update["observations"]
        for part_id, update in probe_updates.items()
        if update.get("observations")
    }
    anchor_patch_keys = {
        int(part_id): update.get("patch_key")
        for part_id, update in probe_updates.items()
        if update.get("patch_key") is not None
    }
    for part_id, update in probe_updates.items():
        candidate_tracks[int(part_id)] = update["observations"]
        track_summaries[int(part_id)] = update["summary"]
    track_ids = _cross_track_ids(source_ids, track_summaries, candidate_tracks, reg, args)
    tracks_by_part = {pid: candidate_tracks[pid] for pid in track_ids}

    if not patch_entries or not track_ids:
        diagnostics = {
            "status": "no_cross_candidate_pool",
            "source_candidate_faces": int(source_faces),
            "kept_faces": 0,
            "part_count": 0,
            "cross_candidate": {
                "source_part_ids": source_ids,
                "track_part_ids": track_ids,
                "track_min_fit_ratio": float(args.stage_b_cross_track_min_fit_ratio),
                "track_min_fit_iou": float(args.stage_b_cross_min_track_fit_iou),
                "patch_count": int(len(patch_entries)),
                "track_candidates": [
                    {
                        "part_id": int(pid),
                        "median_fit_iou": track_summaries.get(pid, {}).get("median_fit_iou"),
                        "pose_source": track_summaries.get(pid, {}).get("pose_source", "broad_candidate"),
                    }
                    for pid in source_ids
                ],
                "patch_track_probes": probe_rows,
            },
            "patches": rows + probe_rows,
        }
        payload = _write_stage_b_motion_refined_outputs(
            output_dir, vertices, faces, np.full(faces.shape[0], -1, dtype=np.int64), diagnostics,
            int(args.viewer_max_faces), dirname="patch_refined", mesh_filename="mesh_labeled_patch_refined.ply",
            report_filename="patch_refinement.json",
        )
        return np.full(faces.shape[0], -1, dtype=np.int64), payload, {}

    hypotheses = {}
    for patch_key, entry in patch_entries.items():
        for track_part_id in track_ids:
            hypothesis = _cross_hypothesis_key(patch_key, track_part_id)
            hypotheses[hypothesis] = {
                "patch_key": patch_key,
                "source_part_id": int(entry["source_part_id"]),
                "patch_id": int(entry["patch_id"]),
                "track_part_id": int(track_part_id),
            }
    candidate_lookup = np.zeros(faces.shape[0], dtype=bool)
    for entry in patch_entries.values():
        candidate_lookup[entry["faces"]] = True
    floor_pixels = max(
        int(args.stage_b_patch_composite_min_change_pixels),
        int(math.ceil(float(args.stage_b_patch_min_change_pixels) /
                      float(max(1, int(args.stage_b_patch_composite_downscale)) ** 2))),
    )
    current_tracks = dict(tracks_by_part)
    mesh_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    best = None
    rounds = []
    current_updates = {}
    for round_index in range(max(0, int(args.stage_b_patch_joint_refit_rounds)) + 1):
        views = _build_cross_candidate_views(
            states, reg, scoring_evidence_masks, vertices, faces, patch_entries,
            candidate_lookup, current_tracks, args,
        )
        if not views:
            rounds.append({"round": round_index, "status": "no_views"})
            break
        selected, trace, totals = _cross_select_hypotheses(views, hypotheses, floor_pixels, args)
        stats = _cross_hypothesis_stats(views, hypotheses, selected, floor_pixels, args)
        kept = [hypothesis for hypothesis in selected if _cross_hypothesis_passes(stats[hypothesis], args)]
        coverage_pass = _cross_coverage_passes(totals, args)
        if kept and not coverage_pass:
            kept = []
        track_diagnostics = _cross_round_track_diagnostics(
            current_tracks, anchor_tracks, mesh_diag, args
        )
        result = {
            "round": round_index,
            "views": views,
            "selected": selected,
            "kept": kept,
            "stats": stats,
            "trace": trace,
            "totals": totals,
            "track_updates": dict(current_updates),
            "track_diagnostics": track_diagnostics,
            "coverage_pass": bool(coverage_pass),
        }
        round_report = {
            "round": round_index,
            "selected": selected,
            "kept": kept,
            "explained_pixels": int(totals["explained_pixels"]),
            "spurious_pixels": int(totals["spurious_pixels"]),
            "evidence_pixels": int(totals["evidence_pixels"]),
            "unexplained_pixels": int(totals["unexplained_pixels"]),
            "coverage": float(totals["coverage"]),
            "coverage_pass": bool(coverage_pass),
            "coverage_floor": float(args.stage_b_cross_min_coverage),
            "score": float(totals["score"]),
            "groups": int(totals["groups"]),
            "used_views": int(totals["used_views"]),
            "trace": trace,
            "track_diagnostics": track_diagnostics,
            "refit_attempts": [],
        }
        rounds.append(round_report)
        if _joint_round_is_better(best, kept, totals["score"]):
            best = result
        if round_index >= int(args.stage_b_patch_joint_refit_rounds) or not kept:
            break
        next_tracks = dict(current_tracks)
        next_updates = dict(current_updates)
        for track_part_id in sorted({hypotheses[h]["track_part_id"] for h in kept}):
            anchor_locked = bool(getattr(args, "stage_b_cross_refit_anchor_lock", True)) and int(track_part_id) in anchor_tracks
            anchor_patch_key = anchor_patch_keys.get(int(track_part_id))
            subset_faces, refit_patch_keys = _cross_refit_faces(
                kept,
                hypotheses,
                patch_entries,
                track_part_id,
                anchor_patch_key,
                anchor_locked,
            )
            refit_report = {
                "track_part_id": int(track_part_id),
                "anchor_locked": bool(anchor_locked),
                "anchor_patch_key": anchor_patch_key,
                "patch_keys": refit_patch_keys,
                "faces": int(subset_faces.size),
            }
            if anchor_locked and anchor_patch_key not in patch_entries:
                refit_report["status"] = "skipped_missing_anchor_patch"
                round_report["refit_attempts"].append(refit_report)
                continue
            sub_vertices, sub_faces = _local_submesh(vertices, faces[subset_faces])
            if sub_vertices.shape[0] < 3 or sub_faces.shape[0] == 0:
                refit_report["status"] = "skipped_small_subset"
                round_report["refit_attempts"].append(refit_report)
                continue
            observations = _fit_part_track(states, reg, scoring_evidence_masks, sub_vertices, sub_faces, args)
            summary = _summarize_part_track(track_part_id, observations, args)
            refit_report["actual_fit"] = _cross_track_fit_diagnostics(observations)
            if anchor_locked:
                anchor_drift = _cross_anchor_drift(
                    anchor_tracks[int(track_part_id)], observations, mesh_diag, args
                )
                refit_report["anchor_drift"] = anchor_drift
                if not anchor_drift["passes"]:
                    refit_report["status"] = "rejected_anchor_drift"
                    round_report["refit_attempts"].append(refit_report)
                    continue
            original = track_summaries.get(track_part_id, {})
            summary["median_fit_iou_subset"] = summary.get("median_fit_iou")
            summary["median_fit_iou"] = original.get("median_fit_iou")
            summary["status"] = original.get("status") or summary.get("status")
            summary["refit"] = True
            next_tracks[track_part_id] = observations
            next_updates[f"part_{track_part_id:02d}"] = {"observations": observations, "summary": summary}
            refit_report["status"] = "accepted"
            round_report["refit_attempts"].append(refit_report)
        if next_tracks == current_tracks:
            break
        current_tracks = next_tracks
        current_updates = next_updates

    output_face_part = np.full(faces.shape[0], -1, dtype=np.int64)
    track_updates = {}
    if best is None:
        kept = []
        selected = []
        stats = {}
        totals = {
            "explained_pixels": 0,
            "spurious_pixels": 0,
            "evidence_pixels": 0,
            "unexplained_pixels": 0,
            "coverage": 0.0,
            "score": 0.0,
            "groups": 0,
            "used_views": 0,
        }
        selected_track_ids = []
        coverage_pass = False
    else:
        kept = best["kept"]
        selected = best["selected"]
        stats = best["stats"]
        totals = best["totals"]
        # Stage B and Stage C both address tracks as ``part_00``.  Probe updates
        # used integer ids internally, so normalize them before publishing.
        track_updates = {
            f"part_{int(part_id):02d}": update
            for part_id, update in probe_updates.items()
        }
        if best["round"] > 0:
            track_updates.update(best["track_updates"])
        selected_track_ids = sorted({int(hypotheses[h]["track_part_id"]) for h in kept})
        coverage_pass = bool(best.get("coverage_pass", False))
        for hypothesis in kept:
            spec = hypotheses[hypothesis]
            output_face_part[patch_entries[spec["patch_key"]]["faces"]] = int(spec["track_part_id"])

    selected_set = set(selected)
    kept_set = set(kept)
    coverage_rejected_set = set(selected) if best is not None and not coverage_pass else set()
    selection_gain = {
        str(row["hypothesis"]): float(row["gain"])
        for row in (best or {}).get("trace", [])
        if row.get("accepted")
    }
    selection_rank = {hypothesis: index + 1 for index, hypothesis in enumerate(selected)}
    for hypothesis, spec in sorted(hypotheses.items()):
        stats_row = stats.get(hypothesis, {})
        state_count = int(stats_row.get("state_count", 0))
        mean_precision = float(stats_row.get("sum_change_precision", 0.0)) / float(state_count) if state_count else 0.0
        pooled = _pooled_change_precision(
            stats_row.get("predicted_change_pixels", 0), stats_row.get("overlap_change_pixels", 0)
        )
        if hypothesis in kept_set:
            status = "kept"
        elif hypothesis in coverage_rejected_set:
            status = "selected_global_coverage_failed"
        elif hypothesis in selected_set:
            status = "selected_gate_failed"
        else:
            status = "rejected_not_selected"
        rows.append({
            "source_part_id": int(spec["source_part_id"]),
            "patch_id": int(spec["patch_id"]),
            "track_part_id": int(spec["track_part_id"]),
            "hypothesis": hypothesis,
            "status": status,
            "selection_gain": selection_gain.get(hypothesis),
            "selection_rank": selection_rank.get(hypothesis),
            "faces": int(patch_entries[spec["patch_key"]]["faces"].size),
            "observed_states": state_count,
            "motion_groups": int(len(stats_row.get("groups", set()))),
            "used_states": sorted(set(stats_row.get("used_states", []))),
            "max_change_precision": float(stats_row.get("max_change_precision", 0.0)),
            "mean_change_precision": mean_precision,
            "pooled_change_precision": float(pooled),
            "predicted_change_pixels": int(stats_row.get("predicted_change_pixels", 0)),
            "overlap_change_pixels": int(stats_row.get("overlap_change_pixels", 0)),
            "obj_path": os.path.join(patches_dir, f"src{int(spec['source_part_id']):02d}_patch{int(spec['patch_id']):02d}.obj"),
        })

    overlay_dir = None
    if best is not None and kept:
        overlay_dir = os.path.join(refined_dir, "cross_overlays")
        _write_cross_joint_overlays(best["views"], hypotheses, kept, {st["state_id"]: st for st in states}, args, overlay_dir)
    diagnostics = {
        "status": "ok" if kept else "no_patches_kept",
        "source_candidate_faces": int(source_faces),
        "kept_faces": int(np.count_nonzero(output_face_part >= 0)),
        "part_count": int(len(selected_track_ids)),
        "evidence_source": str(args.stage_b_patch_evidence_source),
        "score_mode": "cross_candidate_joint",
        "depth_raster_backend": _DEPTH_RASTER_BACKEND,
        "cross_candidate": {
            "enabled": True,
            "source_part_ids": source_ids,
            "track_part_ids": track_ids,
            "track_min_fit_ratio": float(args.stage_b_cross_track_min_fit_ratio),
            "track_min_fit_iou": float(args.stage_b_cross_min_track_fit_iou),
            "selected_track_part_ids": selected_track_ids,
            "patch_count": int(len(patch_entries)),
            "hypothesis_count": int(len(hypotheses)),
            "rounds": rounds,
            "used_round": None if best is None else int(best["round"]),
            "refit_rejected": bool(best is not None and len(rounds) > 1 and best["round"] == 0),
            "used_track_diagnostics": [] if best is None else best.get("track_diagnostics", []),
            "explained_pixels": int(totals["explained_pixels"]),
            "spurious_pixels": int(totals["spurious_pixels"]),
            "evidence_pixels": int(totals["evidence_pixels"]),
            "unexplained_pixels": int(totals["unexplained_pixels"]),
            "coverage": float(totals["coverage"]),
            "coverage_pass": bool(coverage_pass),
            "coverage_floor": float(args.stage_b_cross_min_coverage),
            "normalized_score": float(totals["score"]),
            "groups": int(totals["groups"]),
            "overlay_dir": overlay_dir,
            "track_candidates": [
                {
                    "part_id": int(pid),
                    "median_fit_iou": track_summaries.get(pid, {}).get("median_fit_iou"),
                    "pose_source": track_summaries.get(pid, {}).get("pose_source", "broad_candidate"),
                    "pose_anchor_patch_id": track_summaries.get(pid, {}).get("pose_anchor_patch_id"),
                }
                for pid in source_ids
            ],
            "patch_track_probes": probe_rows,
        },
        "patches": rows,
    }
    payload = _write_stage_b_motion_refined_outputs(
        output_dir, vertices, faces, output_face_part, diagnostics, int(args.viewer_max_faces),
        dirname="patch_refined", mesh_filename="mesh_labeled_patch_refined.ply", report_filename="patch_refinement.json",
    )
    return output_face_part, payload, track_updates


def _patch_refine_face_part_from_tracks(
    states,
    reg,
    proposal_evidence_masks,
    rgb_evidence_masks,
    vertices,
    faces,
    face_part,
    tracks,
    part_summaries,
    args,
    output_dir,
):
    """Select local candidate patches that repeatedly prefer their moving pose.

    A Stage-A component can contain a real handle plus static wooden frame surfaces.
    Fitting the whole component is useful to propose a transform, but keeping the
    whole component is not.  This pass holds that transform fixed and asks every
    locally connected mesh patch whether its predicted rest-vs-moving image change
    recurs in independently captured RGB motion evidence.  Hybrid residuals remain
    useful to *propose* a moving pose, but cannot certify a final patch by default.
    """
    if bool(args.skip_stage_b_patch_refine):
        return None, None, None
    evidence_source = str(args.stage_b_patch_evidence_source)
    score_mode = str(args.stage_b_patch_score_mode)
    scoring_evidence_masks = rgb_evidence_masks if evidence_source == "rgb_motion" else proposal_evidence_masks
    if score_mode == "composite_joint" and not bool(args.skip_stage_b_cross_candidate_selection):
        return _cross_candidate_patch_refine(
            states,
            reg,
            proposal_evidence_masks,
            scoring_evidence_masks,
            vertices,
            faces,
            face_part,
            tracks,
            part_summaries,
            args,
            output_dir,
        )
    summary_by_part = {int(p.get("part_id")): p for p in part_summaries}
    by_state = {st["state_id"]: st for st in states}
    output_face_part = np.full(faces.shape[0], -1, dtype=np.int64)
    rows = []
    source_faces = 0
    kept_faces = 0
    joint_info = {}
    track_updates = {}

    for part_id in sorted(int(x) for x in np.unique(face_part) if int(x) >= 0):
        summary = summary_by_part.get(part_id, {})
        fit = summary.get("median_fit_iou")
        if summary.get("status") != "moving_candidate" or fit is None or float(fit) < float(args.stage_b_patch_min_fit_iou):
            rows.append({"source_part_id": part_id, "status": "skipped_low_fit_or_static", "median_fit_iou": fit})
            continue
        candidate = np.nonzero(face_part == part_id)[0]
        source_faces += int(candidate.size)
        patches = _candidate_face_patches(vertices, faces, candidate, args)
        if score_mode == "composite_joint":
            joint = _joint_refine_candidate(
                states,
                reg,
                proposal_evidence_masks,
                scoring_evidence_masks,
                vertices,
                faces,
                part_id,
                patches,
                tracks.get(f"part_{part_id:02d}", []),
                summary,
                args,
                output_dir,
            )
            rows.extend(joint["rows"])
            if joint["kept_faces"].size:
                output_face_part[joint["kept_faces"]] = part_id
                kept_faces += int(joint["kept_faces"].size)
            else:
                rows.append({"source_part_id": part_id, "status": "no_patch_kept", "median_fit_iou": fit})
            joint_info[f"part_{part_id:02d}"] = joint["info"]
            if joint.get("track_update") is not None:
                track_updates[f"part_{part_id:02d}"] = {
                    "observations": joint["track_update"],
                    "summary": joint.get("summary_update"),
                }
            continue
        if score_mode == "composite_depth":
            # Render the fixed full scene once per state, then let every local
            # patch compete against the same depth/owner evidence.
            patch_stats = {}
            for patch_id, patch_faces in enumerate(patches):
                if patch_faces.size < int(args.stage_b_patch_min_faces):
                    rows.append({"source_part_id": part_id, "patch_id": patch_id, "status": "skipped_too_small", "faces": int(patch_faces.size)})
                    continue
                patch_stats[patch_id] = {
                    "faces": patch_faces,
                    "state_count": 0,
                    "groups": set(),
                    "max_change_precision": 0.0,
                    "sum_change_precision": 0.0,
                    "predicted_change_pixels": 0,
                    "overlap_change_pixels": 0,
                    "used_states": [],
                }
            for obs in tracks.get(f"part_{part_id:02d}", []):
                if obs.get("status") != "fit":
                    continue
                sid = obs.get("state_id")
                st = by_state.get(sid)
                record = reg.get(sid) or {}
                body_T = record.get("T_world_object")
                part_T = obs.get("T_world_part")
                masks = scoring_evidence_masks.get(sid)
                if st is None or body_T is None or part_T is None or not masks:
                    continue
                cameras = rs._load_state_cameras(st["state_dir"])
                if not cameras:
                    continue
                body_world = rs._apply_transform(vertices, np.asarray(body_T, dtype=np.float64))
                moving_world = rs._apply_transform(vertices, np.asarray(part_T, dtype=np.float64))
                composite_views = _build_patch_composite_views(body_world, faces, cameras, masks, args)
                if len(composite_views) < int(args.stage_b_patch_min_observed_cameras):
                    continue
                group = str(record.get("placement_group") or obs.get("placement_group") or "default")
                for stats in patch_stats.values():
                    change_precision, change_cams, predicted_pixels, overlap_pixels = _state_patch_composite_change_precision(
                        body_world,
                        moving_world,
                        faces,
                        stats["faces"],
                        composite_views,
                        int(args.stage_b_patch_change_dilate),
                        int(args.stage_b_patch_min_change_pixels),
                        int(args.stage_b_patch_composite_min_change_pixels),
                        int(args.stage_b_patch_composite_downscale),
                    )
                    if change_cams < int(args.stage_b_patch_min_observed_cameras):
                        continue
                    stats["max_change_precision"] = max(stats["max_change_precision"], change_precision)
                    stats["sum_change_precision"] += change_precision
                    stats["predicted_change_pixels"] += predicted_pixels
                    stats["overlap_change_pixels"] += overlap_pixels
                    stats["state_count"] += 1
                    stats["used_states"].append(sid)
                    if change_precision >= float(args.stage_b_patch_min_change_precision):
                        stats["groups"].add(group)
            selected = 0
            for patch_id, stats in patch_stats.items():
                state_count = int(stats["state_count"])
                mean_change_precision = stats["sum_change_precision"] / float(state_count) if state_count else 0.0
                pooled_change_precision = _pooled_change_precision(
                    stats["predicted_change_pixels"], stats["overlap_change_pixels"]
                )
                keep = (
                    state_count >= int(args.stage_b_patch_min_states)
                    and len(stats["groups"]) >= int(args.stage_b_patch_min_groups)
                    and stats["max_change_precision"] >= float(args.stage_b_patch_min_change_precision)
                    and mean_change_precision >= float(args.stage_b_patch_min_mean_change_precision)
                    and pooled_change_precision >= float(args.stage_b_patch_min_pooled_change_precision)
                )
                if keep:
                    output_face_part[stats["faces"]] = part_id
                    kept_faces += int(stats["faces"].size)
                    selected += 1
                rows.append({
                    "source_part_id": part_id,
                    "patch_id": patch_id,
                    "status": "kept" if keep else "rejected",
                    "faces": int(stats["faces"].size),
                    "observed_states": state_count,
                    "motion_groups": int(len(stats["groups"])),
                    "used_states": sorted(set(stats["used_states"])),
                    "max_advantage": 0.0,
                    "mean_advantage": 0.0,
                    "max_moving_support": 0.0,
                    "max_change_precision": float(stats["max_change_precision"]),
                    "mean_change_precision": float(mean_change_precision),
                    "pooled_change_precision": float(pooled_change_precision),
                    "predicted_change_pixels": int(stats["predicted_change_pixels"]),
                    "overlap_change_pixels": int(stats["overlap_change_pixels"]),
                })
            if selected == 0:
                rows.append({"source_part_id": part_id, "status": "no_patch_kept", "median_fit_iou": fit})
            continue
        selected = 0
        for patch_id, patch_faces in enumerate(patches):
            if patch_faces.size < int(args.stage_b_patch_min_faces):
                rows.append({"source_part_id": part_id, "patch_id": patch_id, "status": "skipped_too_small", "faces": int(patch_faces.size)})
                continue
            max_adv = -np.inf
            sum_adv = 0.0
            state_count = 0
            groups = set()
            max_moving = 0.0
            max_change_precision = 0.0
            sum_change_precision = 0.0
            predicted_change_pixels = 0
            overlap_change_pixels = 0
            used_states = []
            for obs in tracks.get(f"part_{part_id:02d}", []):
                if obs.get("status") != "fit":
                    continue
                sid = obs.get("state_id")
                st = by_state.get(sid)
                record = reg.get(sid) or {}
                body_T = record.get("T_world_object")
                part_T = obs.get("T_world_part")
                masks = scoring_evidence_masks.get(sid)
                if st is None or body_T is None or part_T is None or not masks:
                    continue
                cameras = rs._load_state_cameras(st["state_dir"])
                if not cameras:
                    continue
                body_world = rs._apply_transform(vertices, np.asarray(body_T, dtype=np.float64))
                moving_world = rs._apply_transform(vertices, np.asarray(part_T, dtype=np.float64))
                group = str(record.get("placement_group") or obs.get("placement_group") or "default")
                if score_mode == "centroid":
                    body_sup, body_cams = _state_patch_centroid_support(
                        body_world, faces, patch_faces, cameras, masks, float(args.stage_b_patch_min_facing_cos)
                    )
                    moving_sup, moving_cams = _state_patch_centroid_support(
                        moving_world, faces, patch_faces, cameras, masks, float(args.stage_b_patch_min_facing_cos)
                    )
                    if max(body_cams, moving_cams) < int(args.stage_b_patch_min_observed_cameras):
                        continue
                    advantage = moving_sup - body_sup
                    max_adv = max(max_adv, advantage)
                    max_moving = max(max_moving, moving_sup)
                    sum_adv += advantage
                    state_count += 1
                    used_states.append(sid)
                    if moving_sup >= float(args.stage_b_patch_min_moving_support) and advantage >= float(args.stage_b_patch_min_advantage):
                        groups.add(group)
                else:
                    change_precision, change_cams, predicted_pixels, overlap_pixels = _state_patch_change_precision(
                        body_world,
                        moving_world,
                        faces,
                        patch_faces,
                        cameras,
                        masks,
                        float(args.stage_b_patch_min_facing_cos),
                        int(args.stage_b_patch_change_dilate),
                        int(args.stage_b_patch_min_change_pixels),
                    )
                    if change_cams < int(args.stage_b_patch_min_observed_cameras):
                        continue
                    max_change_precision = max(max_change_precision, change_precision)
                    sum_change_precision += change_precision
                    predicted_change_pixels += predicted_pixels
                    overlap_change_pixels += overlap_pixels
                    state_count += 1
                    used_states.append(sid)
                    if change_precision >= float(args.stage_b_patch_min_change_precision):
                        groups.add(group)
            mean_adv = sum_adv / float(state_count) if state_count else 0.0
            mean_change_precision = sum_change_precision / float(state_count) if state_count else 0.0
            pooled_change_precision = _pooled_change_precision(predicted_change_pixels, overlap_change_pixels)
            if score_mode == "centroid":
                keep = (
                    state_count >= int(args.stage_b_patch_min_states)
                    and len(groups) >= int(args.stage_b_patch_min_groups)
                    and max_moving >= float(args.stage_b_patch_min_moving_support)
                    and max_adv >= float(args.stage_b_patch_min_advantage)
                    and mean_adv >= float(args.stage_b_patch_min_mean_advantage)
                )
            else:
                keep = (
                    state_count >= int(args.stage_b_patch_min_states)
                    and len(groups) >= int(args.stage_b_patch_min_groups)
                    and max_change_precision >= float(args.stage_b_patch_min_change_precision)
                    and mean_change_precision >= float(args.stage_b_patch_min_mean_change_precision)
                    and pooled_change_precision >= float(args.stage_b_patch_min_pooled_change_precision)
                )
            if keep:
                output_face_part[patch_faces] = part_id
                kept_faces += int(patch_faces.size)
                selected += 1
            rows.append({
                "source_part_id": part_id,
                "patch_id": patch_id,
                "status": "kept" if keep else "rejected",
                "faces": int(patch_faces.size),
                "observed_states": int(state_count),
                "motion_groups": int(len(groups)),
                "used_states": sorted(set(used_states)),
                "max_advantage": float(max_adv) if np.isfinite(max_adv) else 0.0,
                "mean_advantage": float(mean_adv),
                "max_moving_support": float(max_moving),
                "max_change_precision": float(max_change_precision),
                "mean_change_precision": float(mean_change_precision),
                "pooled_change_precision": float(pooled_change_precision),
                "predicted_change_pixels": int(predicted_change_pixels),
                "overlap_change_pixels": int(overlap_change_pixels),
            })
        if selected == 0:
            rows.append({"source_part_id": part_id, "status": "no_patch_kept", "median_fit_iou": fit})

    diagnostics = {
        "status": "ok" if kept_faces else "no_patches_kept",
        "source_candidate_faces": int(source_faces),
        "kept_faces": int(kept_faces),
        "part_count": int(len([x for x in np.unique(output_face_part) if int(x) >= 0])),
        "evidence_source": evidence_source,
        "score_mode": score_mode,
        "depth_raster_backend": _DEPTH_RASTER_BACKEND if score_mode in ("composite_depth", "composite_joint") else None,
        "joint": joint_info or None,
        "patches": rows,
        "params": {
            "min_fit_iou": args.stage_b_patch_min_fit_iou,
            "min_faces": args.stage_b_patch_min_faces,
            "normal_angle_deg": args.stage_b_patch_normal_angle_deg,
            "min_facing_cos": args.stage_b_patch_min_facing_cos,
            "min_states": args.stage_b_patch_min_states,
            "min_groups": args.stage_b_patch_min_groups,
            "min_change_pixels": args.stage_b_patch_min_change_pixels,
            "change_dilate": args.stage_b_patch_change_dilate,
            "min_change_precision": args.stage_b_patch_min_change_precision,
            "min_mean_change_precision": args.stage_b_patch_min_mean_change_precision,
            "min_pooled_change_precision": args.stage_b_patch_min_pooled_change_precision,
            "composite_downscale": args.stage_b_patch_composite_downscale,
            "composite_max_cameras": args.stage_b_patch_composite_max_cameras,
            "composite_min_change_pixels": args.stage_b_patch_composite_min_change_pixels,
            "min_moving_support": args.stage_b_patch_min_moving_support,
            "min_advantage": args.stage_b_patch_min_advantage,
            "joint_spurious_weight": args.stage_b_patch_joint_spurious_weight,
            "joint_min_gain_pixels": args.stage_b_patch_joint_min_gain_pixels,
            "joint_min_gain_frac": args.stage_b_patch_joint_min_gain_frac,
            "joint_max_selected": args.stage_b_patch_joint_max_selected,
            "joint_max_patches": args.stage_b_patch_joint_max_patches,
            "joint_refit_rounds": args.stage_b_patch_joint_refit_rounds,
        },
    }
    payload = _write_stage_b_motion_refined_outputs(
        output_dir, vertices, faces, output_face_part, diagnostics, int(args.viewer_max_faces),
        dirname="patch_refined", mesh_filename="mesh_labeled_patch_refined.ply", report_filename="patch_refinement.json",
    )
    return output_face_part, payload, (track_updates or None)


def _run_stage_b_observations(states, reg, vertices, faces, face_part, args, output_dir):
    """Fit each Stage-A movable candidate to beyond-body residual masks and score motion."""
    stage_dir = os.path.join(output_dir, "segmentation", "stage_b")
    os.makedirs(stage_dir, exist_ok=True)
    body_faces = faces[face_part < 0]
    part_ids = sorted(int(x) for x in np.unique(face_part) if int(x) >= 0)
    if not part_ids:
        return {"enabled": True, "status": "no_movable_parts", "parts": [], "states": {}}, {}

    source = str(args.stage_b_residual_source)
    if source == "full_mesh":
        residuals, residual_reports = _build_stage_b_residuals(
            states, reg, vertices, faces, args, output_dir,
            "residual_overlays", "full_mesh",
        )
        comparison_residuals, comparison_reports = _build_stage_b_residuals(
            states, reg, vertices, body_faces, args, output_dir,
            "residual_overlays_stage_a_body", "stage_a_body",
        )
    else:
        residuals, residual_reports = _build_stage_b_residuals(
            states, reg, vertices, body_faces, args, output_dir,
            "residual_overlays", "stage_a_body",
        )
        comparison_residuals, comparison_reports = _build_stage_b_residuals(
            states, reg, vertices, faces, args, output_dir,
            "residual_overlays_full_mesh", "full_mesh",
        )

    image_motion, image_motion_reports, image_motion_dir = _build_stage_b_image_motion(
        states, reg, vertices, faces, args, output_dir
    )
    fit_evidence, rgb_motion_evidence = _combine_stage_b_evidence(residuals, image_motion, args)

    # A full-mesh residual lives at the DEPLOYED part location. Before estimating
    # the part transform, canonical candidate vertices are still at their REST
    # location, so body-pose residual-drop refinement would erase the true handle.
    refinement_payload = None
    if source == "full_mesh":
        observation_face_part = face_part
        source_vertices = np.zeros(vertices.shape[0], dtype=bool)
        if np.any(face_part >= 0):
            source_vertices[np.unique(faces[face_part >= 0])] = True
        refinement_diag = {
            "status": "skipped_full_mesh_primary",
            "reason": "full_mesh_residual_requires_part_pose_before_vertex_refinement",
            "original_movable_vertices": int(np.count_nonzero(source_vertices)),
            "residual_seed_vertices": 0,
            "residual_eligible_vertices": 0,
            "refined_movable_vertices": int(np.count_nonzero(source_vertices)),
            "refined_part_count": int(len(part_ids)),
        }
        refinement_payload = refinement_diag
    else:
        observation_face_part, refinement_diag = _refine_face_part_by_residual(
            states, reg, residuals, vertices, faces, face_part, args
        )
        if refinement_diag is not None:
            refinement_payload = _write_stage_b_refined_outputs(
                output_dir, vertices, faces, observation_face_part, refinement_diag
            )
        else:
            observation_face_part = face_part
    part_summaries = []
    tracks = {}

    part_ids = sorted(int(x) for x in np.unique(observation_face_part) if int(x) >= 0)
    for part_id in part_ids:
        pf = faces[observation_face_part == part_id]
        part_vertices, part_faces = _local_submesh(vertices, pf)
        if part_vertices.shape[0] < 3 or part_faces.shape[0] == 0:
            continue
        observations = _fit_part_track(states, reg, fit_evidence, part_vertices, part_faces, args)
        tracks[f"part_{part_id:02d}"] = observations
        part_summaries.append(_summarize_part_track(part_id, observations, args))

    part_summaries.sort(key=lambda p: (p["status"] != "moving_candidate", -p["successful_states"], -(p["median_fit_iou"] or 0.0)))
    legacy_motion_face_part, motion_refinement_payload = _motion_refine_face_part_from_tracks(
        states, reg, fit_evidence, vertices, faces, observation_face_part, tracks, part_summaries, args, output_dir
    )
    patch_motion_face_part, patch_refinement_payload, patch_track_updates = _patch_refine_face_part_from_tracks(
        states,
        reg,
        fit_evidence,
        rgb_motion_evidence,
        vertices,
        faces,
        observation_face_part,
        tracks,
        part_summaries,
        args,
        output_dir,
    )
    if patch_track_updates:
        # A joint-subset refit produced better part poses; Stage C and the joint
        # fitting stage must consume those, not the broad-candidate proposals.
        for key, update in patch_track_updates.items():
            observations = update.get("observations")
            if observations:
                tracks[key] = observations
            summary_update = update.get("summary")
            if summary_update:
                for existing in part_summaries:
                    if int(existing.get("part_id", -1)) == int(summary_update.get("part_id", -2)):
                        existing.update(summary_update)
                        break
    if patch_motion_face_part is not None and np.any(patch_motion_face_part >= 0):
        primary_motion_face_part = patch_motion_face_part
        primary_motion_source = "patch_refined"
    elif (
        patch_refinement_payload is not None
        and str(args.stage_b_patch_evidence_source) == "rgb_motion"
        and str(args.stage_b_patch_score_mode) in ("composite_joint", "composite_depth", "raster_change")
    ):
        # Precision-first default: a failed RGB-validated patch pass must not hand
        # its broad legacy candidate to Stage C as if it were a moving handle.
        primary_motion_face_part = np.full(faces.shape[0], -1, dtype=np.int64)
        primary_motion_source = "no_rgb_validated_patch"
    else:
        primary_motion_face_part = legacy_motion_face_part
        primary_motion_source = "legacy_vertex_refined_fallback"
    # Cross-candidate selection labels selected faces by the track they borrowed,
    # not by their Stage-A source candidate.  Hand Stage C only those selected
    # tracks and the corresponding face union; otherwise it would also process the
    # broad, low-quality source track that the global competition intentionally lost.
    stage_c_observation_face_part = observation_face_part
    stage_c_part_summaries = part_summaries
    cross_info = (patch_refinement_payload or {}).get("cross_candidate") or {}
    selected_track_ids = {int(x) for x in cross_info.get("selected_track_part_ids", []) or []}
    if primary_motion_source == "patch_refined" and selected_track_ids:
        stage_c_observation_face_part = primary_motion_face_part
        stage_c_part_summaries = [
            row for row in part_summaries if int(row.get("part_id", -1)) in selected_track_ids
        ]
    _geometry_face_part, geometry_refinement_payload = _geometry_refine_face_part(
        vertices, faces, observation_face_part, part_summaries, args, output_dir
    )
    payload = {
        "enabled": True,
        "status": "ok",
        "residual_source": source,
        "states": residual_reports,
        "evidence_source": str(args.stage_b_evidence_source),
        "image_motion_states": image_motion_reports,
        "image_motion_overlay_dir": image_motion_dir,
        "evidence_state_count": int(len(fit_evidence)),
        "rgb_motion_evidence_state_count": int(len(rgb_motion_evidence)),
        "comparison_residual_source": "stage_a_body" if source == "full_mesh" else "full_mesh",
        "comparison_states": comparison_reports,
        "residual_overlay_dir": os.path.join(stage_dir, "residual_overlays"),
        "comparison_residual_overlay_dir": os.path.join(
            stage_dir,
            "residual_overlays_stage_a_body" if source == "full_mesh" else "residual_overlays_full_mesh",
        ),
        "parts": part_summaries,
        "tracks": tracks,
        "refinement": refinement_payload,
        "motion_refinement": motion_refinement_payload,
        "patch_refinement": patch_refinement_payload,
        "primary_motion_source": primary_motion_source,
        "geometry_refinement": geometry_refinement_payload,
        "params": {
            "residual_source": source,
            "evidence_source": args.stage_b_evidence_source,
            "image_motion_threshold": args.stage_b_image_motion_threshold,
            "image_motion_object_band": args.stage_b_image_motion_object_band,
            "hybrid_residual_dilate": args.stage_b_hybrid_residual_dilate,
            "patch_evidence_source": args.stage_b_patch_evidence_source,
            "patch_score_mode": args.stage_b_patch_score_mode,
            "cross_candidate_selection": not bool(args.skip_stage_b_cross_candidate_selection),
            "body_dilate": args.stage_b_body_dilate,
            "min_residual_fraction": args.stage_b_min_residual_fraction,
            "min_fit_iou": args.stage_b_min_fit_iou,
            "min_motion_states": args.stage_b_min_motion_states,
            "motion_min_rotation_deg": args.stage_b_motion_min_rotation_deg,
            "motion_min_translation": args.stage_b_motion_min_translation,
            "geometry_refine_enabled": bool(args.enable_stage_b_geometry_refine),
        },
    }
    with open(os.path.join(stage_dir, "part_observations.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(stage_dir, "part_motion_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "part_id", "status", "successful_states", "median_fit_iou",
                "translation_span", "rotation_span_deg",
            ],
        )
        writer.writeheader()
        for row in part_summaries:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    # Context (numpy arrays, not JSON) handed to Stage C so it can grow each moving
    # candidate from the same residuals/tracks without re-fitting.
    context = {
        "residuals": residuals,
        "tracks": tracks,
        "part_summaries": stage_c_part_summaries,
        "observation_face_part": stage_c_observation_face_part,
        "motion_face_part": primary_motion_face_part,
        "motion_face_part_source": primary_motion_source,
        "geometry_face_part": _geometry_face_part,
        "residual_reports": residual_reports,
    }
    return payload, context


# ===========================================================================
# Stage C: solid part completion (grow the moving surface into a closed volume)
# ===========================================================================


def _mesh_components(mask_bool, faces, extra_edges=None):
    """Connected components of the True vertices over mesh edges (+ bridge edges)."""
    mask_bool = np.asarray(mask_bool, dtype=bool)
    uf = _UnionFind(mask_bool.shape[0])
    for face in faces:
        mv = [int(x) for x in face if mask_bool[int(x)]]
        for x in mv[1:]:
            uf.union(mv[0], x)
    if extra_edges is not None:
        for a, b in np.asarray(extra_edges, dtype=np.int64).reshape(-1, 2):
            if mask_bool[int(a)] and mask_bool[int(b)]:
                uf.union(int(a), int(b))
    comps: Dict[int, list] = {}
    for v in np.nonzero(mask_bool)[0]:
        comps.setdefault(uf.find(int(v)), []).append(int(v))
    return [np.asarray(idx, dtype=np.int64) for idx in comps.values()]


def _largest_component_mask(mask_bool, faces, extra_edges=None):
    """Keep only the largest connected component of a vertex mask."""
    out = np.zeros(np.asarray(mask_bool).shape[0], dtype=bool)
    comps = _mesh_components(mask_bool, faces, extra_edges)
    if comps:
        out[max(comps, key=lambda c: c.shape[0])] = True
    return out


def _component_bridge_edges(vertices, wid, n_welded, wf, tol):
    """Bridge edges between DISCONNECTED welded components within a spatial tolerance.

    Real OBJ exports are often bags of pieces (frame mouldings, backboard, stand,
    hinge, ...) that touch but share no topology, so edge-growth can never leave the
    piece the seed sits on and 'largest component' caps out at the biggest piece (the
    seed=97%-of-mesh but part=13%-of-mesh run). Welded vertices from different face
    components whose positions are within ``tol`` get an explicit bridge edge, so the
    grow/component logic sees one object while the static wall still blocks per
    vertex. Returns (edges[k,2] in welded ids, stats dict).
    """
    stats = {"welded_components": 0, "largest_component_welded": 0, "bridge_edges": 0, "bridged_components": 0}
    if n_welded == 0:
        return np.zeros((0, 2), dtype=np.int64), stats
    uf = _UnionFind(n_welded)
    for face in wf:
        uf.union(int(face[0]), int(face[1]))
        uf.union(int(face[1]), int(face[2]))
    comp = np.asarray([uf.find(i) for i in range(n_welded)], dtype=np.int64)
    _roots, counts = np.unique(comp, return_counts=True)
    stats["welded_components"] = int(_roots.shape[0])
    stats["largest_component_welded"] = int(counts.max())
    if tol <= 0:
        stats["bridged_components"] = stats["welded_components"]
        return np.zeros((0, 2), dtype=np.int64), stats
    first = np.zeros(n_welded, dtype=np.int64)
    first[wid[::-1]] = np.arange(wid.shape[0], dtype=np.int64)[::-1]  # earliest occurrence
    pos = np.asarray(vertices, dtype=np.float64)[first]
    cells = np.floor(pos / float(tol)).astype(np.int64)
    buckets: Dict[Tuple[int, int, int], list] = {}
    for i in range(n_welded):
        buckets.setdefault((int(cells[i, 0]), int(cells[i, 1]), int(cells[i, 2])), []).append(i)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    edges = []
    for i in range(n_welded):
        ci = cells[i]
        for off in offsets:
            for j in buckets.get((int(ci[0]) + off[0], int(ci[1]) + off[1], int(ci[2]) + off[2]), ()):
                if j <= i or comp[j] == comp[i]:
                    continue
                if float(np.linalg.norm(pos[i] - pos[j])) <= float(tol):
                    edges.append((i, j))
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    for a, b in edges:
        uf.union(int(a), int(b))
    stats["bridge_edges"] = int(edges.shape[0])
    stats["bridged_components"] = int(len({uf.find(i) for i in range(n_welded)}))
    return edges, stats


def _weld_ids(vertices, tol):
    """Map spatially-coincident vertices to a shared id.

    Meshes exported from Blender/OBJ are usually NOT vertex-welded (faces reference
    distinct indices for coincident corners), so face-index connectivity shatters the
    mesh into one component per triangle and any grow/component logic collapses to 3
    vertices. Welding by rounded position restores true connectivity for the graph
    ops without touching the geometry used for projection. Returns (weld_id, count).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    n = verts.shape[0]
    if tol <= 0 or n == 0:
        return np.arange(n, dtype=np.int64), n
    keys = np.round(verts / float(tol)).astype(np.int64)
    _uni, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv, dtype=np.int64).reshape(-1)
    return inv, int(inv.max()) + 1 if inv.size else 0


def _stage_c_state_trust(record, observation, residual_report, args):
    """Decide whether one body/part observation can enter Stage-C aggregation.

    A bad body pose makes ``object mask - body render`` nearly the whole object.
    The independently fit part pose can then scatter false motion advantage across
    the mesh. Reject that state before any per-vertex maxima or group votes exist.
    """
    reasons = []
    body_fit = record.get("fit_iou")
    part_fit = observation.get("fit_iou")
    min_body_fit = args.stage_c_min_body_fit_iou
    if min_body_fit is None:
        min_body_fit = args.min_fit_iou
    if body_fit is None:
        reasons.append("missing_body_fit")
    elif float(body_fit) < float(min_body_fit):
        reasons.append("low_body_fit")
    if part_fit is None:
        reasons.append("missing_part_fit")
    elif float(part_fit) < float(args.stage_c_min_part_fit_iou):
        reasons.append("low_part_fit")
    max_residual = float((residual_report or {}).get("max_residual_object_fraction", 0.0) or 0.0)
    if float(args.stage_c_max_residual_object_fraction) > 0.0 and max_residual > float(args.stage_c_max_residual_object_fraction):
        reasons.append("residual_too_large")
    return {
        "trusted": not reasons,
        "reason": ",".join(reasons) if reasons else "trusted",
        "body_fit_iou": None if body_fit is None else float(body_fit),
        "part_fit_iou": None if part_fit is None else float(part_fit),
        "max_residual_object_fraction": max_residual,
        "mean_residual_object_fraction": float((residual_report or {}).get("mean_residual_object_fraction", 0.0) or 0.0),
        "placement_group": str(record.get("placement_group") or observation.get("placement_group") or "default"),
    }


def _count_boundary_edges(faces_subset):
    """Edges used by exactly one face (open boundary).

    A thin closed part (front + back) has only the hinge cut seam here; an open
    shell (front faces only) has a large boundary. This is the shell-vs-solid metric.
    """
    if faces_subset is None or faces_subset.shape[0] == 0:
        return 0
    edges = np.sort(
        np.vstack([faces_subset[:, [0, 1]], faces_subset[:, [1, 2]], faces_subset[:, [0, 2]]]),
        axis=1,
    )
    _uni, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1))


def _stage_c_vertex_motion_evidence(states, reg, vertices, faces, observations, residuals, residual_reports, args):
    """Per-vertex explaining-away evidence over EVERY vertex (movable vs static).

    The plain object-mask advantage (part-pose support minus body-pose support) is
    depth-blind: a flat handle folded against the frame has its rest footprint INSIDE
    the frame silhouette, and its deployed footprint is inside the object mask too, so
    the advantage is 0 over the whole flap and only the protruding hinge scores > 0
    ("only the hinge got captured"). The discriminating question is not "inside the
    object mask?" but "does this pose explain pixels the BODY CANNOT explain?":
      - movable evidence  res_adv = residual support at the part pose minus residual
        support at the body pose, against the beyond-body residual (object mask minus
        the rendered body silhouette, from Stage B). A true handle vertex lands in the
        deployed-handle residual from the side/back/top views; a frame vertex cannot
        land in a residual it defines. Subtracting the body-pose term cancels the
        registration-rim mode (a barely-moved bad fit puts EVERY vertex on the rim
        residual at both poses -> res_adv ~ 0, not uniformly high).
      - static evidence  exit = object support at the body pose minus at the part
        pose. The part pose throws a far-from-axis body vertex into free space
        (exit >> 0); a true part vertex never exits (both footprints are inside).
    The two are combined PER STATE: a state contributes motion evidence for a vertex
    only when the vertex does NOT exit in that same state. A static vertex swung by
    part_T can graze the thin residual band in a few views by accident (and max-over-
    states hands it one lucky state), but in that very state it exits in most other
    views - counting hits without checking the misses inflated the seed to ~the whole
    mesh. Near the rotation axis motion vanishes, so BOTH scores are ~0 there for
    handle and frame alike - that band is intrinsically ambiguous to any image test
    and is left to the connectivity grow. Residual support uses ALL object-mask
    cameras in the denominator (zero mask where a camera has no residual), so a few
    residual cameras cannot inflate the score. State quality is gated before scoring,
    and motion is aggregated as independent placement-group votes rather than one
    lucky state winning through max-over-states.
    """
    n = vertices.shape[0]
    by_state = {st["state_id"]: st for st in states}
    max_adv = np.full(n, -np.inf, dtype=np.float64)
    max_exit = np.full(n, -np.inf, dtype=np.float64)
    max_res = np.zeros(n, dtype=np.float64)
    persist = np.zeros(n, dtype=np.float64)
    group_adv = {}
    group_exit = {}
    state_rows = []
    adv_gate = float(args.stage_c_persist_advantage)
    exit_cap = float(args.stage_c_motion_max_exit)
    dilate = int(args.stage_c_mask_dilate)
    min_obs = int(args.stage_c_min_observed_cameras)
    used_states = []
    for obs in observations:
        if obs.get("status") != "fit":
            continue
        sid = obs.get("state_id")
        if sid not in by_state:
            continue
        record = reg.get(sid) or {}
        trust = _stage_c_state_trust(record, obs, (residual_reports or {}).get(sid), args)
        row = {"state_id": sid, **trust}
        if not trust["trusted"]:
            state_rows.append(row)
            continue
        body_T = record.get("T_world_object")
        part_T = obs.get("T_world_part")
        if body_T is None or part_T is None:
            row.update({"trusted": False, "reason": "missing_pose"})
            state_rows.append(row)
            continue
        st = by_state[sid]
        cameras = rs._load_state_cameras(st["state_dir"])
        masks = rs._load_state_masks(st["mask_dir"])
        if not cameras or not masks:
            row.update({"trusted": False, "reason": "missing_camera_or_mask"})
            state_rows.append(row)
            continue
        # Residual masks over the FULL object-mask camera set: cameras whose residual
        # was empty/filtered contribute an all-zero mask so they still count in the
        # support denominator (a couple of residual cameras must not dominate).
        state_res = residuals.get(sid) or {}
        res_masks = {}
        for serial, obj_mask in masks.items():
            r = state_res.get(serial)
            if r is None:
                r = np.zeros(np.asarray(obj_mask).shape[:2], dtype=bool)
            res_masks[serial] = r
        body_world = rs._apply_transform(vertices, np.asarray(body_T, dtype=np.float64))
        part_world = rs._apply_transform(vertices, np.asarray(part_T, dtype=np.float64))
        body_sup, body_obs = _state_vertex_support(body_world, cameras, masks, dilate, min_obs)
        part_sup, part_obs = _state_vertex_support(part_world, cameras, masks, dilate, min_obs)
        body_res, _ = _state_vertex_support(body_world, cameras, res_masks, dilate, min_obs)
        part_res, _ = _state_vertex_support(part_world, cameras, res_masks, dilate, min_obs)
        valid = body_obs & part_obs  # both poses must be observed so the difference is meaningful
        if not np.any(valid):
            row.update({"trusted": False, "reason": "no_mutual_visibility"})
            state_rows.append(row)
            continue
        res_adv = part_res - body_res
        exit_pen = body_sup - part_sup
        gate_ok = valid & (exit_pen <= exit_cap)  # motion evidence only where the state does not exit
        state_motion = gate_ok & (res_adv >= adv_gate)
        motion_fraction = float(np.count_nonzero(state_motion)) / float(max(np.count_nonzero(valid), 1))
        row["motion_vertex_fraction"] = motion_fraction
        row["valid_vertices"] = int(np.count_nonzero(valid))
        row["motion_vertices"] = int(np.count_nonzero(state_motion))
        if float(args.stage_c_max_state_motion_fraction) > 0.0 and motion_fraction > float(args.stage_c_max_state_motion_fraction):
            row.update({"trusted": False, "reason": "motion_coverage_too_large"})
            state_rows.append(row)
            continue
        max_adv = np.maximum(max_adv, np.where(gate_ok, res_adv, -np.inf))
        max_exit[valid] = np.maximum(max_exit[valid], exit_pen[valid])
        max_res[valid] = np.maximum(max_res[valid], part_res[valid])
        persist[state_motion] += 1.0
        group = trust["placement_group"]
        if group not in group_adv:
            group_adv[group] = np.full(n, -np.inf, dtype=np.float64)
            group_exit[group] = np.full(n, -np.inf, dtype=np.float64)
        group_adv[group] = np.maximum(group_adv[group], np.where(gate_ok, res_adv, -np.inf))
        group_exit[group][valid] = np.maximum(group_exit[group][valid], exit_pen[valid])
        used_states.append(sid)
        state_rows.append(row)
    max_adv[~np.isfinite(max_adv)] = 0.0
    max_exit[~np.isfinite(max_exit)] = 0.0
    motion_groups = np.zeros(n, dtype=np.float64)
    static_groups = np.zeros(n, dtype=np.float64)
    group_rows = []
    for group in sorted(group_adv):
        advantage = group_adv[group]
        exit_score = group_exit[group]
        advantage[~np.isfinite(advantage)] = 0.0
        exit_score[~np.isfinite(exit_score)] = 0.0
        motion = advantage >= float(args.stage_c_persist_advantage)
        static = (
            (exit_score >= float(args.stage_c_static_exit))
            & (advantage < float(args.stage_c_eligible_advantage))
        )
        motion_groups += motion.astype(np.float64)
        static_groups += static.astype(np.float64)
        group_rows.append({
            "placement_group": group,
            "motion_vertices": int(np.count_nonzero(motion)),
            "static_vertices": int(np.count_nonzero(static)),
        })
    return (
        max_adv, max_exit, max_res, persist, motion_groups, static_groups,
        sorted(set(used_states)), state_rows, group_rows,
    )


def _stage_c_solidify_part(faces, wid, n_welded, bridges, max_adv, max_exit, persist, motion_groups, static_groups, fallback_seed, args):
    """Grow the confident moving seed through the connected moving sub-volume.

    All connectivity runs in WELDED index space (coincident vertices merged) so an
    unwelded OBJ does not shatter the mesh into per-triangle islands, and component
    BRIDGES let the grow cross between disconnected pieces (real exports are bags of
    pieces; without bridges the part caps out at whichever piece the seed sits on).
    Three vertex classes come from the explaining-away evidence:
      - movable (gated residual advantage recurring across placement groups): seed,
      - definitely static (strong object-mask EXIT at the part pose, no residual
        evidence): the part pose throws these into free space, so they are body,
      - ambiguous (both ~0): the near-axis band where motion vanishes; handle and
        frame vertices there are indistinguishable to ANY image test.
    The wall is the UNION of all definitely-static regions (not the largest
    component: a fragmented mesh has many static islands and every unwalled one
    would be grabbable). The seed grows through everything that is not the wall -
    crossing the ambiguous band and the bridges - interior holes close, and only the
    largest (bridged) component is kept. If nothing is definitely static (degenerate
    tiny motion) the wall falls back to the largest no-evidence component. Returns
    per-ORIGINAL-vertex (part, seed, static_core, eligible).
    """
    wf = wid[faces]

    def up_max(a):
        w = np.full(n_welded, -np.inf, dtype=np.float64)
        np.maximum.at(w, wid, np.asarray(a, dtype=np.float64))
        w[~np.isfinite(w)] = 0.0
        return w

    def up_any(b):
        w = np.zeros(n_welded, dtype=bool)
        if b is not None and np.any(b):
            w[wid[np.asarray(b, dtype=bool)]] = True
        return w

    wadv = up_max(max_adv)
    wexit = up_max(max_exit)
    wpersist = up_max(persist)
    wmotion_groups = up_max(motion_groups)
    wstatic_groups = up_max(static_groups)
    wshell = up_any(fallback_seed)

    has_motion = (
        (wadv >= float(args.stage_c_eligible_advantage))
        & (wmotion_groups >= float(args.stage_c_min_motion_groups))
    )
    # The Stage-B shell is a soft seed only. A broad Stage-B candidate must not
    # veto trusted static evidence and become an irreversible Stage-C label.
    definitely_static = (
        (wexit >= float(args.stage_c_static_exit))
        & (wstatic_groups >= float(args.stage_c_min_static_groups))
        & ~has_motion
    )
    if np.any(definitely_static):
        static_core_w = definitely_static
    else:
        static_core_w = _largest_component_mask(~has_motion, wf, bridges)
    eligible_w = ~static_core_w

    seed_w = (
        (wpersist >= float(args.stage_c_seed_min_states))
        & (wmotion_groups >= float(args.stage_c_min_motion_groups))
        & (wadv >= float(args.stage_c_seed_advantage))
    )
    seed_w = (seed_w | wshell) & eligible_w
    grown_w = _grow_movable(seed_w, eligible_w, wf, max_iters=int(args.stage_c_grow_iters), extra_edges=bridges)
    grown_w = _close_label_holes(grown_w, wf, float(args.stage_c_close_holes_frac))
    grown_w &= ~static_core_w
    part_w = _largest_component_mask(grown_w, wf, bridges)

    return part_w[wid], seed_w[wid], static_core_w[wid], eligible_w[wid]


def _run_stage_c_solid_parts(states, reg, vertices, faces, context, args, output_dir):
    """Complete each Stage-B moving candidate into a closed solid part submesh."""
    tracks = context.get("tracks") or {}
    part_summaries = context.get("part_summaries") or []
    observation_face_part = context.get("observation_face_part")
    motion_face_part = context.get("motion_face_part")
    motion_face_part_source = str(context.get("motion_face_part_source") or "stage_b_motion_refined")
    residuals = context.get("residuals") or {}
    residual_reports = context.get("residual_reports") or {}
    if not tracks:
        return {"enabled": True, "status": "no_residual_tracks", "part_count": 0, "parts": []}
    if motion_face_part_source == "no_rgb_validated_patch":
        return {
            "enabled": True,
            "status": "no_rgb_validated_patch",
            "part_count": 0,
            "parts": [],
            "reason": "Stage B did not find a patch with repeated RGB raster-change support.",
        }
    moving = [p for p in part_summaries if p.get("status") == "moving_candidate"]
    if not moving:
        return {"enabled": True, "status": "no_moving_candidate", "part_count": 0, "parts": []}

    stage_dir = os.path.join(output_dir, "segmentation", "stage_c")
    parts_dir = os.path.join(stage_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    # Welded connectivity once (unwelded OBJ would otherwise collapse the grow to one
    # triangle), then bridge disconnected pieces (bag-of-pieces exports) so grow and
    # component logic see one object.
    mesh_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    weld_tol = float(args.stage_c_weld_tol_frac) * (mesh_diag if mesh_diag > 0 else 1.0)
    wid, n_welded = _weld_ids(vertices, weld_tol)

    wf = wid[faces]
    bridge_tol = float(args.stage_c_bridge_dist_frac) * (mesh_diag if mesh_diag > 0 else 1.0)
    bridges, mesh_stats = _component_bridge_edges(vertices, wid, n_welded, wf, bridge_tol)
    mesh_stats["welded_vertices"] = int(n_welded)

    def welded_largest(mask):
        wm = np.zeros(n_welded, dtype=bool)
        if np.any(mask):
            wm[wid[np.asarray(mask, dtype=bool)]] = True
        wm = _largest_component_mask(wm, wf, bridges)
        return wm[wid]

    # Pass 1: solidify each moving candidate independently.
    candidates = []
    for summary in moving[: max(1, int(args.stage_c_max_candidates))]:
        src_id = int(summary.get("part_id"))
        observations = tracks.get(f"part_{src_id:02d}", [])
        max_adv, max_exit, max_res, persist, motion_groups, static_groups, used, state_rows, group_rows = _stage_c_vertex_motion_evidence(
            states, reg, vertices, faces, observations, residuals, residual_reports, args
        )
        fallback_seed = None
        seed_source = "none"
        if observation_face_part is not None and motion_face_part is not None and np.any(observation_face_part == src_id):
            candidate_vertices = np.zeros(vertices.shape[0], dtype=bool)
            candidate_vertices[np.unique(faces[observation_face_part == src_id])] = True
            motion_vertices = np.zeros(vertices.shape[0], dtype=bool)
            if np.any(motion_face_part >= 0):
                motion_vertices[np.unique(faces[motion_face_part >= 0])] = True
            # Motion-refined components are rebuilt and no longer preserve Stage-B
            # ids, so keep only their overlap with this source candidate.
            fallback_seed = candidate_vertices & motion_vertices
            if np.any(fallback_seed):
                seed_source = f"stage_b_{motion_face_part_source}_overlap"
            else:
                fallback_seed = None
        part_mask, seed_mask, static_core, eligible_mask = _stage_c_solidify_part(
            faces, wid, n_welded, bridges, max_adv, max_exit, persist,
            motion_groups, static_groups, fallback_seed, args
        )
        shell_faces = (
            faces[np.any(fallback_seed[faces], axis=1)]
            if fallback_seed is not None
            else np.zeros((0, 3), dtype=np.int64)
        )
        candidates.append({
            "src_id": src_id,
            "part_mask": part_mask,
            "seed_mask": seed_mask,
            "static_core": static_core,
            "shell_faces": shell_faces,
            "evidence": {
                "used_states": used,
                "state_diagnostics": state_rows,
                "group_diagnostics": group_rows,
                "seed_source": seed_source,
                "adv_p50": round(float(np.percentile(max_adv, 50)), 4),
                "adv_p90": round(float(np.percentile(max_adv, 90)), 4),
                "adv_p99": round(float(np.percentile(max_adv, 99)), 4),
                "adv_positive_vertices": int(np.count_nonzero(max_adv > 0.0)),
                "exit_p50": round(float(np.percentile(max_exit, 50)), 4),
                "exit_p90": round(float(np.percentile(max_exit, 90)), 4),
                "static_definite_vertices": int(
                    np.count_nonzero((max_exit >= float(args.stage_c_static_exit)) & (max_adv < float(args.stage_c_eligible_advantage)))
                ),
                "res_support_p90": round(float(np.percentile(max_res, 90)), 4),
                "persist_ge1_vertices": int(np.count_nonzero(persist >= 1.0)),
                "motion_groups_p50": round(float(np.percentile(motion_groups, 50)), 4),
                "motion_groups_p90": round(float(np.percentile(motion_groups, 90)), 4),
                "static_groups_p50": round(float(np.percentile(static_groups, 50)), 4),
                "eligible_vertices": int(np.count_nonzero(eligible_mask)),
                "seed_vertices": int(np.count_nonzero(seed_mask)),
                "static_core_vertices": int(np.count_nonzero(static_core)),
                "shell_vertices": int(np.count_nonzero(fallback_seed)) if fallback_seed is not None else 0,
            },
        })

    # Pass 2: claim largest-first (a smaller candidate cannot re-take the handle), then
    # drop parts far smaller than the biggest (leftover noise / a duplicate candidate of
    # the same physical part - the "weird small plane" the user saw).
    candidates.sort(key=lambda c: -int(np.count_nonzero(c["part_mask"])))
    claimed = np.zeros(vertices.shape[0], dtype=bool)
    for c in candidates:
        km = welded_largest(c["part_mask"] & ~claimed)
        c["kept_mask"] = km
        c["kept"] = int(np.count_nonzero(km))
        if c["kept"] > 0:
            claimed |= km
    max_kept = max((c["kept"] for c in candidates), default=0)
    size_floor = max(int(args.stage_c_min_part_vertices), int(float(args.stage_c_min_part_frac) * max_kept))

    solid_face_part = np.full(faces.shape[0], -1, dtype=np.int64)
    part_reports = []
    preview_meshes = []
    out_id = 0
    for c in candidates:
        if c["kept"] < size_floor:
            part_reports.append(
                {"source_part_id": c["src_id"], "status": "too_small", "solid_vertices": c["kept"],
                 "evidence": c["evidence"]}
            )
            continue
        part_mask = c["kept_mask"]
        face_hits = part_mask.astype(np.int64)[faces].sum(axis=1)
        pf_mask = face_hits >= 2
        solid_face_part[pf_mask] = out_id
        part_faces = faces[pf_mask]
        _write_obj_submesh(os.path.join(parts_dir, f"part_{out_id:02d}.obj"), vertices, part_faces)
        color = _PART_COLORS[out_id % len(_PART_COLORS)]
        preview = _preview_geometry(vertices, part_faces, color, int(args.viewer_max_faces))
        preview["name"] = f"stage_c part_{out_id:02d} (solid)"
        preview_meshes.append(preview)
        part_reports.append(
            {
                "source_part_id": c["src_id"],
                "part_id": out_id,
                "status": "solid",
                "seed_vertices": int(c["evidence"].get("seed_vertices", 0)),
                "static_core_vertices": int(np.count_nonzero(c["static_core"])),
                "solid_vertices": c["kept"],
                "solid_faces": int(part_faces.shape[0]),
                "shell_source_faces": int(c["shell_faces"].shape[0]),
                "shell_boundary_edges": _count_boundary_edges(wid[c["shell_faces"]]) if c["shell_faces"].shape[0] else 0,
                "solid_boundary_edges": _count_boundary_edges(wid[part_faces]),
                "evidence": c["evidence"],
            }
        )
        out_id += 1

    body_faces = faces[solid_face_part < 0]
    _write_obj_submesh(os.path.join(parts_dir, "body.obj"), vertices, body_faces)
    vertex_colors = np.array([_BODY_COLOR] * vertices.shape[0], dtype=np.int64)
    for pid in range(out_id):
        vertex_colors[np.unique(faces[solid_face_part == pid])] = _PART_COLORS[pid % len(_PART_COLORS)]
    labeled_path = os.path.join(stage_dir, "mesh_labeled_solid.ply")
    _write_colored_ply(labeled_path, vertices, faces, vertex_colors)
    combined_preview = _labeled_preview_geometry(vertices, faces, solid_face_part, int(args.viewer_max_faces))
    if combined_preview is not None:
        combined_preview["name"] = "stage C solid parts"

    payload = {
        "enabled": True,
        "status": "ok" if out_id > 0 else "no_solid_parts",
        "part_count": int(out_id),
        "parts": part_reports,
        "mesh": mesh_stats,
        "mesh_labeled_path": labeled_path,
        "params": {
            "weld_tol_frac": args.stage_c_weld_tol_frac,
            "bridge_dist_frac": args.stage_c_bridge_dist_frac,
            "min_body_fit_iou": args.stage_c_min_body_fit_iou,
            "min_part_fit_iou": args.stage_c_min_part_fit_iou,
            "max_residual_object_fraction": args.stage_c_max_residual_object_fraction,
            "max_state_motion_fraction": args.stage_c_max_state_motion_fraction,
            "min_motion_groups": args.stage_c_min_motion_groups,
            "min_static_groups": args.stage_c_min_static_groups,
            "eligible_advantage": args.stage_c_eligible_advantage,
            "persist_advantage": args.stage_c_persist_advantage,
            "seed_advantage": args.stage_c_seed_advantage,
            "seed_min_states": args.stage_c_seed_min_states,
            "static_exit": args.stage_c_static_exit,
            "motion_max_exit": args.stage_c_motion_max_exit,
            "min_part_frac": args.stage_c_min_part_frac,
        },
    }
    with open(os.path.join(stage_dir, "part_solids.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    payload["_preview_meshes"] = preview_meshes
    payload["_combined_preview"] = combined_preview
    return payload


def _print_stage_c_diagnostics(stage_c, seg_dir):
    if not stage_c or not bool(stage_c.get("enabled")):
        return
    status = stage_c.get("status")

    def _ev(row):
        e = row.get("evidence") or {}
        if not e:
            return ""
        return (
            f" res_adv(p50/p90/p99)={e.get('adv_p50')}/{e.get('adv_p90')}/{e.get('adv_p99')}"
            f" exit(p50/p90)={e.get('exit_p50')}/{e.get('exit_p90')}"
            f" persist>=1={e.get('persist_ge1_vertices')} eligible={e.get('eligible_vertices')}"
            f" shell={e.get('shell_vertices')} static_core={e.get('static_core_vertices')}"
            f" static_definite={e.get('static_definite_vertices')}"
        )

    def _print_state_rows(row):
        evidence = row.get("evidence") or {}
        for state in evidence.get("state_diagnostics", []) or []:
            body_fit = state.get("body_fit_iou")
            part_fit = state.get("part_fit_iou")
            motion = state.get("motion_vertex_fraction")
            body_text = "-" if body_fit is None else f"{float(body_fit):.3f}"
            part_text = "-" if part_fit is None else f"{float(part_fit):.3f}"
            motion_text = "-" if motion is None else f"{float(motion):.3f}"
            action = "used" if state.get("trusted") else f"dropped:{state.get('reason', '-')}"
            print(
                "[STAGEC:state] "
                f"src_{int(row.get('source_part_id', -1)):02d} state={state.get('state_id')} "
                f"group={state.get('placement_group', '-')} body_iou={body_text} part_iou={part_text} "
                f"res_obj={float(state.get('max_residual_object_fraction', 0.0) or 0.0):.3f} "
                f"motion={motion_text} {action}"
            )

    mesh = stage_c.get("mesh") or {}
    if mesh:
        print(
            "[STAGEC] mesh: welded_verts={0} pieces={1} largest_piece={2} bridge_edges={3} -> pieces_bridged={4}".format(
                mesh.get("welded_vertices"), mesh.get("welded_components"),
                mesh.get("largest_component_welded"), mesh.get("bridge_edges"),
                mesh.get("bridged_components"),
            )
        )
        if float((stage_c.get("params") or {}).get("bridge_dist_frac", 0.0) or 0.0) > 0.0 and int(mesh.get("bridged_components") or 1) > 1:
            print("[STAGEC] note: mesh is still {0} disconnected groups after bridging; if the part misses pieces, "
                  "raise --stage-c-bridge-dist-frac.".format(mesh.get("bridged_components")))
    if status != "ok":
        print(f"[STAGEC] status={status} (no solid parts) -> {os.path.join(seg_dir, 'stage_c')}")
        if status == "no_moving_candidate":
            print("[STAGEC] hint: Stage B found no moving_candidate; lower --stage-b-min-fit-iou / "
                  "--stage-b-motion-min-rotation-deg or check stage_b/part_observations.json.")
        elif status == "no_rgb_validated_patch":
            print("[STAGEC] hint: Stage B rejected every local patch under RGB raster-change validation; "
                  "inspect stage_b/image_motion_overlays and patch_refined/patch_refinement.json before relaxing thresholds.")
        elif status == "no_residual_tracks":
            print("[STAGEC] hint: no beyond-body residual to fit; check stage_b residual_overlays "
                  "(body silhouette may be covering the whole mask).")
        rows = stage_c.get("parts", []) or []
        for row in rows:
            print(
                f"[STAGEC] src part_{int(row.get('source_part_id', -1)):02d} status={row.get('status')} "
                f"solid_verts={int(row.get('solid_vertices', 0) or 0)} (min={_ev(row)})"
            )
            _print_state_rows(row)
        if rows and all(int((r.get("evidence") or {}).get("static_definite_vertices", 0) or 0) == 0 for r in rows):
            print("[STAGEC] hint: no vertex reached --stage-c-static-exit (definitely-static wall is a fallback); "
                  "part motion may be too small across the fitted states, or lower --stage-c-static-exit.")
        return
    print(f"[STAGEC] solid_parts={int(stage_c.get('part_count', 0) or 0)} -> {os.path.join(seg_dir, 'stage_c')}")
    for row in stage_c.get("parts", []) or []:
        if row.get("status") != "solid":
            print(f"[STAGEC] src part_{int(row.get('source_part_id', -1)):02d} status={row.get('status')} ({_ev(row)})")
            continue
        print(
            "[STAGEC] "
            f"part_{int(row.get('part_id', -1)):02d} <- src_{int(row.get('source_part_id', -1)):02d} "
            f"seed={int(row.get('seed_vertices', 0) or 0)} "
            f"solid_verts={int(row.get('solid_vertices', 0) or 0)} faces={int(row.get('solid_faces', 0) or 0)} "
            f"boundary_edges: shell={int(row.get('shell_boundary_edges', 0) or 0)} "
            f"-> solid={int(row.get('solid_boundary_edges', 0) or 0)}"
        )
        _print_state_rows(row)


# ===========================================================================
# Mesh export
# ===========================================================================


def _write_obj_submesh(path, vertices, faces_subset):
    used = np.unique(faces_subset)
    remap = {int(v): i + 1 for i, v in enumerate(used)}
    with open(path, "w", encoding="utf-8") as f:
        for vi in used:
            x, y, z = vertices[int(vi)]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for face in faces_subset:
            f.write(f"f {remap[int(face[0])]} {remap[int(face[1])]} {remap[int(face[2])]}\n")


def _write_colored_ply(path, vertices, faces, vertex_colors):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v, c in zip(vertices, vertex_colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _preview_geometry(vertices, faces_subset, color, max_faces):
    """Small, self-contained geometry payload for the HTML canvas viewers."""
    if faces_subset is None or faces_subset.shape[0] == 0:
        return {
            "vertices": [],
            "faces": [],
            "color": [int(color[0]), int(color[1]), int(color[2])],
            "source_faces": 0,
            "preview_faces": 0,
        }
    faces_preview = np.asarray(faces_subset, dtype=np.int64)
    max_faces = int(max_faces)
    if max_faces > 0 and faces_preview.shape[0] > max_faces:
        idx = np.linspace(0, faces_preview.shape[0] - 1, max_faces).astype(np.int64)
        faces_preview = faces_preview[idx]
    used = np.unique(faces_preview)
    remap = {int(v): i for i, v in enumerate(used)}
    local_faces = [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces_preview]
    local_vertices = [
        [round(float(vertices[int(vi), 0]), 6), round(float(vertices[int(vi), 1]), 6), round(float(vertices[int(vi), 2]), 6)]
        for vi in used
    ]
    return {
        "vertices": local_vertices,
        "faces": local_faces,
        "color": [int(color[0]), int(color[1]), int(color[2])],
        "source_faces": int(faces_subset.shape[0]),
        "preview_faces": int(faces_preview.shape[0]),
    }


def _labeled_preview_geometry(vertices, faces, face_part, max_faces):
    """Whole-mesh preview with per-face body/part colors."""
    if faces is None or faces.shape[0] == 0:
        return None
    faces_preview = np.asarray(faces, dtype=np.int64)
    face_part_preview = np.asarray(face_part, dtype=np.int64)
    max_faces = int(max_faces)
    if max_faces > 0 and faces_preview.shape[0] > max_faces:
        idx = np.linspace(0, faces_preview.shape[0] - 1, max_faces).astype(np.int64)
        faces_preview = faces_preview[idx]
        face_part_preview = face_part_preview[idx]
    used = np.unique(faces_preview)
    remap = {int(v): i for i, v in enumerate(used)}
    local_faces = [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces_preview]
    local_vertices = [
        [round(float(vertices[int(vi), 0]), 6), round(float(vertices[int(vi), 1]), 6), round(float(vertices[int(vi), 2]), 6)]
        for vi in used
    ]
    face_colors = []
    for part in face_part_preview:
        if int(part) < 0:
            color = _BODY_COLOR
        else:
            color = _PART_COLORS[int(part) % len(_PART_COLORS)]
        face_colors.append([int(color[0]), int(color[1]), int(color[2])])
    return {
        "name": "full labeled mesh",
        "vertices": local_vertices,
        "faces": local_faces,
        "color": [int(_BODY_COLOR[0]), int(_BODY_COLOR[1]), int(_BODY_COLOR[2])],
        "face_colors": face_colors,
        "source_faces": int(faces.shape[0]),
        "preview_faces": int(faces_preview.shape[0]),
    }


# ===========================================================================
# Debug overlays + HTML
# ===========================================================================


def _write_label_overlay(path, image_bgr, mask, world_vertices, cam, labels, part_of_vertex):
    import cv2

    height, width = mask.shape[:2]
    canvas = image_bgr.copy() if image_bgr is not None else np.zeros((height, width, 3), dtype=np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 255), 1)

    pixels = rs._project_points(cam["projection"], world_vertices)
    depths = rs._depths_in_camera(cam["cam_from_world"], world_vertices)
    u = np.rint(pixels[:, 0])
    v = np.rint(pixels[:, 1])
    onimg = np.isfinite(u) & np.isfinite(v) & np.isfinite(depths) & (depths > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    for i in np.nonzero(onimg)[0]:
        if labels[i] == 0:
            continue  # draw movable points only (body would clutter)
        part = int(part_of_vertex[i])
        color_rgb = _PART_COLORS[part % len(_PART_COLORS)] if part >= 0 else (200, 200, 200)
        canvas[int(v[i]), int(u[i])] = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))  # BGR
    cv2.imwrite(path, canvas)


def _write_segmentation_html(output_dir, mesh_path, parts_info, overlay_paths, summary, viewer_meshes=None, combined_mesh=None, viewer_sections=None):
    import html as html_lib

    esc = html_lib.escape

    def rel(path):
        try:
            return os.path.relpath(path, output_dir).replace(os.sep, "/")
        except Exception:
            return None

    parts = [
        "<style>",
        "body{font-family:system-ui,Arial,sans-serif;margin:16px;background:#f6f7f9;color:#1a1a1a}",
        "h1{font-size:20px}h2{font-size:16px;margin:16px 0 8px}",
        ".summary div{display:inline-block;background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px 14px;margin-right:12px}",
        "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:4px 10px;font-size:13px}",
        ".sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:middle}",
        ".viewer-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;margin:8px 0 14px}",
        ".viewer-card{background:#fff;border:1px solid #d5d8dc;border-radius:8px;padding:8px}",
        ".viewer-title{font-size:13px;font-weight:600;margin-bottom:4px;display:flex;justify-content:space-between;gap:8px}",
        ".viewer-meta{font-size:11px;color:#555;font-weight:400}",
        ".mesh-viewer{width:100%;height:260px;background:#ffffff;border:1px solid #ccd1d6;border-radius:4px;display:block;cursor:grab}",
        ".mesh-viewer.full{height:420px}",
        ".mesh-viewer:active{cursor:grabbing}",
        ".viewer-help{font-size:12px;color:#555;margin:4px 0 10px}",
        ".note{background:#fff8df;border:1px solid #e5c76a;border-radius:6px;padding:8px 10px;font-size:12px;margin:8px 0;color:#423800}",
        ".thumbs{display:flex;flex-wrap:wrap;gap:6px}.thumbs img{width:220px;border:1px solid #ccc;border-radius:4px}",
        "</style>",
        "<h1>Part segmentation (static / movable)</h1>",
        f"<div class='summary'><div><b>{summary['vertices']}</b> vertices</div>"
        f"<div><b>{summary['movable_vertices']}</b> movable</div>"
        f"<div><b>{summary['part_count']}</b> parts</div>"
        f"<div>states used: <b>{summary['used_states']}</b></div></div>",
        f"<div style='font-size:12px;color:#444'>mesh: {esc(mesh_path)}</div>",
        "<div class='note'><b>Read this first:</b> the 3D part viewers below show each part separately, per stage. "
        "Stage A = raw mask-consistency labels; Stage B patch-refined = the primary moving-surface shell; "
        "Stage C = the moving part completed into a closed solid (the mesh to use downstream).</div>",
    ]

    # Per-part 3D viewers (per stage), rendered first so the parts are the headline.
    viewer_payload = []
    if viewer_sections:
        parts.append("<h2>part 3D viewers (per stage)</h2>")
        parts.append("<div class='viewer-help'>drag = rotate, wheel = zoom. geometry is embedded in this HTML (works from a local file).</div>")
        for section in viewer_sections:
            meshes = section.get("meshes") or []
            if not meshes:
                continue
            parts.append(f"<h3 style='font-size:14px;margin:14px 0 4px'>{esc(str(section.get('title', '')))}</h3>")
            if section.get("help"):
                parts.append(f"<div class='viewer-help'>{esc(str(section.get('help')))}</div>")
            parts.append("<div class='viewer-grid'>")
            for mesh in meshes:
                mi = len(viewer_payload)
                viewer_payload.append(mesh)
                color = mesh.get("color", [170, 170, 170])
                sw = f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"
                parts.append(
                    f"<div class='viewer-card'><div class='viewer-title'>"
                    f"<span><span class='sw' style='background:{sw}'></span>{esc(str(mesh.get('name', mi)))}</span>"
                    f"<span class='viewer-meta'>{int(mesh.get('preview_faces', 0))}/{int(mesh.get('source_faces', 0))} faces</span>"
                    f"</div><canvas class='mesh-viewer' data-mesh-index='{mi}'></canvas></div>"
                )
            parts.append("</div>")

    parts.append("<h2>stage A parts</h2><table><tr><th>part</th><th>vertices</th><th>faces</th><th>mesh</th></tr>")
    for p in parts_info:
        color = _BODY_COLOR if p["part_id"] == "body" else _PART_COLORS[int(p["part_id"]) % len(_PART_COLORS)]
        sw = f"rgb({color[0]},{color[1]},{color[2]})"
        link = rel(p.get("mesh_path"))
        link_html = f"<a href='{esc(link)}'>obj</a>" if link else "-"
        parts.append(
            f"<tr><td><span class='sw' style='background:{sw}'></span>{esc(str(p['part_id']))}</td>"
            f"<td>{p['vertices']}</td><td>{p['faces']}</td><td>{link_html}</td></tr>"
        )
    parts.append("</table>")

    extra_viewer_meshes = []
    stage_b = summary.get("stage_b") or {}
    if stage_b:
        parts.append("<h2>stage B motion evidence and patch check</h2>")
        residual_source = str(stage_b.get("residual_source", "stage_a_body"))
        evidence_source = str(stage_b.get("evidence_source", "residual"))
        comparison_source = str(stage_b.get("comparison_residual_source", "-"))
        primary_dir = rel(stage_b.get("residual_overlay_dir"))
        comparison_dir = rel(stage_b.get("comparison_residual_overlay_dir"))
        source_text = (
            "the full known rest mesh" if residual_source == "full_mesh"
            else "the Stage-A body submesh"
        )
        parts.append(
            "<div class='viewer-help'>Residual = object mask minus "
            f"{esc(source_text)}. Stage-B fitting evidence is <code>{esc(evidence_source)}</code>: "
            "the default hybrid uses RGB change inside the dilated placement-group SAM support and only lets residual pixels join near that change. "
            "A missing SAM mask pixel is unknown, not negative; RGB change outside the group object support is retained only as a red diagnostic, never fit evidence.</div>"
        )
        image_motion_dir = rel(stage_b.get("image_motion_overlay_dir"))
        if image_motion_dir:
            parts.append(
                f"<div class='note'><b>Image motion:</b> <code>{esc(image_motion_dir)}</code>. "
                "Cyan = within-placement RGB change; yellow = SAM object-mask outline. Cyan need not lie inside yellow because a SAM3 miss is retained as usable positive evidence.</div>"
            )
        if primary_dir or comparison_dir:
            primary_text = esc(primary_dir or "-")
            comparison_text = esc(comparison_dir or "-")
            parts.append(
                "<div class='note'><b>Residual comparison:</b> primary "
                f"<code>{esc(residual_source)}</code> overlays: <code>{primary_text}</code>; "
                f"comparison <code>{esc(comparison_source)}</code> overlays: <code>{comparison_text}</code>. "
                "Magenta should isolate newly exposed moving geometry in the primary overlays; broad frame/interior magenta only in the comparison overlays identifies Stage-A body-hole leakage.</div>"
            )
        refinement = stage_b.get("refinement") or {}
        refined_mesh = rel(refinement.get("mesh_labeled_path"))
        if refined_mesh:
            parts.append(
                f"<div class='viewer-help'>Refined residual-support labels: "
                f"<a href='{esc(refined_mesh)}'>segmentation/stage_b/refined/mesh_labeled_refined.ply</a></div>"
            )
        motion_refinement = stage_b.get("motion_refinement") or {}
        motion_mesh = rel(motion_refinement.get("mesh_labeled_path"))
        if motion_mesh:
            parts.append(
                f"<div class='note'><b>Primary Stage B output:</b> motion-consistency refined labels are the current best signal "
                f"for separating moving surfaces from static false splits. "
                f"<a href='{esc(motion_mesh)}'>segmentation/stage_b/motion_refined/mesh_labeled_motion_refined.ply</a></div>"
            )
        if motion_refinement.get("preview_mesh"):
            extra_viewer_meshes.append(("primary motion-consistency refined mesh", motion_refinement["preview_mesh"]))
        if motion_refinement.get("parts"):
            parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>motion refined OBJ outputs</h3>")
            parts.append("<table><tr><th>part</th><th>faces</th><th>mesh</th></tr>")
            for p in motion_refinement.get("parts", []):
                link = rel(p.get("mesh_path"))
                link_html = f"<a href='{esc(link)}'>obj</a>" if link else "-"
                pid = p.get("part_id")
                label = "body" if pid == "body" else f"part_{int(pid):02d}"
                parts.append(
                    f"<tr><td>{esc(label)}</td><td>{int(p.get('faces', 0) or 0)}</td><td>{link_html}</td></tr>"
                )
            parts.append("</table>")
        patch_refinement = stage_b.get("patch_refinement") or {}
        patch_mesh = rel(patch_refinement.get("mesh_labeled_path"))
        if patch_mesh:
            primary_source = str(stage_b.get("primary_motion_source", "patch_refined"))
            parts.append(
                f"<div class='note'><b>Primary Stage B output:</b> local mesh patches are depth-composited with the fixed body at rest and fitted moving poses; only their front-most owner changes must align with RGB motion and recur across placement groups. "
                f"source=<code>{esc(primary_source)}</code>, evidence=<code>{esc(patch_refinement.get('evidence_source', '-'))}</code>, score=<code>{esc(patch_refinement.get('score_mode', '-'))}</code>. "
                f"<a href='{esc(patch_mesh)}'>segmentation/stage_b/patch_refined/mesh_labeled_patch_refined.ply</a></div>"
            )
        cross = patch_refinement.get("cross_candidate") or {}
        if cross:
            sources = ", ".join(f"part_{int(pid):02d}" for pid in cross.get("source_part_ids", []) or []) or "-"
            tracks = ", ".join(f"part_{int(pid):02d}" for pid in cross.get("track_part_ids", []) or []) or "-"
            selected_tracks = ", ".join(f"part_{int(pid):02d}" for pid in cross.get("selected_track_part_ids", []) or []) or "-"
            parts.append(
                "<div class='note'><b>Cross-candidate assignment:</b> source patches from "
                f"<code>{esc(sources)}</code> compete for reliable tracks <code>{esc(tracks)}</code> "
                f"(fit IoU &gt;= {float(cross.get('track_min_fit_iou', 0.0) or 0.0):.2f}, ratio &gt;= {float(cross.get('track_min_fit_ratio', 0.0) or 0.0):.2f}); selected rigid components use <code>{esc(selected_tracks)}</code>. "
                f"RGB coverage={float(cross.get('coverage', 0.0) or 0.0):.3f} "
                f"({int(cross.get('explained_pixels', 0) or 0)}/{int(cross.get('evidence_pixels', 0) or 0)} pixels, floor={float(cross.get('coverage_floor', 0.0) or 0.0):.3f}, pass={bool(cross.get('coverage_pass', False))}). "
                "Inspect the table below for <code>source part / patch -> track</code>; a handle patch assigned to the hinge track is the intended result.</div>"
            )
        if patch_refinement.get("preview_mesh"):
            extra_viewer_meshes.append(("primary visibility-aware patch-refined mesh", patch_refinement["preview_mesh"]))
        if patch_refinement.get("parts"):
            parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>patch refined OBJ outputs</h3>")
            parts.append("<table><tr><th>part</th><th>faces</th><th>mesh</th></tr>")
            for p in patch_refinement.get("parts", []):
                link = rel(p.get("mesh_path"))
                link_html = f"<a href='{esc(link)}'>obj</a>" if link else "-"
                pid = p.get("part_id")
                label = "body" if pid == "body" else f"part_{int(pid):02d}"
                parts.append(
                    f"<tr><td>{esc(label)}</td><td>{int(p.get('faces', 0) or 0)}</td><td>{link_html}</td></tr>"
                )
            parts.append("</table>")
        if cross:
            parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>cross-candidate patch assignments</h3>")
            parts.append("<table><tr><th>source part</th><th>patch</th><th>assigned track</th><th>decision</th><th>gain</th><th>faces</th><th>pooled precision</th><th>groups</th></tr>")
            cross_rows = [row for row in patch_refinement.get("patches", []) or [] if row.get("track_part_id") is not None]
            for row in sorted(cross_rows, key=lambda item: (str(item.get("status")) != "kept", int(item.get("source_part_id", -1)), int(item.get("patch_id", -1)), int(item.get("track_part_id", -1)))):
                gain = row.get("selection_gain")
                gain_cell = "-" if gain is None else f"{float(gain):.4f}"
                parts.append(
                    f"<tr><td>part_{int(row.get('source_part_id', -1)):02d}</td>"
                    f"<td>{int(row.get('patch_id', -1)):02d}</td>"
                    f"<td>part_{int(row.get('track_part_id', -1)):02d}</td>"
                    f"<td>{esc(str(row.get('status', '-')))}</td>"
                    f"<td>{gain_cell}</td>"
                    f"<td>{int(row.get('faces', 0) or 0)}</td>"
                    f"<td>{float(row.get('pooled_change_precision', 0.0) or 0.0):.3f}</td>"
                    f"<td>{int(row.get('motion_groups', 0) or 0)}</td></tr>"
                )
            parts.append("</table>")
            used_tracks = cross.get("used_track_diagnostics", []) or []
            if used_tracks:
                parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>selected cross-track pose audit</h3>")
                parts.append("<table><tr><th>track</th><th>actual median fit IoU</th><th>fit states</th><th>anchor rotation drift</th><th>anchor translation drift</th><th>anchor lock</th></tr>")
                for row in used_tracks:
                    drift = row.get("anchor_drift") or {}
                    rotation = drift.get("max_rotation_deg")
                    translation = drift.get("max_translation_fraction")
                    rotation_text = "-" if rotation is None else f"{float(rotation):.1f} deg"
                    translation_text = "-" if translation is None else f"{float(translation):.3f} diag"
                    parts.append(
                        f"<tr><td>part_{int(row.get('track_part_id', -1)):02d}</td>"
                        f"<td>{float(row.get('median_fit_iou', 0.0) or 0.0):.3f}</td>"
                        f"<td>{int(row.get('successful_states', 0) or 0)}</td>"
                        f"<td>{rotation_text}</td><td>{translation_text}</td>"
                        f"<td>{esc(str(drift.get('reason', '-')))}</td></tr>"
                    )
                parts.append("</table>")
            refit_attempts = [
                (round_info.get("round"), attempt)
                for round_info in cross.get("rounds", []) or []
                for attempt in round_info.get("refit_attempts", []) or []
            ]
            if refit_attempts:
                parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>cross refit attempts</h3>")
                parts.append("<table><tr><th>round</th><th>track</th><th>anchor patch</th><th>refit fit IoU</th><th>anchor rotation drift</th><th>anchor translation drift</th><th>decision</th></tr>")
                for round_index, attempt in refit_attempts:
                    actual_fit = (attempt.get("actual_fit") or {}).get("median_fit_iou")
                    drift = attempt.get("anchor_drift") or {}
                    rotation = drift.get("max_rotation_deg")
                    translation = drift.get("max_translation_fraction")
                    fit_text = "-" if actual_fit is None else f"{float(actual_fit):.3f}"
                    rotation_text = "-" if rotation is None else f"{float(rotation):.1f} deg"
                    translation_text = "-" if translation is None else f"{float(translation):.3f} diag"
                    parts.append(
                        f"<tr><td>{esc(str(round_index))}</td>"
                        f"<td>part_{int(attempt.get('track_part_id', -1)):02d}</td>"
                        f"<td>{esc(str(attempt.get('anchor_patch_key') or '-'))}</td>"
                        f"<td>{fit_text}</td><td>{rotation_text}</td><td>{translation_text}</td>"
                        f"<td>{esc(str(attempt.get('status', '-')))}</td></tr>"
                    )
                parts.append("</table>")
            probe_rows = cross.get("patch_track_probes", []) or []
            if probe_rows:
                parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>pre-cross patch pose probes</h3>")
                parts.append("<table><tr><th>source part</th><th>patch</th><th>faces</th><th>area</th><th>median fit IoU</th><th>status</th><th>anchor</th></tr>")
                for row in sorted(probe_rows, key=lambda item: (not bool(item.get("selected_as_anchor")), -float(item.get("median_fit_iou") or 0.0))):
                    parts.append(
                        f"<tr><td>part_{int(row.get('source_part_id', -1)):02d}</td>"
                        f"<td>{int(row.get('patch_id', -1)):02d}</td>"
                        f"<td>{int(row.get('faces', 0) or 0)}</td>"
                        f"<td>{float(row.get('area', 0.0) or 0.0):.6f}</td>"
                        f"<td>{float(row.get('median_fit_iou', 0.0) or 0.0):.3f}</td>"
                        f"<td>{esc(str(row.get('status', '-')))}</td>"
                        f"<td>{bool(row.get('selected_as_anchor', False))}</td></tr>"
                    )
                parts.append("</table>")
        geometry_refinement = stage_b.get("geometry_refinement") or {}
        geometry_mesh = rel(geometry_refinement.get("mesh_labeled_path"))
        if geometry_mesh:
            parts.append(
                f"<div class='viewer-help'>Experimental geometry-protrusion labels (not primary; useful only when slab protrusion matches the moving part): "
                f"<a href='{esc(geometry_mesh)}'>segmentation/stage_b/geometry_refined/mesh_labeled_geometry_refined.ply</a></div>"
            )
        if geometry_refinement.get("preview_mesh"):
            extra_viewer_meshes.append(("geometry protrusion refined mesh", geometry_refinement["preview_mesh"]))
        if geometry_refinement.get("parts"):
            parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>geometry refined OBJ outputs</h3>")
            parts.append("<table><tr><th>part</th><th>faces</th><th>mesh</th></tr>")
            for p in geometry_refinement.get("parts", []):
                link = rel(p.get("mesh_path"))
                link_html = f"<a href='{esc(link)}'>obj</a>" if link else "-"
                pid = p.get("part_id")
                label = "body" if pid == "body" else f"part_{int(pid):02d}"
                parts.append(
                    f"<tr><td>{esc(label)}</td><td>{int(p.get('faces', 0) or 0)}</td><td>{link_html}</td></tr>"
                )
            parts.append("</table>")
        parts.append("<table><tr><th>part</th><th>status</th><th>fit states</th><th>median fit IoU</th><th>translation span</th><th>rotation span</th></tr>")
        for p in stage_b.get("parts", []):
            part_id = int(p.get("part_id", -1))
            color = _PART_COLORS[part_id % len(_PART_COLORS)] if part_id >= 0 else _BODY_COLOR
            sw = f"rgb({color[0]},{color[1]},{color[2]})"
            fit = p.get("median_fit_iou")
            fit_text = "-" if fit is None else f"{float(fit):.3f}"
            parts.append(
                f"<tr><td><span class='sw' style='background:{sw}'></span>part_{part_id:02d}</td>"
                f"<td>{esc(str(p.get('status', '-')))}</td>"
                f"<td>{int(p.get('successful_states', 0))}</td>"
                f"<td>{fit_text}</td>"
                f"<td>{float(p.get('translation_span', 0.0)):.4f}</td>"
                f"<td>{float(p.get('rotation_span_deg', 0.0)):.1f} deg</td></tr>"
            )
        parts.append("</table>")
        parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>refinement diagnostics</h3>")
        parts.append("<table><tr><th>stage</th><th>status</th><th>source vertices</th><th>seed / eligible</th><th>kept vertices</th><th>parts</th><th>notes</th></tr>")

        def _diag_row(name, data, source_key, seed_key, eligible_key, kept_key, part_key, note):
            if not data:
                parts.append(f"<tr><td>{esc(name)}</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>{esc(note)}</td></tr>")
                return
            seed = data.get(seed_key)
            eligible = data.get(eligible_key)
            seed_text = "-"
            if seed is not None or eligible is not None:
                seed_text = f"{int(seed or 0)} / {int(eligible or 0)}"
            parts.append(
                f"<tr><td>{esc(name)}</td><td>{esc(str(data.get('status', 'ok')))}</td>"
                f"<td>{int(data.get(source_key, 0) or 0)}</td><td>{seed_text}</td>"
                f"<td>{int(data.get(kept_key, 0) or 0)}</td><td>{int(data.get(part_key, 0) or 0)}</td>"
                f"<td>{note}</td></tr>"
            )

        _diag_row(
            "residual support",
            refinement,
            "original_movable_vertices",
            "residual_seed_vertices",
            "residual_eligible_vertices",
            "refined_movable_vertices",
            "refined_part_count",
            "support drop inside residual masks",
        )
        _diag_row(
            "motion consistency",
            motion_refinement,
            "source_movable_vertices",
            None,
            None,
            "kept_vertices",
            "part_count",
            "moving-pose support over body-pose support",
        )
        _diag_row(
            "visibility-aware patches",
            patch_refinement,
            "source_candidate_faces",
            None,
            None,
            "kept_faces",
            "part_count",
            "local welded patches, front/projected face support, cross-group positive evidence",
        )
        slab = geometry_refinement.get("body_slab") or {}
        slab_note = (
            f"slab_half={float(slab.get('half_thickness', 0.0)):.4f}, "
            f"seed_thr={float(slab.get('seed_threshold', 0.0)):.4f}, "
            f"grow_thr={float(slab.get('grow_threshold', 0.0)):.4f}"
        ) if slab else "body slab/protrusion prior"
        _diag_row(
            "geometry protrusion",
            geometry_refinement,
            "source_candidate_vertices",
            "seed_vertices",
            "eligible_vertices",
            "kept_vertices",
            "part_count",
            slab_note,
        )
        parts.append("</table>")
        if geometry_refinement.get("used_parts") or geometry_refinement.get("skipped_parts"):
            parts.append("<h3 style='font-size:14px;margin:12px 0 6px'>geometry candidate filtering</h3>")
            parts.append("<table><tr><th>part</th><th>decision</th><th>fit IoU</th><th>status/reason</th><th>vertices</th></tr>")
            for row in geometry_refinement.get("used_parts", []):
                parts.append(
                    f"<tr><td>part_{int(row.get('part_id', -1)):02d}</td><td>used</td>"
                    f"<td>{float(row.get('median_fit_iou') or 0.0):.3f}</td>"
                    f"<td>{esc(str(row.get('status', '-')))}</td>"
                    f"<td>{int(row.get('source_vertices', 0) or 0)}</td></tr>"
                )
            for row in geometry_refinement.get("skipped_parts", []):
                fit = row.get("median_fit_iou")
                fit_text = "-" if fit is None else f"{float(fit):.3f}"
                parts.append(
                    f"<tr><td>part_{int(row.get('part_id', -1)):02d}</td><td>skipped</td>"
                    f"<td>{fit_text}</td><td>{esc(str(row.get('reason', '-')))} / {esc(str(row.get('status', '-')))}</td>"
                    "<td>-</td></tr>"
                )
            parts.append("</table>")
        parts.append("<div class='viewer-help'>Residual overlays: magenta = object-mask area not explained by the primary rendered mesh; yellow = SAM object mask; grey = primary rendered mesh. Image-motion overlays: cyan = accepted RGB change inside dilated placement-group object support; red = detached RGB change rejected as background; yellow contour = current SAM object mask.</div>")

    if combined_mesh:
        ci = len(viewer_payload)
        viewer_payload.append(combined_mesh)
        parts.append("<h2>full labeled mesh viewer (Stage A)</h2>")
        parts.append("<div class='viewer-help'>grey = body, colored faces = movable part labels. drag = rotate, wheel = zoom.</div>")
        parts.append("<div class='viewer-card'>")
        parts.append(
            f"<div class='viewer-title'><span>full labeled mesh</span>"
            f"<span class='viewer-meta'>{int(combined_mesh.get('preview_faces', 0))}/{int(combined_mesh.get('source_faces', 0))} faces</span></div>"
            f"<canvas class='mesh-viewer full' data-mesh-index='{ci}'></canvas>"
        )
        parts.append("</div>")

    if extra_viewer_meshes:
        parts.append("<h2>refined mesh viewers</h2>")
        parts.append("<div class='viewer-grid'>")
        for title, mesh in extra_viewer_meshes:
            mesh_index = len(viewer_payload)
            viewer_payload.append(mesh)
            parts.append(
                f"<div class='viewer-card'><div class='viewer-title'>"
                f"<span>{esc(str(title))}</span>"
                f"<span class='viewer-meta'>{int(mesh.get('preview_faces', 0))}/{int(mesh.get('source_faces', 0))} faces</span>"
                f"</div><canvas class='mesh-viewer full' data-mesh-index='{mesh_index}'></canvas></div>"
            )
        parts.append("</div>")

    if viewer_meshes and not viewer_sections:
        parts.append("<h2>part mesh viewers</h2>")
        parts.append("<div class='viewer-help'>drag = rotate, wheel = zoom. Geometry is embedded in this HTML, so it works from a local file.</div>")
        parts.append("<div class='viewer-grid'>")
        for idx, mesh in enumerate(viewer_meshes):
            mesh_index = len(viewer_payload)
            viewer_payload.append(mesh)
            color = mesh.get("color", [170, 170, 170])
            sw = f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"
            parts.append(
                f"<div class='viewer-card'><div class='viewer-title'>"
                f"<span><span class='sw' style='background:{sw}'></span>{esc(str(mesh.get('name', idx)))}</span>"
                f"<span class='viewer-meta'>{int(mesh.get('preview_faces', 0))}/{int(mesh.get('source_faces', 0))} faces</span>"
                f"</div><canvas class='mesh-viewer' data-mesh-index='{mesh_index}'></canvas></div>"
            )
        parts.append("</div>")

    colored = rel(summary.get("labeled_mesh_path"))
    if colored:
        parts.append(f"<h2>labeled mesh</h2><div style='font-size:13px'>{esc(colored)} "
                     "(body grey, movable colored; open in a PLY viewer)</div>")

    if overlay_paths:
        parts.append("<h2>movable-vertex projections (yellow = mask)</h2><div class='thumbs'>")
        for path in overlay_paths:
            r = rel(path)
            if r:
                parts.append(f"<a href='{esc(r)}'><img src='{esc(r)}' loading='lazy'></a>")
        parts.append("</div>")

    report_path = os.path.join(output_dir, "segmentation_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        viewer_json = json.dumps(viewer_payload, ensure_ascii=True, separators=(",", ":"))
        viewer_script = r"""
<script>
(function(){
  const meshes = JSON.parse(document.getElementById("mesh-preview-data").textContent || "[]");
  function shade(color, factor){
    return "rgb(" + color.map(v => Math.max(0, Math.min(255, Math.round(v * factor)))).join(",") + ")";
  }
  function makeViewer(canvas, mesh){
    const ctx = canvas.getContext("2d");
    const verts = mesh.vertices || [];
    const faces = mesh.faces || [];
    const color = mesh.color || [170, 170, 170];
    const faceColors = mesh.face_colors || null;
    let rx = -0.55, ry = 0.75, rz = 0.0, zoom = 1.0;
    let dragging = false, lastX = 0, lastY = 0;
    const center = [0, 0, 0];
    let radius = 1;
    if (verts.length) {
      const mn = verts[0].slice(), mx = verts[0].slice();
      for (const v of verts) for (let i = 0; i < 3; i++) { mn[i] = Math.min(mn[i], v[i]); mx[i] = Math.max(mx[i], v[i]); }
      for (let i = 0; i < 3; i++) center[i] = 0.5 * (mn[i] + mx[i]);
      radius = Math.max(1e-9, Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) * 0.5);
    }
    function transform(v){
      let x = (v[0] - center[0]) / radius;
      let y = (v[1] - center[1]) / radius;
      let z = (v[2] - center[2]) / radius;
      const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry), cz = Math.cos(rz), sz = Math.sin(rz);
      let y1 = y * cx - z * sx, z1 = y * sx + z * cx; y = y1; z = z1;
      let x2 = x * cy + z * sy, z2 = -x * sy + z * cy; x = x2; z = z2;
      let x3 = x * cz - y * sz, y3 = x * sz + y * cz; x = x3; y = y3;
      return [x, y, z];
    }
    function resize(){
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width * dpr));
      const h = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    }
    function render(){
      resize();
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, w, h);
      if (!verts.length || !faces.length) {
        ctx.fillStyle = "#777";
        ctx.font = Math.max(12, Math.floor(w / 24)) + "px sans-serif";
        ctx.fillText("empty mesh", 16, 28);
        return;
      }
      const scale = Math.min(w, h) * 0.42 * zoom;
      const tv = verts.map(transform);
      const pv = tv.map(p => [w * 0.5 + p[0] * scale, h * 0.53 - p[1] * scale, p[2]]);
      const drawFaces = faces.map((f, faceIndex) => {
        const a = tv[f[0]], b = tv[f[1]], c = tv[f[2]];
        const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const light = Math.max(0.25, Math.min(1.15, 0.62 + 0.38 * (nz / (Math.hypot(nx, ny, nz) || 1))));
        const baseColor = faceColors && faceColors[faceIndex] ? faceColors[faceIndex] : color;
        return {f, z:(pv[f[0]][2] + pv[f[1]][2] + pv[f[2]][2]) / 3, light, color:baseColor};
      }).sort((a,b) => a.z - b.z);
      ctx.lineWidth = Math.max(0.6, (window.devicePixelRatio || 1) * 0.6);
      for (const item of drawFaces) {
        const f = item.f, p0 = pv[f[0]], p1 = pv[f[1]], p2 = pv[f[2]];
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.closePath();
        ctx.fillStyle = shade(item.color, item.light);
        ctx.strokeStyle = "rgba(35,35,35,0.20)";
        ctx.fill(); ctx.stroke();
      }
    }
    canvas.addEventListener("mousedown", ev => { dragging = true; lastX = ev.clientX; lastY = ev.clientY; });
    window.addEventListener("mouseup", () => { dragging = false; });
    canvas.addEventListener("mousemove", ev => {
      if (!dragging) return;
      ry += (ev.clientX - lastX) * 0.01;
      rx += (ev.clientY - lastY) * 0.01;
      lastX = ev.clientX; lastY = ev.clientY;
      render();
    });
    canvas.addEventListener("wheel", ev => {
      ev.preventDefault();
      zoom *= Math.exp(-ev.deltaY * 0.001);
      zoom = Math.max(0.25, Math.min(5.0, zoom));
      render();
    }, {passive:false});
    window.addEventListener("resize", render);
    render();
  }
  document.querySelectorAll(".mesh-viewer").forEach(canvas => {
    const idx = Number(canvas.getAttribute("data-mesh-index"));
    makeViewer(canvas, meshes[idx] || {});
  });
})();
</script>
"""
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Part segmentation</title></head><body>" + "".join(parts)
                + f"<script id='mesh-preview-data' type='application/json'>{viewer_json}</script>"
                + viewer_script + "</body></html>")
    return report_path


def _print_stage_b_diagnostics(stage_b, seg_dir):
    if not stage_b or not bool(stage_b.get("enabled")):
        return
    parts_count = len(stage_b.get("parts", []))
    moving_count = sum(1 for p in stage_b.get("parts", []) if p.get("status") == "moving_candidate")
    print(f"[STAGEB] parts={parts_count} moving_candidates={moving_count} -> {os.path.join(seg_dir, 'stage_b')}")
    print(
        "[STAGEB:residual] "
        f"primary={stage_b.get('residual_source', '-')} "
        f"overlays={stage_b.get('residual_overlay_dir', '-')} "
        f"comparison={stage_b.get('comparison_residual_source', '-')} "
        f"comparison_overlays={stage_b.get('comparison_residual_overlay_dir', '-')}"
    )
    print(
        "[STAGEB:evidence] "
        f"source={stage_b.get('evidence_source', '-')} "
        f"states={int(stage_b.get('evidence_state_count', 0) or 0)} "
        f"rgb_states={int(stage_b.get('rgb_motion_evidence_state_count', 0) or 0)} "
        f"image_motion_overlays={stage_b.get('image_motion_overlay_dir', '-')}"
    )

    refinement = stage_b.get("refinement") or {}
    if refinement:
        print(
            "[STAGEB:residual_refine] "
            f"status={refinement.get('status', 'ok')} "
            f"seed={int(refinement.get('residual_seed_vertices', 0) or 0)} "
            f"eligible={int(refinement.get('residual_eligible_vertices', 0) or 0)} "
            f"kept={int(refinement.get('refined_movable_vertices', 0) or 0)}/"
            f"{int(refinement.get('original_movable_vertices', 0) or 0)} "
            f"parts={int(refinement.get('refined_part_count', 0) or 0)}"
        )

    motion_refinement = stage_b.get("motion_refinement") or {}
    if motion_refinement:
        print(
            "[STAGEB:motion_refine] "
            f"kept={int(motion_refinement.get('kept_vertices', 0) or 0)}/"
            f"{int(motion_refinement.get('source_movable_vertices', 0) or 0)} "
            f"parts={int(motion_refinement.get('part_count', 0) or 0)}"
        )

    patch_refinement = stage_b.get("patch_refinement") or {}
    if patch_refinement:
        print(
            "[STAGEB:patch_refine] "
            f"status={patch_refinement.get('status', 'ok')} "
            f"kept_faces={int(patch_refinement.get('kept_faces', 0) or 0)}/"
            f"{int(patch_refinement.get('source_candidate_faces', 0) or 0)} "
            f"parts={int(patch_refinement.get('part_count', 0) or 0)} "
            f"evidence={patch_refinement.get('evidence_source', '-')} "
            f"score={patch_refinement.get('score_mode', '-')} "
            f"backend={patch_refinement.get('depth_raster_backend') or '-'} "
            f"primary={stage_b.get('primary_motion_source', '-')}"
        )
        joint = patch_refinement.get("joint") or {}
        faces_by_patch = {}
        for row in patch_refinement.get("patches", []) or []:
            if row.get("patch_id") is not None:
                faces_by_patch[(row.get("source_part_id"), int(row["patch_id"]))] = row
        for key in sorted(joint):
            info = joint[key] or {}
            for rnd in info.get("rounds", []) or []:
                fit = rnd.get("median_fit_iou")
                fit_text = "-" if fit is None else f"{float(fit):.4f}"
                subset_fit = rnd.get("median_fit_iou_subset")
                subset_text = "-" if subset_fit is None else f"{float(subset_fit):.4f}"
                print(
                    "[STAGEB:joint] "
                    f"{key} round={rnd.get('round')} selected={rnd.get('selected')} kept={rnd.get('kept')} "
                    f"explained={rnd.get('explained_pixels')} spurious={rnd.get('spurious_pixels')} "
                    f"score={float(rnd.get('score') or 0.0):.1f} median_fit={fit_text} subset_fit={subset_text}"
                )
            print(
                "[STAGEB:joint] "
                f"{key} used_round={info.get('used_round', '-')} "
                f"refit_rejected={info.get('refit_rejected', False)} "
                f"stop={info.get('stop_reason', '-')} "
                f"patches_dir={info.get('patches_dir', '-')} "
                f"overlays={info.get('overlay_dir', '-')}"
            )
            part_id = info.get("part_id")
            for pid in info.get("kept_patches", []) or []:
                row = faces_by_patch.get((part_id, int(pid))) or {}
                print(
                    "[STAGEB:joint] "
                    f"{key} kept patch={int(pid):02d} faces={row.get('faces', '-')} "
                    f"gain={row.get('selection_gain')} pooled={float(row.get('pooled_change_precision') or 0.0):.3f} "
                    f"groups={row.get('motion_groups', '-')}"
                )
        cross = patch_refinement.get("cross_candidate") or {}
        if cross:
            print(
                "[STAGEB:cross] "
                f"sources={cross.get('source_part_ids', [])} "
                f"tracks={cross.get('track_part_ids', [])} "
                f"selected_tracks={cross.get('selected_track_part_ids', [])} "
                f"min_track_fit={float(cross.get('track_min_fit_iou', 0.0) or 0.0):.3f} "
                f"patches={int(cross.get('patch_count', 0) or 0)} "
                f"hypotheses={int(cross.get('hypothesis_count', 0) or 0)} "
                f"round={cross.get('used_round', '-')} "
                f"score={float(cross.get('normalized_score', 0.0) or 0.0):.4f} "
                f"explained={int(cross.get('explained_pixels', 0) or 0)} "
                f"unexplained={int(cross.get('unexplained_pixels', 0) or 0)} "
                f"coverage={float(cross.get('coverage', 0.0) or 0.0):.3f} "
                f"coverage_pass={bool(cross.get('coverage_pass', False))} "
                f"spurious={int(cross.get('spurious_pixels', 0) or 0)}"
            )
            for probe in cross.get("patch_track_probes", []) or []:
                if not probe.get("selected_as_anchor"):
                    continue
                print(
                    "[STAGEB:cross] "
                    f"probe src_{int(probe.get('source_part_id', -1)):02d} patch_{int(probe.get('patch_id', -1)):02d} "
                    f"fit={float(probe.get('median_fit_iou', 0.0) or 0.0):.3f} selected_as_anchor=True"
                )
            for track in cross.get("used_track_diagnostics", []) or []:
                fit = track.get("median_fit_iou")
                fit_text = "-" if fit is None else f"{float(fit):.3f}"
                drift = track.get("anchor_drift") or {}
                if drift:
                    print(
                        "[STAGEB:cross] "
                        f"used track_{int(track.get('track_part_id', -1)):02d} actual_fit={fit_text} "
                        f"anchor_drift={float(drift.get('max_rotation_deg', 0.0) or 0.0):.1f}deg/"
                        f"{float(drift.get('max_translation_fraction', 0.0) or 0.0):.3f}diag "
                        f"anchor_lock={drift.get('passes', False)}"
                    )
            for row in patch_refinement.get("patches", []) or []:
                if row.get("status") != "kept" or row.get("track_part_id") is None:
                    continue
                print(
                    "[STAGEB:cross] "
                    f"src_{int(row.get('source_part_id', -1)):02d} patch_{int(row.get('patch_id', -1)):02d} "
                    f"-> track_{int(row.get('track_part_id', -1)):02d} "
                    f"faces={int(row.get('faces', 0) or 0)} "
                    f"gain={row.get('selection_gain')} "
                    f"pooled={float(row.get('pooled_change_precision', 0.0) or 0.0):.3f} "
                    f"groups={int(row.get('motion_groups', 0) or 0)}"
                )

    geometry = stage_b.get("geometry_refinement") or {}
    if geometry:
        slab = geometry.get("body_slab") or {}
        print(
            "[STAGEB:geometry_refine] "
            f"status={geometry.get('status', 'ok')} "
            f"used_parts={len(geometry.get('used_parts', []) or [])} "
            f"skipped_parts={len(geometry.get('skipped_parts', []) or [])} "
            f"seed={int(geometry.get('seed_vertices', 0) or 0)} "
            f"eligible={int(geometry.get('eligible_vertices', 0) or 0)} "
            f"kept={int(geometry.get('kept_vertices', 0) or 0)}/"
            f"{int(geometry.get('source_candidate_vertices', 0) or 0)} "
            f"parts={int(geometry.get('part_count', 0) or 0)}"
        )
        if slab:
            print(
                "[STAGEB:geometry_refine] "
                f"slab_half={float(slab.get('half_thickness', 0.0)):.6f} "
                f"seed_threshold={float(slab.get('seed_threshold', 0.0)):.6f} "
                f"grow_threshold={float(slab.get('grow_threshold', 0.0)):.6f}"
            )
        for row in geometry.get("used_parts", []) or []:
            print(
                "[STAGEB:geometry_refine] "
                f"used part_{int(row.get('part_id', -1)):02d} "
                f"fit={float(row.get('median_fit_iou') or 0.0):.3f} "
                f"status={row.get('status', '-')}"
            )
        for row in geometry.get("skipped_parts", []) or []:
            fit = row.get("median_fit_iou")
            fit_text = "-" if fit is None else f"{float(fit):.3f}"
            print(
                "[STAGEB:geometry_refine] "
                f"skipped part_{int(row.get('part_id', -1)):02d} "
                f"fit={fit_text} reason={row.get('reason', '-')} status={row.get('status', '-')}"
            )
    elif not bool((stage_b.get("params") or {}).get("geometry_refine_enabled", False)):
        print("[STAGEB:geometry_refine] disabled (experimental; enable with --enable-stage-b-geometry-refine)")


# ===========================================================================
# Main
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static/movable part segmentation from cross-state multiview mask consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session-dir", required=True, help="Capture session dir with per-state cam_param.")
    parser.add_argument("--mesh-path", required=True, help="Known metric mesh (.obj) that was registered.")
    parser.add_argument("--masks-dir", default=None, help="Object-mask root. Default: <session>/processed/object_masks.")
    parser.add_argument("--output-dir", default=None, help="Pipeline output root (also where registration lives). Default: <session>/processed.")
    parser.add_argument("--image-dirname", default="images", help="Per-state undistorted image folder (for overlays).")
    parser.add_argument("--min-fit-iou", type=float, default=0.5, help="Skip states whose registration fit_iou is below this.")
    parser.add_argument("--mask-dilate", type=int, default=2, help="Mask dilation (px) tolerance for the inside test (scale with image resolution; too large washes out the movable signal).")
    parser.add_argument("--min-observed-cameras", type=int, default=3, help="Cameras that must see a vertex for a state to count it.")
    parser.add_argument("--min-observed-states", type=int, default=2, help="States that must observe a vertex to label it.")
    parser.add_argument("--support-drop", type=float, default=0.35, help="Confident-seed threshold: support drop (max-min across states).")
    parser.add_argument("--grow-support-drop", type=float, default=0.15, help="Region-grow threshold toward the joint (weaker drop; body has ~0 drop).")
    parser.add_argument("--present-support", type=float, default=0.6, help="A movable vertex must reach >= this support in some state.")
    parser.add_argument("--close-holes-frac", type=float, default=0.6, help="Fill an interior non-movable vertex whose movable-neighbor fraction is >= this.")
    parser.add_argument("--min-part-vertices", type=int, default=20, help="Movable components smaller than this are merged back into the body.")
    parser.add_argument("--merge-dist-frac", type=float, default=0.05, help="Merge movable components within this fraction of the mesh diagonal (one part split by noise).")
    parser.add_argument("--overlay-states", type=int, default=3, help="States to render movable-vertex overlays for.")
    parser.add_argument("--overlay-cameras", type=int, default=3, help="Cameras per overlay state.")
    parser.add_argument("--viewer-max-faces", type=int, default=12000, help="Max faces embedded per part in segmentation_report.html viewers (0 = all).")
    parser.add_argument("--skip-stage-b", action="store_true", help="Skip beyond-body residual part observation scoring.")
    parser.add_argument("--stage-b-residual-source", choices=("full_mesh", "stage_a_body"), default="full_mesh", help="Primary Stage-B residual. full_mesh is conservative and avoids Stage-A false body holes; stage_a_body is the legacy refinement residual.")
    parser.add_argument("--stage-b-body-dilate", type=int, default=2, help="Dilate rendered body silhouette before subtracting it from the object mask.")
    parser.add_argument("--stage-b-min-residual-fraction", type=float, default=0.0005, help="Minimum residual mask image fraction for a camera to be used.")
    parser.add_argument("--stage-b-min-component-area", type=int, default=40, help="Drop residual connected components smaller than this many pixels.")
    parser.add_argument("--stage-b-overlay-cameras", type=int, default=3, help="Residual overlay cameras per state.")
    parser.add_argument("--stage-b-evidence-source", choices=("hybrid", "image_motion", "residual"), default="hybrid", help="Positive evidence fitted by Stage B. hybrid keeps RGB motion and admits residual only near it; image_motion ignores silhouette residual; residual reproduces the prior behavior.")
    parser.add_argument("--stage-b-image-motion-threshold", type=float, default=18.0, help="Per-channel grayscale difference from a within-placement median image that counts as positive motion evidence.")
    parser.add_argument("--stage-b-image-motion-min-component-area", type=int, default=50, help="Drop smaller connected components from within-placement RGB motion maps.")
    parser.add_argument("--stage-b-image-motion-object-band", type=int, default=12, help="Pixels to dilate the within-group SAM-mask union before accepting RGB motion. Detached table/shadow change remains diagnostic-only.")
    parser.add_argument("--stage-b-image-motion-body-edge-band", type=int, default=4, help="Suppress RGB motion within this many pixels of the registered body silhouette boundary. A fixed dark body flickers at its high-contrast edges and those false thin bands feed both the part-pose fit and rim-shaped static patches; a real part sweeps an area so losing a thin rim is cheap. 0 disables.")
    parser.add_argument("--stage-b-hybrid-residual-dilate", type=int, default=8, help="Pixels from the silhouette residual may join hybrid evidence only within this many pixels of RGB motion. 0 requires overlap.")
    parser.add_argument("--stage-b-min-cameras", type=int, default=3, help="Minimum residual-mask cameras needed to fit a candidate part.")
    parser.add_argument("--stage-b-score-points", type=int, default=800, help="Part mesh points projected per view during Stage-B part fitting.")
    parser.add_argument("--stage-b-normal-dirs", type=int, default=24, help="Face-normal directions in the Stage-B part orientation grid.")
    parser.add_argument("--stage-b-inplane-steps", type=int, default=8, help="In-plane rotation steps per Stage-B normal direction.")
    parser.add_argument("--stage-b-coarse-cameras", type=int, default=8, help="Residual views used in the Stage-B coarse part search (0 = all).")
    parser.add_argument("--stage-b-coarse-downscale", type=int, default=4, help="Residual mask downscale during Stage-B coarse search.")
    parser.add_argument("--stage-b-refine-cameras", type=int, default=12, help="Residual views used in Stage-B local refine (0 = all).")
    parser.add_argument("--stage-b-refine-downscale", type=int, default=2, help="Residual mask downscale during Stage-B local refine.")
    parser.add_argument("--stage-b-refine-seeds", type=int, default=2, help="Best Stage-B coarse poses locally refined.")
    parser.add_argument("--stage-b-refine-rounds", type=int, default=8, help="Max Stage-B greedy refine rounds.")
    parser.add_argument("--stage-b-precision-weight", type=float, default=1.0, help="Stage-B chamfer precision weight.")
    parser.add_argument("--stage-b-recall-weight", type=float, default=0.5, help="Stage-B chamfer recall weight.")
    parser.add_argument("--stage-b-trim-fraction", type=float, default=0.3, help="Worst-view fraction dropped from Stage-B part fit scoring.")
    parser.add_argument("--stage-b-min-fit-iou", type=float, default=0.05, help="Minimum median residual-fit IoU before a part can be marked moving_candidate.")
    parser.add_argument("--stage-b-min-motion-states", type=int, default=2, help="Minimum fitted states before a part can be marked moving_candidate.")
    parser.add_argument("--stage-b-motion-min-rotation-deg", type=float, default=12.0, help="Body-relative rotation span needed for moving_candidate.")
    parser.add_argument("--stage-b-motion-min-translation", type=float, default=0.02, help="Body-relative translation span needed for moving_candidate, in mesh units.")
    parser.add_argument("--skip-stage-b-refine", action="store_true", help="Skip residual-support refinement of Stage-A part labels.")
    parser.add_argument("--stage-b-refine-mask-dilate", type=int, default=1, help="Residual mask dilation used for vertex support refinement.")
    parser.add_argument("--stage-b-refine-min-observed-cameras", type=int, default=2, help="Residual cameras that must see a vertex for a state to count it.")
    parser.add_argument("--stage-b-refine-min-observed-states", type=int, default=2, help="Residual states that must observe a vertex for refinement.")
    parser.add_argument("--stage-b-refine-present-support", type=float, default=0.15, help="A refined moving vertex must reach this residual support in some state.")
    parser.add_argument("--stage-b-refine-seed-drop", type=float, default=0.25, help="Confident residual-support drop threshold for refined moving seeds.")
    parser.add_argument("--stage-b-refine-grow-drop", type=float, default=0.12, help="Residual-support drop threshold for growing refined moving labels.")
    parser.add_argument("--stage-b-refine-grow-iters", type=int, default=80, help="Max mesh-neighbor grow iterations for Stage-B refined labels.")
    parser.add_argument("--stage-b-refine-close-holes-frac", type=float, default=0.65, help="Fill refined label holes whose movable-neighbor fraction reaches this.")
    parser.add_argument("--stage-b-refine-min-part-vertices", type=int, default=20, help="Refined components smaller than this are merged back into the body.")
    parser.add_argument("--stage-b-refine-merge-dist-frac", type=float, default=0.02, help="Merge refined movable components within this fraction of mesh diagonal.")
    parser.add_argument("--skip-stage-b-motion-refine", action="store_true", help="Skip body-pose-vs-moving-pose patch-level motion-consistency refinement.")
    parser.add_argument("--stage-b-motion-refine-min-fit-iou", type=float, default=0.08, help="Minimum candidate median fit IoU before motion-consistency refinement scores its vertices.")
    parser.add_argument("--stage-b-motion-refine-mask-dilate", type=int, default=1, help="Residual mask dilation used for body-vs-moving vertex scoring.")
    parser.add_argument("--stage-b-motion-refine-min-observed-cameras", type=int, default=2, help="Residual cameras needed to count a vertex in one motion-refine state.")
    parser.add_argument("--stage-b-motion-refine-min-states", type=int, default=2, help="States needed to keep a motion-refined vertex.")
    parser.add_argument("--stage-b-motion-refine-min-moving-support", type=float, default=0.08, help="Minimum residual support at the fitted moving pose.")
    parser.add_argument("--stage-b-motion-refine-min-advantage", type=float, default=0.08, help="Minimum max support gain of moving pose over body pose.")
    parser.add_argument("--stage-b-motion-refine-min-mean-advantage", type=float, default=0.0, help="Minimum mean support gain of moving pose over body pose.")
    parser.add_argument("--stage-b-motion-refine-close-holes-frac", type=float, default=0.70, help="Fill motion-refined label holes whose movable-neighbor fraction reaches this.")
    parser.add_argument("--stage-b-motion-refine-min-part-vertices", type=int, default=20, help="Motion-refined components smaller than this are merged back into the body.")
    parser.add_argument("--stage-b-motion-refine-merge-dist-frac", type=float, default=0.015, help="Merge motion-refined movable components within this fraction of mesh diagonal.")
    parser.add_argument("--skip-stage-b-patch-refine", action="store_true", help="Skip Stage-B local mesh-patch motion scoring and fall back to legacy vertex scoring for the Stage-C seed.")
    parser.add_argument("--stage-b-patch-min-fit-iou", type=float, default=0.08, help="Candidate evidence-fit IoU needed before its mesh patches are scored.")
    parser.add_argument("--stage-b-patch-min-faces", type=int, default=4, help="Faces required for one local mesh patch to be scored.")
    parser.add_argument("--stage-b-patch-normal-angle-deg", type=float, default=40.0, help="Welded adjacent faces differing by more than this angle start a new Stage-B patch.")
    parser.add_argument("--stage-b-patch-weld-tol-frac", type=float, default=1e-5, help="Coordinate weld tolerance as a mesh-diagonal fraction for Stage-B patch adjacency only.")
    parser.add_argument("--stage-b-patch-min-facing-cos", type=float, default=-0.15, help="Legacy centroid/raster mode: face projection is eligible when its camera-facing cosine is at least this. Keep slightly negative for imperfect OBJ winding.")
    parser.add_argument("--stage-b-patch-min-observed-cameras", type=int, default=2, help="Projection-visible evidence cameras needed to score one patch in one state.")
    parser.add_argument("--stage-b-patch-min-states", type=int, default=2, help="States with patch observations needed to retain a patch.")
    parser.add_argument("--stage-b-patch-min-groups", type=int, default=2, help="Independent placement groups with positive patch motion needed to retain it. Use 1 only for a single-placement capture.")
    parser.add_argument("--stage-b-patch-evidence-source", choices=("rgb_motion", "proposal"), default="rgb_motion", help="Evidence that certifies final local patches. rgb_motion requires direct within-group image change; proposal uses the selected hybrid/residual fit evidence for legacy ablations.")
    parser.add_argument("--stage-b-patch-score-mode", choices=("composite_joint", "composite_depth", "raster_change", "centroid"), default="composite_joint", help="Patch score. composite_joint (default) selects a patch SUBSET greedily by marginal explained-change gain under the depth-composited scene, so a static surface whose change is caused by the mover is explained away; composite_depth is the prior independent per-patch test; raster_change is the patch-only XOR ablation; centroid is the legacy sparse-hit score.")
    parser.add_argument("--stage-b-patch-change-dilate", type=int, default=2, help="Pixels to dilate RGB evidence before raster-change overlap scoring for projection tolerance.")
    parser.add_argument("--stage-b-patch-min-change-pixels", type=int, default=12, help="Predicted rest-vs-moving pixels required in one camera before it can score a patch state.")
    parser.add_argument("--stage-b-patch-min-change-precision", type=float, default=0.05, help="Minimum per-state precision of a patch's predicted raster change against RGB motion for that placement group to count.")
    parser.add_argument("--stage-b-patch-min-mean-change-precision", type=float, default=0.015, help="Minimum mean raster-change precision across observed patch states.")
    parser.add_argument("--stage-b-patch-min-pooled-change-precision", type=float, default=0.12, help="Minimum pixel-weighted precision pooled across every scored camera. Rejects broad patches whose small edge overlaps are diluted by a large predicted change area.")
    parser.add_argument("--stage-b-patch-composite-downscale", type=int, default=4, help="Downscale for depth-composite patch scoring. Change-pixel and evidence-dilate thresholds are scaled with it; lower is more exact but slower.")
    parser.add_argument("--stage-b-patch-composite-max-cameras", type=int, default=8, help="Strongest RGB-motion cameras evaluated per state by depth-composite scoring (0 = all).")
    parser.add_argument("--stage-b-patch-composite-min-change-pixels", type=int, default=4, help="Floor for predicted visible-owner-change pixels in one downscaled composite camera. Prevents one-pixel quantization changes from voting.")
    parser.add_argument("--stage-b-patch-joint-spurious-weight", type=float, default=0.25, help="Penalty per predicted-change pixel that lands outside RGB evidence in the joint subset objective. Keep moderate: untextured surfaces underreport true motion, so predicted-minus-evidence is not always false.")
    parser.add_argument("--stage-b-patch-joint-min-gain-pixels", type=int, default=8, help="Absolute marginal-gain floor (downscaled pixels, summed over all scored cameras) for adding one more patch to the joint subset.")
    parser.add_argument("--stage-b-patch-joint-min-gain-frac", type=float, default=0.05, help="Marginal gain required for each further patch, as a fraction of the first selected patch's gain. Stops noise accretion after the dominant mover is found.")
    parser.add_argument("--stage-b-patch-joint-max-selected", type=int, default=6, help="Maximum patches the greedy joint selection may keep per candidate (0 = unlimited).")
    parser.add_argument("--stage-b-patch-joint-max-patches", type=int, default=24, help="Largest-by-faces patches entering joint selection per candidate; smaller slivers are reported as skipped_joint_patch_cap (0 = unlimited).")
    parser.add_argument("--stage-b-patch-joint-refit-rounds", type=int, default=1, help="After selection, refit the part pose on the selected subset and reselect this many times. Cross-candidate refits retain an independently probed anchor patch by default; 0 disables.")
    parser.add_argument("--stage-b-patch-joint-overlay-states", type=int, default=2, help="Strongest-evidence states written as joint explained/spurious/unexplained overlays (0 disables).")
    parser.add_argument("--stage-b-patch-joint-overlay-cameras", type=int, default=2, help="Cameras per overlay state for the joint selection overlays.")
    parser.add_argument("--skip-stage-b-cross-candidate-selection", action="store_true", help="Use the older per-candidate composite-joint selection. The default global pool lets a patch borrow another candidate's SE(3) track and makes candidates explain evidence away from each other.")
    parser.add_argument("--stage-b-cross-min-track-fit-iou", type=float, default=0.08, help="Absolute Stage-B fit IoU required before a pose can anchor cross-candidate selection. Prevents the least-bad broad candidate from becoming a track when every initial fit is weak.")
    parser.add_argument("--stage-b-cross-track-min-fit-ratio", type=float, default=0.90, help="A candidate becomes a cross-selection pose track only when its Stage-B fit IoU reaches this fraction of the best candidate fit. Its source patches can still borrow an admitted track. 0 disables this relative-quality gate.")
    parser.add_argument("--stage-b-cross-max-tracks", type=int, default=2, help="Maximum reliable pose tracks admitted after the fit-ratio gate. 0 = all; source patches still come from every moving candidate.")
    parser.add_argument("--stage-b-cross-max-selected", type=int, default=8, help="Maximum patch x track hypotheses kept by global cross-candidate selection. 0 = unlimited.")
    parser.add_argument("--stage-b-cross-min-gain", type=float, default=0.002, help="Minimum group-normalized marginal explaining-away score needed to add a global patch x track hypothesis.")
    parser.add_argument("--stage-b-cross-min-gain-frac", type=float, default=0.05, help="After the first global hypothesis, required marginal gain as a fraction of the first gain.")
    parser.add_argument("--stage-b-cross-min-coverage", type=float, default=0.05, help="Minimum explained RGB-evidence fraction for a selected cross subset. Below this the result is rejected rather than exporting a small patch that only happens to overlap a much larger unexplained motion region.")
    parser.add_argument("--stage-b-cross-probe-max-patches", type=int, default=0, help="Largest-area local patches per source candidate independently refit to RGB evidence before cross selection (0 disables; this is expensive). Use to test whether a broad Stage-A candidate still contains a patch with a viable handle pose.")
    parser.add_argument("--stage-b-cross-probe-min-fit-gain", type=float, default=0.01, help="Required median-fit IoU improvement over a broad candidate before an independently fitted patch track replaces it as that source candidate's cross anchor.")
    parser.add_argument("--stage-b-cross-refit-anchor-lock", dest="stage_b_cross_refit_anchor_lock", action="store_true", default=True, help="Keep a probe anchor patch in each cross refit and reject a refit whose per-state pose drifts too far from that anchor (default on).")
    parser.add_argument("--no-stage-b-cross-refit-anchor-lock", dest="stage_b_cross_refit_anchor_lock", action="store_false", help="Allow legacy cross refits to replace a probe-anchor pose without an anchor-drift gate.")
    parser.add_argument("--stage-b-cross-refit-max-anchor-rotation-deg", type=float, default=15.0, help="Maximum per-state rotation drift from a probe-anchor pose permitted for a cross refit.")
    parser.add_argument("--stage-b-cross-refit-max-anchor-translation-frac", type=float, default=0.03, help="Maximum per-state translation drift from a probe-anchor pose, as a mesh-diagonal fraction, permitted for a cross refit.")
    parser.add_argument("--stage-b-patch-min-moving-support", type=float, default=0.06, help="Positive evidence support required at a patch's fitted moving pose.")
    parser.add_argument("--stage-b-patch-min-advantage", type=float, default=0.04, help="Moving-pose support gain over body-pose support required per positive patch state.")
    parser.add_argument("--stage-b-patch-min-mean-advantage", type=float, default=0.0, help="Mean moving-pose support gain required across every observed patch state.")
    parser.add_argument("--enable-stage-b-geometry-refine", action="store_true", default=False, help="Enable experimental body-slab/protrusion geometry refinement.")
    parser.add_argument("--skip-stage-b-geometry-refine", dest="enable_stage_b_geometry_refine", action="store_false", help="Deprecated compatibility flag; geometry refinement is disabled by default.")
    parser.set_defaults(enable_stage_b_geometry_refine=False)
    parser.add_argument("--stage-b-geometry-min-fit-iou", type=float, default=0.08, help="Minimum Stage-B median fit IoU before a candidate can enter geometry refinement.")
    parser.add_argument("--stage-b-geometry-require-moving-candidate", action="store_true", help="Only geometry-refine candidates already marked moving_candidate.")
    parser.add_argument("--stage-b-geometry-slab-percentile", type=float, default=95.0, help="Body vertex signed-distance percentile used as slab half-thickness.")
    parser.add_argument("--stage-b-geometry-seed-margin-frac", type=float, default=0.015, help="Extra mesh-diagonal fraction outside body slab required for geometry seed vertices.")
    parser.add_argument("--stage-b-geometry-grow-margin-frac", type=float, default=0.004, help="Extra mesh-diagonal fraction outside body slab allowed for growing toward the joint.")
    parser.add_argument("--stage-b-geometry-grow-iters", type=int, default=80, help="Max mesh-neighbor grow iterations for geometry refinement.")
    parser.add_argument("--stage-b-geometry-close-holes-frac", type=float, default=0.70, help="Fill geometry-refined label holes whose movable-neighbor fraction reaches this.")
    parser.add_argument("--stage-b-geometry-min-part-vertices", type=int, default=20, help="Geometry-refined components smaller than this are merged back into the body.")
    parser.add_argument("--stage-b-geometry-merge-dist-frac", type=float, default=0.015, help="Merge geometry-refined components within this fraction of mesh diagonal.")
    parser.add_argument("--skip-stage-c", action="store_true", help="Skip Stage-C solid part completion (grow the moving surface into a closed sub-volume).")
    parser.add_argument("--stage-c-mask-dilate", type=int, default=2, help="Object-mask dilation (px) for the Stage-C body-pose vs part-pose support test.")
    parser.add_argument("--stage-c-min-observed-cameras", type=int, default=2, help="Cameras that must see a vertex at both poses to count it in one Stage-C state.")
    parser.add_argument("--stage-c-weld-tol-frac", type=float, default=1e-5, help="Vertex weld tolerance as a fraction of the mesh diagonal; coincident vertices merge for connectivity (unwelded-OBJ safety). 0 disables.")
    parser.add_argument("--stage-c-bridge-dist-frac", type=float, default=0.0, help="Experimental: bridge disconnected mesh pieces within this fraction of mesh diagonal. Disabled by default because proximity is not part identity; inspect mesh-piece diagnostics before enabling.")
    parser.add_argument("--stage-c-min-body-fit-iou", type=float, default=None, help="Minimum registration fit IoU for a Stage-C state. Default: --min-fit-iou. Lower only when overlays confirm the body pose is good.")
    parser.add_argument("--stage-c-min-part-fit-iou", type=float, default=0.08, help="Minimum residual-fit IoU for an individual Stage-B part observation to contribute Stage-C evidence.")
    parser.add_argument("--stage-c-max-residual-object-fraction", type=float, default=0.85, help="Drop a state when any camera leaves more than this fraction of its object mask unexplained by the body render. 0 disables; keep high for a large separable child.")
    parser.add_argument("--stage-c-max-state-motion-fraction", type=float, default=0.75, help="Drop a state if gated motion evidence covers more than this fraction of mutually visible mesh vertices. 0 disables; catches whole-mesh residual pollution.")
    parser.add_argument("--stage-c-min-motion-groups", type=int, default=2, help="Independent placement groups that must support a vertex before it is motion-backed. Use 1 for a single-placement capture.")
    parser.add_argument("--stage-c-min-static-groups", type=int, default=2, help="Independent placement groups that must mark a vertex static before it becomes the Stage-C wall. Use 1 for a single-placement capture.")
    parser.add_argument("--stage-c-persist-advantage", type=float, default=0.10, help="Per-state residual-support gain (part pose over body pose, vs the beyond-body residual) that counts as motion evidence for a vertex.")
    parser.add_argument("--stage-c-eligible-advantage", type=float, default=0.05, help="Max residual advantage above which a vertex counts as motion-backed (excluded from the static wall).")
    parser.add_argument("--stage-c-seed-min-states", type=int, default=2, help="States with motion evidence needed for a confident Stage-C seed vertex.")
    parser.add_argument("--stage-c-seed-advantage", type=float, default=0.15, help="Max residual advantage needed for a confident Stage-C seed vertex.")
    parser.add_argument("--stage-c-static-exit", type=float, default=0.30, help="Object-support LOSS at the part pose (body minus part support) above which a no-evidence vertex is definitely static; the wall is the union of all such regions.")
    parser.add_argument("--stage-c-motion-max-exit", type=float, default=0.25, help="A state contributes motion evidence for a vertex only if the object-support loss at the part pose stays below this in that state (accidental residual grazes while exiting elsewhere do not count).")
    parser.add_argument("--stage-c-grow-iters", type=int, default=200, help="Max mesh-neighbor grow iterations when filling the solid part.")
    parser.add_argument("--stage-c-close-holes-frac", type=float, default=0.6, help="Fill a Stage-C solid hole whose part-neighbor fraction reaches this.")
    parser.add_argument("--stage-c-min-part-vertices", type=int, default=20, help="Solid parts smaller than this many vertices are dropped back to the body.")
    parser.add_argument("--stage-c-min-part-frac", type=float, default=0.15, help="Drop a solid part whose vertex count is below this fraction of the largest part (leftover noise / duplicate candidate of one physical part).")
    parser.add_argument("--stage-c-max-candidates", type=int, default=2, help="Maximum moving candidates completed into solid parts.")
    parser.add_argument("--no-html", dest="write_html", action="store_false", default=True, help="Skip segmentation_report.html.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = os.path.abspath(os.path.expanduser(args.session_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or os.path.join(session_dir, "processed")))
    masks_root = os.path.abspath(os.path.expanduser(args.masks_dir or os.path.join(output_dir, "object_masks")))
    mesh_path = os.path.abspath(os.path.expanduser(args.mesh_path))

    states = rs._discover_states(session_dir, masks_root, None)
    if not states:
        raise SystemExit(f"No states with masks under {masks_root} and cam_param under {session_dir}.")
    reg = _load_registration(output_dir)
    if not reg:
        raise SystemExit(f"No registration output under {output_dir}; run register_states.py first.")

    vertices, faces = rs._load_mesh_vertices_faces(mesh_path)
    print(f"[SEG] mesh={mesh_path} vertices={vertices.shape[0]} faces={faces.shape[0]} states={len(states)}")

    labels, min_sup, max_sup, diag = _label_vertices(vertices, faces, states, reg, args)
    print(f"[SEG] used {len(diag['used_states'])} good-fit states; observed {diag['observed_vertices']} vertices; "
          f"movable seed={diag['movable_seed']} grown={diag['movable_vertices_raw']}")

    mesh_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    merge_dist = float(args.merge_dist_frac) * mesh_diag
    face_part, part_vertex_sets = _movable_parts(vertices, faces, labels, args.min_part_vertices, merge_dist)
    # Reassign vertices in dropped components back to body for a consistent label set.
    part_of_vertex = np.full(vertices.shape[0], -1, dtype=np.int64)
    for part_id, vset in enumerate(part_vertex_sets):
        for vi in vset:
            part_of_vertex[vi] = part_id
    final_labels = (part_of_vertex >= 0).astype(np.int64)

    seg_dir = os.path.join(output_dir, "segmentation")
    parts_dir = os.path.join(seg_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    parts_info = []
    viewer_meshes = []
    combined_mesh = _labeled_preview_geometry(vertices, faces, face_part, int(args.viewer_max_faces))
    body_faces = faces[face_part < 0]
    body_path = os.path.join(parts_dir, "body.obj")
    _write_obj_submesh(body_path, vertices, body_faces)
    parts_info.append({"part_id": "body", "vertices": int(np.count_nonzero(final_labels == 0)),
                       "faces": int(body_faces.shape[0]), "mesh_path": body_path})
    body_preview = _preview_geometry(vertices, body_faces, _BODY_COLOR, int(args.viewer_max_faces))
    body_preview["name"] = "body"
    viewer_meshes.append(body_preview)
    for part_id, vset in enumerate(part_vertex_sets):
        pf = faces[face_part == part_id]
        pp = os.path.join(parts_dir, f"part_{part_id:02d}.obj")
        _write_obj_submesh(pp, vertices, pf)
        parts_info.append({"part_id": part_id, "vertices": len(vset), "faces": int(pf.shape[0]), "mesh_path": pp})
        color = _PART_COLORS[part_id % len(_PART_COLORS)]
        part_preview = _preview_geometry(vertices, pf, color, int(args.viewer_max_faces))
        part_preview["name"] = f"part_{part_id:02d}"
        viewer_meshes.append(part_preview)

    vertex_colors = np.array([_BODY_COLOR] * vertices.shape[0], dtype=np.int64)
    for vi in range(vertices.shape[0]):
        if part_of_vertex[vi] >= 0:
            vertex_colors[vi] = _PART_COLORS[int(part_of_vertex[vi]) % len(_PART_COLORS)]
    labeled_mesh_path = os.path.join(seg_dir, "mesh_labeled.ply")
    _write_colored_ply(labeled_mesh_path, vertices, faces, vertex_colors)

    overlay_paths = []
    overlays_dir = os.path.join(seg_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    good_states = [st for st in states if st["state_id"] in diag["used_states"]]
    for st in good_states[: max(0, int(args.overlay_states))]:
        sid = st["state_id"]
        T = np.asarray(reg[sid]["T_world_object"], dtype=np.float64)
        cameras = rs._load_state_cameras(st["state_dir"])
        masks = rs._load_state_masks(st["mask_dir"])
        world_vertices = rs._apply_transform(vertices, T)
        for serial in list(sorted(masks))[: max(0, int(args.overlay_cameras))]:
            if serial not in cameras:
                continue
            image = rs._load_image_bgr(st["state_dir"], serial, args.image_dirname)
            path = os.path.join(overlays_dir, f"{sid}_{serial}.jpg")
            _write_label_overlay(path, image, masks[serial], world_vertices, cameras[serial], final_labels, part_of_vertex)
            overlay_paths.append(path)

    stage_b = {"enabled": False, "status": "skipped"}
    stage_b_ctx = {}
    if not bool(args.skip_stage_b):
        stage_b, stage_b_ctx = _run_stage_b_observations(states, reg, vertices, faces, face_part, args, output_dir)
        _print_stage_b_diagnostics(stage_b, seg_dir)

    stage_c = {"enabled": False, "status": "skipped"}
    stage_c_previews = []
    if not bool(args.skip_stage_c) and not bool(args.skip_stage_b) and bool(stage_b.get("enabled")):
        stage_c = _run_stage_c_solid_parts(states, reg, vertices, faces, stage_b_ctx, args, output_dir)
        stage_c_previews = stage_c.pop("_preview_meshes", None) or []
        stage_c.pop("_combined_preview", None)
        _print_stage_c_diagnostics(stage_c, seg_dir)

    # Per-part 3D viewers, one section per stage (every part shown as its own PLY view).
    viewer_sections = [{
        "title": "Stage A parts (raw mask-consistency labels)",
        "help": "grey = body, colored = each connected movable part. drag = rotate, wheel = zoom.",
        "meshes": viewer_meshes,
    }]
    motion_face_part = stage_b_ctx.get("motion_face_part")
    if motion_face_part is not None:
        sb_meshes = []
        bp = _preview_geometry(vertices, faces[motion_face_part < 0], _BODY_COLOR, int(args.viewer_max_faces))
        bp["name"] = "body"
        sb_meshes.append(bp)
        for pid in sorted(int(x) for x in np.unique(motion_face_part) if int(x) >= 0):
            mp = _preview_geometry(vertices, faces[motion_face_part == pid], _PART_COLORS[pid % len(_PART_COLORS)], int(args.viewer_max_faces))
            mp["name"] = f"motion part_{pid:02d}"
            sb_meshes.append(mp)
        viewer_sections.append({
            "title": "Stage B patch-refined parts (primary Stage B output)",
            "help": "local mesh patches retained only when their visible projected surface repeatedly prefers the fitted moving pose over the body pose. usually still a shell (camera-facing side only).",
            "meshes": sb_meshes,
        })
    if stage_c_previews:
        viewer_sections.append({
            "title": "Stage C solid parts (closed volume, use this)",
            "help": "the moving part grown into a closed solid (front + back + interior). this is the part mesh to hand downstream.",
            "meshes": stage_c_previews,
        })

    summary = {
        "mesh_path": mesh_path,
        "vertices": int(vertices.shape[0]),
        "movable_vertices": int(np.count_nonzero(final_labels == 1)),
        "part_count": len(part_vertex_sets),
        "used_states": len(diag["used_states"]),
        "labeled_mesh_path": labeled_mesh_path,
        "parts": [{k: (v if k != "mesh_path" else v) for k, v in p.items()} for p in parts_info],
        "params": {
            "min_fit_iou": args.min_fit_iou, "support_drop": args.support_drop,
            "present_support": args.present_support, "min_part_vertices": args.min_part_vertices,
        },
        "stage_b": {
            "enabled": bool(stage_b.get("enabled")),
            "status": stage_b.get("status"),
            "parts": stage_b.get("parts", []),
            "states": stage_b.get("states", {}),
            "residual_source": stage_b.get("residual_source"),
            "residual_overlay_dir": stage_b.get("residual_overlay_dir"),
            "comparison_residual_source": stage_b.get("comparison_residual_source"),
            "comparison_residual_overlay_dir": stage_b.get("comparison_residual_overlay_dir"),
            "evidence_source": stage_b.get("evidence_source"),
            "image_motion_states": stage_b.get("image_motion_states", {}),
            "image_motion_overlay_dir": stage_b.get("image_motion_overlay_dir"),
            "primary_motion_source": stage_b.get("primary_motion_source"),
            "refinement": stage_b.get("refinement"),
            "motion_refinement": stage_b.get("motion_refinement"),
            "patch_refinement": stage_b.get("patch_refinement"),
            "geometry_refinement": stage_b.get("geometry_refinement"),
        },
    }
    summary["stage_c"] = stage_c
    with open(os.path.join(seg_dir, "segmentation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_path = None
    if getattr(args, "write_html", True):
        report_path = _write_segmentation_html(
            output_dir, mesh_path, parts_info, overlay_paths, summary, viewer_meshes, combined_mesh,
            viewer_sections=viewer_sections,
        )

    print(f"[SEG] parts={len(part_vertex_sets)} movable_vertices={summary['movable_vertices']} -> {seg_dir}")
    if report_path:
        print(f"[SEG] debug report: {report_path}")


if __name__ == "__main__":
    main()
