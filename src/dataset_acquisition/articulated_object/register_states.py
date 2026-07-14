#!/usr/bin/env python3
"""Standalone multiview mask-silhouette registration for the articulated-object pipeline.

This is the first real module of the SAM3-era rewrite (REDESIGN_PLAN.md): it fits
the known metric mesh directly to the per-state object masks, with NO plates, NO
visual hull, and NO triangulation. It consumes only what the new centered-capture
flow already produces:

    <session>/<state>/cam_param/{intrinsics,extrinsics}.json   (from capture)
    <session>/<state>/images/<serial>.png                      (undistorted; preprocess.py)
    <masks-dir>/<state>/<serial>.png                           (SAM3; generate_masks_sam3.py)
    --mesh-path <object>.obj                                    (known metric mesh)

Method (analysis-by-synthesis):
  * translation from a robust IRLS intersection of the per-view mask-centroid rays;
  * orientation from a coarse SO(3) grid scored by a trimmed-mean chamfer /
    distance-transform silhouette cost, then a local greedy refine;
  * placement-group sharing: states captured at the same object orientation share
    one body rotation, resolved by scoring each state's in-plane 90-degree variants
    AND the front/back flip against the group's pooled masks, then a group-joint
    CONTINUOUS refine of the shared rotation plus each state's translation.

Grouping is explicit (a centered object cannot be grouped by cloud position):
  * --group-by size  chunks consecutive states (default 3 = one pose's joint states);
  * --group-by label groups by the capture label prefix (e.g. p0_j1 -> group p0).

Outputs per state a pose (registration.json), silhouette overlays (orange = coarse,
green = final, yellow = mask), a fit IoU (final silhouette vs mask), and an
at-a-glance registration_report.html.

Self-contained: the pose-math helpers below were validated in and copied from
calc_states.py; this module is their canonical home once calc_states is retired.
It depends only on numpy + cv2 + trimesh (+ PIL), never on paradex, so it can run in
any environment that has those; it reads masks made in the SAM3 env off disk.
"""

import argparse
import html as html_lib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ===========================================================================
# Reused pose math (validated in / copied from calc_states.py). This module is
# the canonical home for these helpers once calc_states.py is retired.
# ===========================================================================


def _project_points(projection: "np.ndarray", points_world: "np.ndarray") -> "np.ndarray":
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    pixels_h = (projection @ points_h.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def _depths_in_camera(cam_from_world: "np.ndarray", points_world: "np.ndarray") -> "np.ndarray":
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    return (cam_from_world @ points_h.T).T[:, 2]


def _apply_transform(points: "np.ndarray", transform: "np.ndarray") -> "np.ndarray":
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    out = (transform @ points_h.T).T
    return out[:, :3]


def _camera_center_world(cam_from_world: "np.ndarray") -> "np.ndarray":
    return -cam_from_world[:3, :3].T @ cam_from_world[:3, 3]


def _subsample_points(points: "np.ndarray", max_points: int) -> "np.ndarray":
    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    return points[idx]


def _pca_axes(points: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid.reshape(1, 3)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    extents = np.ptp(centered @ axes, axis=0)
    return centroid, axes, extents


def _rotation_about_axis(axis: "np.ndarray", angle_rad: float) -> "np.ndarray":
    a = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(a))
    if norm < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = a / norm
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _fibonacci_directions(count: int) -> "np.ndarray":
    """~Evenly distributed unit vectors on the sphere (spiral / Fibonacci lattice)."""
    count = max(1, int(count))
    idx = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * idx / count
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    theta = golden * idx
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    return np.stack([x, y, np.cos(phi)], axis=1)


def _align_axis_rotation(source_axis: "np.ndarray", target_axis: "np.ndarray") -> "np.ndarray":
    """Minimal 3x3 rotation mapping unit ``source_axis`` onto unit ``target_axis``."""
    a = np.asarray(source_axis, dtype=np.float64)
    b = np.asarray(target_axis, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1.0e-12 or nb < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    a = a / na
    b = b / nb
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1.0e-9:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1.0e-9:
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, perp)
        return _rotation_about_axis(axis, np.pi)
    return _rotation_about_axis(np.cross(a, b), float(np.arccos(c)))


def _pixel_ray_world(cam: dict, pixel: Tuple[float, float]) -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Back-project an image pixel to a world-space ray (origin, unit direction)."""
    K = np.asarray(cam["K"], dtype=np.float64)
    cam_from_world = np.asarray(cam["cam_from_world"], dtype=np.float64)
    rotation = cam_from_world[:, :3]
    try:
        d_cam = np.linalg.inv(K) @ np.array([float(pixel[0]), float(pixel[1]), 1.0], dtype=np.float64)
    except np.linalg.LinAlgError:
        return None
    d_world = rotation.T @ d_cam
    norm = float(np.linalg.norm(d_world))
    if norm < 1.0e-12:
        return None
    origin = np.asarray(cam["camera_center_world"], dtype=np.float64).reshape(3)
    return origin, d_world / norm


def _robust_ray_intersection(origins, directions, iterations: int = 4):
    """IRLS closest point to a bundle of rays (robust to a minority of stray rays)."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    count = origins.shape[0]
    weights = np.ones(count, dtype=np.float64)
    point = np.mean(origins, axis=0)
    resid = np.zeros(count, dtype=np.float64)
    projectors = [np.eye(3, dtype=np.float64) - np.outer(directions[i], directions[i]) for i in range(count)]
    for _ in range(max(1, int(iterations))):
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        for i in range(count):
            A += weights[i] * projectors[i]
            b += weights[i] * (projectors[i] @ origins[i])
        try:
            point = np.linalg.solve(A + 1.0e-9 * np.eye(3), b)
        except np.linalg.LinAlgError:
            break
        for i in range(count):
            resid[i] = float(np.linalg.norm(projectors[i] @ (point - origins[i])))
        scale = float(np.median(resid)) + 1.0e-9
        weights = 1.0 / (1.0 + (resid / scale) ** 2)
    inliers = resid <= 3.0 * (float(np.median(resid)) + 1.0e-9)
    return point, resid, inliers


def _scaled_mask_and_cam(object_mask: "np.ndarray", cam: dict, downscale: int):
    """Downscale a mask and the matching camera projection for cheap silhouette scoring."""
    import cv2

    height, width = int(object_mask.shape[0]), int(object_mask.shape[1])
    ds = max(1, int(downscale))
    small_w = max(1, width // ds)
    small_h = max(1, height // ds)
    if (small_w, small_h) == (width, height):
        return object_mask.astype(bool), cam, (height, width)
    small = cv2.resize(object_mask.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0
    scale = np.array(
        [[small_w / float(width), 0.0, 0.0], [0.0, small_h / float(height), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cam_small = dict(cam)
    cam_small["projection"] = scale @ np.asarray(cam["projection"], dtype=np.float64)
    return small, cam_small, (small_h, small_w)


def _build_chamfer_view(object_mask: "np.ndarray", cam: dict, downscale: int) -> dict:
    """Precompute the per-view data for the distance-transform silhouette cost."""
    import cv2

    mask_small, cam_small, shape = _scaled_mask_and_cam(object_mask, cam, downscale)
    mask_bool = np.asarray(mask_small, dtype=bool)
    inv = (~mask_bool).astype(np.uint8)
    dt_out = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float64)
    mask_area = int(np.count_nonzero(mask_bool))
    return {
        "mask": mask_bool,
        "cam_small": cam_small,
        "shape": (int(shape[0]), int(shape[1])),
        "dt_out": dt_out,
        "mask_area": mask_area,
        "dist_scale": float(np.sqrt(max(mask_area, 1))) + 1.0e-9,
    }


def _chamfer_view_cost(world_points: "np.ndarray", view: dict, precision_weight: float, recall_weight: float) -> float:
    """Distance-transform silhouette cost for one view (lower is better).

    Precision = fraction of points outside the mask (scale-free, strong anti-spill)
    plus a small smooth outside-distance term for a convergence basin, with a ~1.5 px
    boundary tolerance so point-vs-rasterized-mask discretization is not a noise floor.
    Recall = uncovered mask area, from splatting the points into an occupancy grid.
    """
    import cv2

    cam = view["cam_small"]
    height, width = view["shape"]
    mask = view["mask"]
    dt_out = view["dt_out"]
    mask_area = int(view["mask_area"])
    dist_scale = float(view["dist_scale"])
    if mask_area <= 0:
        return float("inf")

    depths = _depths_in_camera(cam["cam_from_world"], world_points)
    front = np.isfinite(depths) & (depths > 0.0)
    if not np.any(front):
        return float(precision_weight + recall_weight)
    pixels = _project_points(cam["projection"], world_points[front])
    finite = np.isfinite(pixels).all(axis=1)
    pixels = pixels[finite]
    if pixels.shape[0] == 0:
        return float(precision_weight + recall_weight)
    xi = np.rint(pixels[:, 0]).astype(np.int64)
    yi = np.rint(pixels[:, 1]).astype(np.int64)
    inb = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    boundary_tol = 1.5
    inside = np.zeros(pixels.shape[0], dtype=bool)
    dist_norm = np.ones(pixels.shape[0], dtype=np.float64)
    if np.any(inb):
        d = dt_out[yi[inb], xi[inb]]
        inside[inb] = d <= boundary_tol
        dist_norm[inb] = np.minimum(np.maximum(d - boundary_tol, 0.0) / dist_scale, 1.0)
    frac_outside = float(1.0 - inside.mean())
    smooth_outside = float(dist_norm.mean())
    precision_penalty = frac_outside + 0.3 * smooth_outside

    recall = 0.0
    if np.any(inb):
        occ = np.zeros((height, width), dtype=np.uint8)
        occ[yi[inb], xi[inb]] = 1
        occ = cv2.dilate(occ, np.ones((3, 3), dtype=np.uint8), iterations=1)
        covered = int(np.count_nonzero((occ > 0) & mask))
        recall = covered / float(mask_area)
    recall_penalty = 1.0 - recall
    return float(precision_weight * precision_penalty + recall_weight * recall_penalty)


def _multiview_chamfer_cost(
    points_obj: "np.ndarray",
    world_T_object: "np.ndarray",
    chamfer_views: List[dict],
    precision_weight: float,
    recall_weight: float,
    trim_fraction: float,
) -> Tuple[float, int]:
    """Trimmed-mean distance-transform silhouette cost over views (lower is better)."""
    world_points = _apply_transform(np.asarray(points_obj, dtype=np.float64), world_T_object)
    scores = []
    for view in chamfer_views:
        cost = _chamfer_view_cost(world_points, view, precision_weight, recall_weight)
        if np.isfinite(cost):
            scores.append(float(cost))
    if not scores:
        return float("inf"), 0
    scores.sort()
    trim = max(0.0, min(0.9, float(trim_fraction)))
    keep = max(1, int(round(len(scores) * (1.0 - trim))))
    return float(np.mean(scores[:keep])), len(scores)


def _translation_refine_pose(
    points_obj, rotation, translation0, chamfer_views,
    precision_weight, recall_weight, trim_fraction, mesh_diag, rounds, steps,
):
    """Greedy translation-only refine with the orientation held fixed."""
    translation = np.asarray(translation0, dtype=np.float64).copy()
    world_axes = np.eye(3, dtype=np.float64)

    def _pose_t(t):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = t
        return transform

    best_cost, _n = _multiview_chamfer_cost(
        points_obj, _pose_t(translation), chamfer_views, precision_weight, recall_weight, trim_fraction
    )
    for _round in range(max(0, int(rounds))):
        best_delta = None
        best_delta_cost = best_cost
        for axis_index in range(3):
            for step in steps:
                for sign in (1.0, -1.0):
                    candidate_t = translation + world_axes[axis_index] * (sign * step * mesh_diag)
                    cost, _n = _multiview_chamfer_cost(
                        points_obj, _pose_t(candidate_t), chamfer_views, precision_weight, recall_weight, trim_fraction
                    )
                    if cost < best_delta_cost:
                        best_delta_cost = cost
                        best_delta = candidate_t
        if best_delta is None or best_delta_cost >= best_cost - 1.0e-9:
            break
        translation = best_delta
        best_cost = best_delta_cost
    return translation, best_cost


def _render_mesh_silhouette_mask(vertices, faces, world_T_object, cam, image_shape, dilate_iters: int = 0):
    """Texture-free body silhouette: rasterize the registered mesh into the view."""
    import cv2

    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return mask.astype(bool)
    world_points = _apply_transform(np.asarray(vertices, dtype=np.float64), world_T_object)
    pixels = _project_points(cam["projection"], world_points)
    depths = _depths_in_camera(cam["cam_from_world"], world_points)
    valid = np.isfinite(pixels).all(axis=1) & np.isfinite(depths) & (depths > 0.0)
    xy = np.clip(np.rint(pixels), -100000, 100000).astype(np.int32)
    faces_arr = np.asarray(faces, dtype=np.int64) if faces is not None else None
    if faces_arr is not None and faces_arr.shape[0] > 0:
        face_valid = valid[faces_arr].all(axis=1)
        tris = xy[faces_arr[face_valid]]
        if tris.shape[0] > 0:
            cv2.fillPoly(mask, [t for t in tris], 1)
    else:
        ok = valid & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        mask[xy[ok, 1], xy[ok, 0]] = 1
    if dilate_iters > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=int(dilate_iters))
    return mask.astype(bool)


def _load_mesh_vertices_faces(mesh_path: str):
    import trimesh

    geom = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        geom = trimesh.util.concatenate(geoms) if geoms else None
    if geom is None or not isinstance(geom, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh for silhouette rendering: {mesh_path}")
    return np.asarray(geom.vertices, dtype=np.float64), np.asarray(geom.faces, dtype=np.int64)


def _load_mesh_sample_points(mesh_path: str, sample_count: int) -> "np.ndarray":
    import trimesh

    geom = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no mesh geometry in scene: {mesh_path}")
        geom = trimesh.util.concatenate(geoms)
    if not isinstance(geom, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh type: {type(geom)}")
    sample_count = int(sample_count)
    if sample_count <= 0:
        return np.asarray(geom.vertices, dtype=np.float64)
    try:
        return np.asarray(geom.sample(sample_count), dtype=np.float64)
    except Exception:
        vertices = np.asarray(geom.vertices, dtype=np.float64)
        if vertices.shape[0] <= sample_count:
            return vertices
        idx = np.linspace(0, vertices.shape[0] - 1, sample_count).astype(np.int64)
        return vertices[idx]


# ===========================================================================
# Input loading
# ===========================================================================


def _reshape_extrinsic(value) -> "np.ndarray":
    E = np.asarray(value, dtype=np.float64)
    if E.shape == (3, 4):
        return E
    if E.size == 12:
        return E.reshape(3, 4)
    if E.size == 16:
        return E.reshape(4, 4)[:3, :]
    raise ValueError(f"unsupported extrinsic shape {E.shape}")


def _load_state_cameras(state_dir: str) -> dict:
    """Camera bundle from <state>/cam_param/{intrinsics,extrinsics}.json (no paradex)."""
    cam_dir = os.path.join(state_dir, "cam_param")
    try:
        with open(os.path.join(cam_dir, "intrinsics.json"), "r", encoding="utf-8") as f:
            intrinsics = json.load(f)
        with open(os.path.join(cam_dir, "extrinsics.json"), "r", encoding="utf-8") as f:
            extrinsics = json.load(f)
    except FileNotFoundError:
        return {}
    cams = {}
    for serial, intr in intrinsics.items():
        if serial not in extrinsics:
            continue
        try:
            K = np.asarray(intr["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
            cfw = _reshape_extrinsic(extrinsics[serial])
        except Exception:
            continue
        cams[serial] = {
            "serial": serial,
            "K": K,
            "cam_from_world": cfw,
            "projection": K @ cfw,
            "camera_center_world": _camera_center_world(cfw),
        }
    return cams


def _load_state_masks(masks_state_dir: str) -> dict:
    """Load <serial>.png object masks (255 = object) as bool arrays."""
    from PIL import Image

    masks = {}
    if not os.path.isdir(masks_state_dir):
        return masks
    for name in sorted(os.listdir(masks_state_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTENSIONS or stem.startswith("overlay_"):
            continue
        arr = np.asarray(Image.open(os.path.join(masks_state_dir, name)).convert("L"))
        if arr.size == 0:
            continue
        masks[stem] = arr > 127
    return masks


def _load_state_label(state_dir: str) -> str:
    path = os.path.join(state_dir, "metadata.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return str(json.load(f).get("state_label") or "")
    except Exception:
        return ""


def _discover_states(session_dir: str, masks_root: str, states_filter) -> list:
    """States that have both an object-mask folder and a cam_param folder."""
    states = []
    if not os.path.isdir(masks_root):
        return states
    for state_id in sorted(os.listdir(masks_root)):
        mask_dir = os.path.join(masks_root, state_id)
        if not os.path.isdir(mask_dir):
            continue
        if states_filter is not None and state_id not in states_filter:
            continue
        state_dir = os.path.join(session_dir, state_id)
        if not os.path.isdir(os.path.join(state_dir, "cam_param")):
            continue
        if not any(os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS and not n.startswith("overlay_")
                   for n in os.listdir(mask_dir)):
            continue
        states.append({"state_id": state_id, "state_dir": state_dir, "mask_dir": mask_dir})
    return states


# ===========================================================================
# Pose helpers
# ===========================================================================


def _pose(rotation: "np.ndarray", centroid_world: "np.ndarray", mesh_centroid: "np.ndarray") -> "np.ndarray":
    """Rigid transform placing the mesh centroid at ``centroid_world`` (rotate in place)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = centroid_world - rotation @ mesh_centroid
    return T


def _centroid_of(T: "np.ndarray", mesh_centroid: "np.ndarray") -> "np.ndarray":
    return T[:3, :3] @ mesh_centroid + T[:3, 3]


def _chamfer(scoring_points, T, views, args):
    return _multiview_chamfer_cost(
        scoring_points, T, views, args.precision_weight, args.recall_weight, args.trim_fraction
    )[0]


# ===========================================================================
# Per-state registration
# ===========================================================================


def _greedy_refine(seed_T, scoring_points, views, args, mesh_diag):
    current = np.asarray(seed_T, dtype=np.float64).copy()
    best = _chamfer(scoring_points, current, views, args)
    world_axes = np.eye(3, dtype=np.float64)
    rot_steps = (10.0, 4.0, 1.5)
    tr_steps = (0.04, 0.015, 0.005)
    for _ in range(max(0, int(args.refine_rounds))):
        center = current[:3, 3].copy()
        best_delta = None
        best_cost = best
        for axis in range(3):
            for step in rot_steps:
                for sign in (1.0, -1.0):
                    rot = _rotation_about_axis(world_axes[axis], np.deg2rad(sign * step))
                    delta = np.eye(4, dtype=np.float64)
                    delta[:3, :3] = rot
                    delta[:3, 3] = center - rot @ center
                    cand = delta @ current
                    cost = _chamfer(scoring_points, cand, views, args)
                    if cost < best_cost:
                        best_cost, best_delta = cost, cand
        for axis in range(3):
            for step in tr_steps:
                for sign in (1.0, -1.0):
                    delta = np.eye(4, dtype=np.float64)
                    delta[:3, 3] = world_axes[axis] * (sign * step * mesh_diag)
                    cand = delta @ current
                    cost = _chamfer(scoring_points, cand, views, args)
                    if cost < best_cost:
                        best_cost, best_delta = cost, cand
        if best_delta is None or best_cost >= best - 1.0e-9:
            break
        current, best = best_delta, best_cost
    return current, best


def _register_one_state(cameras, masks, vertices, mesh_points, mesh_centroid, mesh_diag, args):
    serials = [s for s in sorted(masks) if s in cameras and masks[s] is not None and bool(masks[s].any())]
    if len(serials) < max(2, int(args.min_cameras)):
        return None

    origins, directions = [], []
    for s in serials:
        ys, xs = np.nonzero(masks[s])
        ray = _pixel_ray_world(cameras[s], (float(xs.mean()), float(ys.mean())))
        if ray is not None:
            origins.append(ray[0])
            directions.append(ray[1])
    if len(origins) < max(2, int(args.min_cameras)):
        return None
    translation, _resid, ray_inliers = _robust_ray_intersection(np.asarray(origins), np.asarray(directions))

    scoring_points = _subsample_points(mesh_points, int(args.score_points))
    _centroid, mesh_axes, _extents = _pca_axes(mesh_points)
    normal_axis = mesh_axes[:, 2]

    coarse_serials = serials[: int(args.coarse_cameras)] if args.coarse_cameras > 0 else serials
    coarse_views = [_build_chamfer_view(masks[s], cameras[s], int(args.coarse_downscale)) for s in coarse_serials]

    directions_grid = _fibonacci_directions(int(args.normal_dirs))
    inplane = [2.0 * np.pi * k / int(args.inplane_steps) for k in range(int(args.inplane_steps))]
    scored = []
    for world_dir in directions_grid:
        align = _align_axis_rotation(normal_axis, world_dir)
        for phi in inplane:
            rotation = _rotation_about_axis(world_dir, phi) @ align
            T = _pose(rotation, translation, mesh_centroid)
            scored.append((_chamfer(scoring_points, T, coarse_views, args), T))
    scored.sort(key=lambda item: item[0])
    if not scored or not np.isfinite(scored[0][0]):
        return None
    coarse_T = scored[0][1]

    refine_serials = serials[: int(args.refine_cameras)] if args.refine_cameras > 0 else serials
    refine_views = [_build_chamfer_view(masks[s], cameras[s], int(args.refine_downscale)) for s in refine_serials]
    best_T, best_cost = None, float("inf")
    for _cost, seed in scored[: max(1, int(args.refine_seeds))]:
        refined_T, refined_cost = _greedy_refine(seed, scoring_points, refine_views, args, mesh_diag)
        if refined_cost < best_cost:
            best_T, best_cost = refined_T, refined_cost
    if best_T is None:
        best_T, best_cost = coarse_T, float(scored[0][0])

    return {
        "T": best_T,
        "coarse_T": coarse_T,
        "cost": float(best_cost),
        "serials": serials,
        "mask_cameras": len(serials),
        "ray_inliers": int(np.count_nonzero(ray_inliers)),
    }


# ===========================================================================
# Placement-group shared orientation + group-joint refine
# ===========================================================================


def _refine_shared_rotation(rotation, centroids, group_views, scoring_points, mesh_centroid, args):
    def group_cost(rot):
        total, count = 0.0, 0
        for views, centroid in zip(group_views, centroids):
            value = _chamfer(scoring_points, _pose(rot, centroid, mesh_centroid), views, args)
            if np.isfinite(value):
                total += value
                count += 1
        return total / max(count, 1)

    best = group_cost(rotation)
    world_axes = np.eye(3, dtype=np.float64)
    for _ in range(max(0, int(args.group_rot_rounds))):
        best_delta, best_cost = None, best
        for axis in range(3):
            for step in (4.0, 1.5, 0.5):
                for sign in (1.0, -1.0):
                    candidate = _rotation_about_axis(world_axes[axis], np.deg2rad(sign * step)) @ rotation
                    cost = group_cost(candidate)
                    if cost < best_cost:
                        best_cost, best_delta = cost, candidate
        if best_delta is None or best_cost >= best - 1.0e-9:
            break
        rotation, best = best_delta, best_cost
    return rotation, best


def _share_group(group, vertices, mesh_points, mesh_centroid, mesh_diag, args):
    """Enforce one shared body rotation across a group and jointly refine it.

    ``group`` is a list of per-state dicts with 'cameras', 'masks', 'result'.
    Adds 'T' and 'status' to each on success. A group with <2 registered states is
    left as its independent per-state fit.
    """
    registered = [st for st in group if st.get("result") is not None]
    if len(registered) < 2:
        for st in group:
            if st.get("result") is not None:
                st["T"] = st["result"]["T"]
                st["status"] = "mask_silhouette"
        return

    scoring_points = _subsample_points(mesh_points, int(args.score_points))
    _centroid, mesh_axes, _extents = _pca_axes(mesh_points)
    normal_axis = mesh_axes[:, 2]
    inplane_axis = mesh_axes[:, 0]

    group_views, centroids = [], []
    for st in registered:
        serials = st["result"]["serials"][: int(args.refine_cameras)] if args.refine_cameras > 0 else st["result"]["serials"]
        st["_views"] = [_build_chamfer_view(st["masks"][s], st["cameras"][s], int(args.refine_downscale)) for s in serials]
        group_views.append(st["_views"])
        centroids.append(_centroid_of(st["result"]["T"], mesh_centroid))

    # Discrete ambiguity candidates from each state's independent fit: the four
    # in-plane 90-degree rotations about the mesh normal AND the front/back flip
    # (180 degrees about the long in-plane axis). A flat object is nearly symmetric
    # under both; pooling the group's masks (and the asymmetric part, e.g. a handle
    # on one face) picks the right one where a single state cannot.
    variants = []
    for k in range(4):
        for flip in (0.0, np.pi):
            variants.append(
                _rotation_about_axis(normal_axis, k * np.pi / 2.0) @ _rotation_about_axis(inplane_axis, flip)
            )
    candidates = []
    for st in registered:
        base = st["result"]["T"][:3, :3]
        for variant in variants:
            rot = base @ variant
            if not any(float(np.trace(kept.T @ rot)) > 3.0 - 1.0e-3 for kept in candidates):
                candidates.append(rot)

    best_rotation, best_cost = None, float("inf")
    for rotation in candidates:
        total, count = 0.0, 0
        for views, centroid in zip(group_views, centroids):
            value = _chamfer(scoring_points, _pose(rotation, centroid, mesh_centroid), views, args)
            if np.isfinite(value):
                total += value
                count += 1
        cost = total / max(count, 1)
        if cost < best_cost:
            best_cost, best_rotation = cost, rotation
    if best_rotation is None:
        best_rotation = registered[0]["result"]["T"][:3, :3]

    # Group-joint continuous refine: shared rotation <-> per-state translation.
    rotation = best_rotation
    for _ in range(max(1, int(args.group_iters))):
        rotation, _cost = _refine_shared_rotation(rotation, centroids, group_views, scoring_points, mesh_centroid, args)
        for i, (st, views) in enumerate(zip(registered, group_views)):
            t0 = centroids[i] - rotation @ mesh_centroid
            t_ref, _c = _translation_refine_pose(
                scoring_points, rotation, t0, views, args.precision_weight, args.recall_weight,
                args.trim_fraction, mesh_diag, int(args.refine_rounds), (0.02, 0.008, 0.003),
            )
            centroids[i] = rotation @ mesh_centroid + t_ref

    for st, centroid in zip(registered, centroids):
        st["T"] = _pose(rotation, centroid, mesh_centroid)
        st["status"] = "mask_silhouette_group_shared"
    for st in group:
        if st.get("result") is not None and "T" not in st:
            st["T"] = st["result"]["T"]
            st["status"] = "mask_silhouette"


# ===========================================================================
# Grouping
# ===========================================================================


def _group_states(state_ids, labels, args):
    from collections import OrderedDict

    groups = OrderedDict()
    if str(args.group_by) == "label":
        for sid in state_ids:
            match = re.match(r"(p\d+)", labels.get(sid, ""))
            key = match.group(1) if match else sid
            groups.setdefault(key, []).append(sid)
    else:
        size = max(1, int(args.group_size))
        for index, sid in enumerate(state_ids):
            groups.setdefault(f"g{index // size:03d}", []).append(sid)
    return groups


# ===========================================================================
# Quality metric + overlays + HTML report
# ===========================================================================


def _silhouette_iou(silhouette, mask) -> float:
    inter = int(np.count_nonzero(silhouette & mask))
    union = int(np.count_nonzero(silhouette | mask))
    return inter / union if union > 0 else 0.0


def _fit_iou(vertices, faces, T, cameras, masks, serials, trim_fraction) -> Optional[float]:
    """Trimmed-mean IoU of the final mesh silhouette against the object masks.

    A single scalar 'does green match yellow' quality: the moving articulated part
    and FP/FN masks are the worst views, so the worst ``trim_fraction`` are dropped.
    """
    ious = []
    for s in serials:
        if s not in cameras or s not in masks:
            continue
        m = masks[s]
        sil = _render_mesh_silhouette_mask(vertices, faces, T, cameras[s], m.shape, 0)
        ious.append(_silhouette_iou(sil, m))
    if not ious:
        return None
    ious.sort(reverse=True)
    keep = max(1, int(round(len(ious) * (1.0 - float(trim_fraction)))))
    return float(np.mean(ious[:keep]))


def _load_image_bgr(state_dir, serial, image_dirname):
    import cv2

    for ext in sorted(IMAGE_EXTENSIONS):
        path = os.path.join(state_dir, image_dirname, f"{serial}{ext}")
        if os.path.exists(path):
            data = np.fromfile(path, dtype=np.uint8)
            if data.size:
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None


def _write_overlay(path, image_bgr, mask, coarse_T, final_T, cam, vertices, faces):
    import cv2

    height, width = mask.shape[:2]
    canvas = image_bgr.copy() if image_bgr is not None else np.zeros((height, width, 3), dtype=np.uint8)

    def draw(mask_bool, color, thickness):
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, thickness)

    draw(mask, (0, 255, 255), 2)  # object mask boundary = yellow
    if coarse_T is not None:
        draw(_render_mesh_silhouette_mask(vertices, faces, coarse_T, cam, (height, width), 0), (0, 140, 255), 2)  # orange
    draw(_render_mesh_silhouette_mask(vertices, faces, final_T, cam, (height, width), 0), (0, 255, 0), 2)  # green
    cv2.imwrite(path, canvas)


def _fit_badge(fit: Optional[float]) -> Tuple[str, str]:
    if fit is None:
        return "n/a", "#888"
    if fit >= 0.85:
        return f"{fit:.2f}", "#2e7d32"
    if fit >= 0.6:
        return f"{fit:.2f}", "#b8860b"
    return f"{fit:.2f}", "#c0392b"


def _write_html_report(output_dir, session_dir, mesh_path, groups, records) -> str:
    esc = html_lib.escape

    def rel(path):
        try:
            return os.path.relpath(path, output_dir).replace(os.sep, "/")
        except Exception:
            return None

    total = len(records)
    shared = sum(1 for r in records.values() if r["status"] == "mask_silhouette_group_shared")
    skipped = sum(1 for r in records.values() if r["status"] == "skipped")
    low_fit = sum(1 for r in records.values() if (r.get("fit_iou") is not None and r["fit_iou"] < 0.6))

    parts = [
        "<style>",
        "body{font-family:system-ui,Arial,sans-serif;margin:16px;background:#f6f7f9;color:#1a1a1a}",
        "h1{font-size:20px}h2{font-size:16px;margin:18px 0 8px}",
        ".summary{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0 4px}",
        ".summary div{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px 14px}",
        ".legend span{display:inline-block;margin-right:14px;font-size:13px}",
        ".dot{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:middle}",
        ".cards{display:flex;flex-wrap:wrap;gap:12px}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px;width:300px}",
        ".card.bad{border-color:#c0392b;box-shadow:0 0 0 1px #c0392b}",
        ".card.skip{opacity:.6}",
        ".meta{font-size:12px;color:#444;margin:4px 0}",
        ".badge{color:#fff;border-radius:10px;padding:1px 8px;font-size:12px;font-weight:600}",
        ".thumbs{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}",
        ".thumbs img{width:140px;height:auto;border:1px solid #ccc;border-radius:4px}",
        "</style>",
        "<h1>Registration debug report</h1>",
        f"<div class='meta'>session: {esc(session_dir)}<br>mesh: {esc(mesh_path)}</div>",
        "<div class='summary'>",
        f"<div><b>{total}</b> states</div>",
        f"<div><b>{shared}</b> group-shared</div>",
        f"<div><b>{skipped}</b> skipped</div>",
        f"<div><b>{low_fit}</b> low fit (&lt;0.6)</div>",
        "</div>",
        "<div class='legend'>"
        "<span><span class='dot' style='background:#ffff00'></span>mask (yellow)</span>"
        "<span><span class='dot' style='background:#00ff00'></span>final pose (green)</span>"
        "<span><span class='dot' style='background:#ff8c00'></span>coarse pose (orange)</span>"
        "<span>fit = trimmed-mean silhouette&cap;mask IoU</span>"
        "</div>",
    ]

    for gid, members in groups.items():
        parts.append(f"<h2>group {esc(str(gid))}</h2><div class='cards'>")
        for sid in members:
            rec = records.get(sid, {})
            fit_text, fit_color = _fit_badge(rec.get("fit_iou"))
            classes = "card"
            if rec.get("status") == "skipped":
                classes += " skip"
            elif rec.get("fit_iou") is not None and rec["fit_iou"] < 0.6:
                classes += " bad"
            cost = rec.get("cost")
            parts.append(f"<div class='{classes}'>")
            parts.append(
                f"<div><b>{esc(sid)}</b> "
                f"<span class='badge' style='background:{fit_color}'>fit {fit_text}</span></div>"
            )
            parts.append(
                f"<div class='meta'>label: {esc(rec.get('state_label', '') or '-')} | "
                f"status: {esc(rec.get('status', '-'))}<br>"
                f"masks: {esc(str(rec.get('mask_cameras', '-')))} | "
                f"cost: {('%.4f' % cost) if isinstance(cost, float) else '-'}</div>"
            )
            thumbs = []
            for overlay in rec.get("overlays", []):
                r = rel(overlay)
                if r:
                    thumbs.append(f"<a href='{esc(r)}'><img src='{esc(r)}' loading='lazy'></a>")
            if thumbs:
                parts.append("<div class='thumbs'>" + "".join(thumbs) + "</div>")
            parts.append("</div>")
        parts.append("</div>")

    report_path = os.path.join(output_dir, "registration_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Registration debug report</title></head><body>"
                + "".join(parts) + "</body></html>")
    return report_path


# ===========================================================================
# Main
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multiview mask-silhouette registration of a known mesh to SAM3 object masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session-dir", required=True, help="Capture session dir with per-state cam_param and images.")
    parser.add_argument("--mesh-path", required=True, help="Known metric mesh (.obj) to register.")
    parser.add_argument("--masks-dir", default=None, help="Object-mask root. Default: <session>/processed/object_masks.")
    parser.add_argument("--output-dir", default=None, help="Results root. Default: <session>/processed.")
    parser.add_argument("--image-dirname", default="images", help="Per-state undistorted image folder (for overlays).")
    parser.add_argument("--states", default=None, help="Comma-separated state ids (default: all).")
    parser.add_argument("--sample-count", type=int, default=4000, help="Mesh surface points sampled for PCA/diagonal.")
    parser.add_argument("--score-points", type=int, default=2000, help="Mesh surface points projected per view for the chamfer cost.")
    parser.add_argument("--min-cameras", type=int, default=3, help="Minimum masked views to attempt a state.")
    parser.add_argument("--normal-dirs", type=int, default=48, help="Face-normal directions in the coarse orientation grid.")
    parser.add_argument("--inplane-steps", type=int, default=12, help="In-plane rotation steps per normal direction.")
    parser.add_argument("--coarse-cameras", type=int, default=8, help="Views used in the coarse search (0 = all).")
    parser.add_argument("--coarse-downscale", type=int, default=4, help="Mask downscale during the coarse search.")
    parser.add_argument("--refine-cameras", type=int, default=12, help="Views used in the local/group refine (0 = all).")
    parser.add_argument("--refine-downscale", type=int, default=2, help="Mask downscale during the refine.")
    parser.add_argument("--refine-seeds", type=int, default=2, help="Best coarse orientations locally refined.")
    parser.add_argument("--refine-rounds", type=int, default=12, help="Max greedy coordinate-descent rounds.")
    parser.add_argument("--precision-weight", type=float, default=1.0, help="Chamfer precision weight.")
    parser.add_argument("--recall-weight", type=float, default=0.5, help="Chamfer recall weight.")
    parser.add_argument("--trim-fraction", type=float, default=0.3, help="Worst-view fraction dropped from the multiview cost / fit.")
    parser.add_argument("--group-by", default="size", choices=["size", "label"], help="Placement grouping: consecutive chunks or label prefix (p<k>).")
    parser.add_argument("--group-size", type=int, default=3, help="States per group for --group-by size (one pose's joint states).")
    parser.add_argument("--group-share", dest="group_share", action="store_true", default=True, help="Enforce a shared body orientation per group (default on).")
    parser.add_argument("--no-group-share", dest="group_share", action="store_false", help="Register every state independently.")
    parser.add_argument("--group-iters", type=int, default=2, help="Group-joint refine alternations (shared rotation <-> translations).")
    parser.add_argument("--group-rot-rounds", type=int, default=8, help="Greedy rounds when refining the shared rotation.")
    parser.add_argument("--overlay-cameras", type=int, default=4, help="Cameras per state written as silhouette overlays (0 = none).")
    parser.add_argument("--no-html", dest="write_html", action="store_false", default=True, help="Skip the registration_report.html debug view.")
    return parser.parse_args()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    session_dir = os.path.abspath(os.path.expanduser(args.session_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or os.path.join(session_dir, "processed")))
    masks_root = os.path.abspath(os.path.expanduser(args.masks_dir or os.path.join(output_dir, "object_masks")))
    mesh_path = os.path.abspath(os.path.expanduser(args.mesh_path))

    states_filter = None
    if args.states:
        states_filter = {t.strip() for t in str(args.states).split(",") if t.strip()}
    states = _discover_states(session_dir, masks_root, states_filter)
    if not states:
        raise SystemExit(f"No states with masks under {masks_root} and cam_param under {session_dir}.")

    vertices, faces = _load_mesh_vertices_faces(mesh_path)
    mesh_points = _load_mesh_sample_points(mesh_path, int(args.sample_count))
    mesh_centroid = np.mean(vertices, axis=0)
    mesh_diag = float(np.linalg.norm(mesh_points.max(axis=0) - mesh_points.min(axis=0)))
    print(f"[REG] session={session_dir}")
    print(f"[REG] mesh={mesh_path} diag={mesh_diag:.3f} states={len(states)}")

    for st in states:
        st["cameras"] = _load_state_cameras(st["state_dir"])
        st["masks"] = _load_state_masks(st["mask_dir"])
        st["label"] = _load_state_label(st["state_dir"])
        st["result"] = _register_one_state(
            st["cameras"], st["masks"], vertices, mesh_points, mesh_centroid, mesh_diag, args
        )
        if st["result"] is None:
            st["status"] = "skipped"
            print(f"[REG] {st['state_id']}: skipped (too few masks: {len(st['masks'])})")
        else:
            st["T"] = st["result"]["T"]
            st["status"] = "mask_silhouette"
            r = st["result"]
            print(f"[REG] {st['state_id']}: masks={r['mask_cameras']} rays_in={r['ray_inliers']} cost={r['cost']:.4f}")

    labels = {st["state_id"]: st["label"] for st in states}
    groups = _group_states([st["state_id"] for st in states], labels, args)
    by_id = {st["state_id"]: st for st in states}
    if args.group_share:
        for gid, members in groups.items():
            group = [by_id[s] for s in members]
            _share_group(group, vertices, mesh_points, mesh_centroid, mesh_diag, args)
            shared = sum(1 for st in group if st.get("status") == "mask_silhouette_group_shared")
            print(f"[GROUPREG] group={gid}: states={len(members)} shared={shared}")

    group_of = {s: gid for gid, members in groups.items() for s in members}
    records = {}
    for st in states:
        sid = st["state_id"]
        reg_dir = os.path.join(output_dir, "registration", sid)
        os.makedirs(reg_dir, exist_ok=True)
        fit = None
        overlays = []
        if st.get("result") is not None and st.get("T") is not None:
            fit = _fit_iou(vertices, faces, st["T"], st["cameras"], st["masks"], st["result"]["serials"], args.trim_fraction)
            if int(args.overlay_cameras) > 0:
                for serial in st["result"]["serials"][: int(args.overlay_cameras)]:
                    image = _load_image_bgr(st["state_dir"], serial, args.image_dirname)
                    overlay_path = os.path.join(reg_dir, f"overlay_{serial}.jpg")
                    _write_overlay(overlay_path, image, st["masks"][serial], st["result"]["coarse_T"],
                                   st["T"], st["cameras"][serial], vertices, faces)
                    overlays.append(overlay_path)
        record = {
            "state_id": sid,
            "state_label": st.get("label", ""),
            "placement_group": group_of.get(sid),
            "status": st.get("status", "skipped"),
            "T_world_object": st["T"].tolist() if st.get("T") is not None else None,
            "mask_cameras": (st["result"]["mask_cameras"] if st.get("result") else 0),
            "cost": (st["result"]["cost"] if st.get("result") else None),
            "fit_iou": fit,
        }
        _write_json(os.path.join(reg_dir, "registration.json"), record)
        record["overlays"] = overlays
        records[sid] = record

    manifest = {
        "session_dir": session_dir,
        "mesh_path": mesh_path,
        "groups": {gid: members for gid, members in groups.items()},
        "states": {sid: {k: v for k, v in rec.items() if k != "overlays"} for sid, rec in records.items()},
    }
    _write_json(os.path.join(output_dir, "registration_manifest.json"), manifest)

    report_path = None
    if getattr(args, "write_html", True):
        report_path = _write_html_report(output_dir, session_dir, mesh_path, groups, records)

    ok = sum(1 for st in states if st.get("T") is not None)
    print(f"[REG] wrote {ok}/{len(states)} state poses -> {os.path.join(output_dir, 'registration')}")
    if report_path:
        print(f"[REG] debug report: {report_path}")


if __name__ == "__main__":
    main()
