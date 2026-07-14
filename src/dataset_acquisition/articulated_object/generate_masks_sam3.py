#!/usr/bin/env python3
"""Generate per-state object masks with SAM3 (promptable concept segmentation).

Standalone mask provider for the articulated-object pipeline (REDESIGN_PLAN.md,
Round 0). It runs inside the SAM3 environment (Python 3.12+, torch), e.g.:

    conda run -n sam3 python generate_masks_sam3.py \
        --session-dir /path/to/capture/<object>/<session> \
        --prompt "wooden picture frame"

and must therefore stay free of paradex imports. The only interchange with
calc_states.py is the mask contract on disk:

    <output-dir>/object_masks/<state_id>/<serial>.png   8-bit, 255 = object
    <output-dir>/object_masks/<state_id>/overlay_<serial>.jpg   inspection panels
    <output-dir>/object_masks/mask_stats.csv
    <output-dir>/object_masks/masks_manifest.json       declares the provider

calc_states.py consumes these with ``--object-mask-source external`` (or the
default ``auto``, which prefers an external manifest when one is present).

The script reads the *undistorted* images (``<state>/images/<serial>.png``) so
the masks align with the calibrated camera model. It no longer needs a prior
calc run: when a state has raw captures but no undistorted images, the standalone
``preprocess.py`` (cv2 + numpy, paradex-free) is invoked to undistort them first
(disable with ``--no-auto-undistort``). The undistortion uses only the per-state
``cam_param`` written at capture time.

ROI-aware generation (default when available): the box describing "where the
object is" comes from the per-state *triangulated object cloud*
(``multiview/<state>/object_points_roi.ply``, written by any calc run,
mask-independent): a median-anchored 3D bbox, size-bounded by the input-mesh
diagonal from the calc manifest (the cloud is best-effort object-only and can be
scene-contaminated when the ROI crop was skipped for an off-center placement -
median + mesh bound survive that), expanded by ``--object-box-expand``,
projected per camera. This box is object-centered and object-sized. The ROI *sphere* from ``pipeline_manifest.json`` is only a loose
fallback for the visibility test when the cloud is missing - it is centered on
the camera convergence point (not the object) with radius 1.5x the mesh
diagonal, so it is far too big to prompt with. Effects per camera:

- cameras whose projected object box misses the image are skipped as
  ``object_out_of_view`` (no inference, no false positives on empty views);
- when the concept prompt returns several instances, candidates must overlap
  the projected box (``--roi-min-overlap``) before top-score/union selection;
- when the concept prompt returns nothing although the box is visible (e.g. a
  close-up top view showing only wood grain, which no longer *looks like* a
  "picture frame"), the view is left blank during the SAM pass. After all
  views of the state are processed, the successful SAM masks carve a coarse
  visual hull inside the object box; the projected hull is used as a location
  guide for a second SAM3 call using ``--fallback-prompt`` plus the hull bbox
  (``prompt_mode=hull_guided_sam3``).
  The projected hull itself is only a coarse guide, not the default saved mask.

Notes:
- SAM3 masks can contain small holes. Multi-view hull voting absorbs holes that
  differ between views, and this script additionally fills *enclosed* holes
  below ``--fill-holes-max-frac`` of the mask area. Real openings (e.g. the gap
  behind a fold-out handle) stay open: they are either larger than the limit or
  connected to the outside background.
- Machine-specific locations are arguments, not hardcoded defaults; the only
  built-in convenience is ``--sam3-repo-root ~/sam3``.
"""

import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

_HOLE_FILL_WARNED = [False]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3 object-mask provider for the articulated-object pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--session-dir",
        required=True,
        help="Capture session directory (<capture_root>/<object>/<session>) containing the state folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pipeline output directory; masks go to <output-dir>/object_masks/. Default: <session-dir>/processed.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Noun-phrase concept prompt for SAM3 (e.g. 'wooden picture frame'). Highest precedence.",
    )
    parser.add_argument(
        "--fallback-prompt",
        default="foreground object",
        help=(
            "Text prompt used only for concept-miss views when retrying with the projected-hull bbox. "
            "The main pass still uses --prompt."
        ),
    )
    parser.add_argument(
        "--prompt-json",
        default=None,
        help=(
            "JSON file mapping object names to prompts; used when --prompt is not given. "
            "Accepts {name: 'phrase'} or {name: {'prompt': 'phrase'}}."
        ),
    )
    parser.add_argument(
        "--object-name",
        default=None,
        help=(
            "Object name used as the --prompt-json key and, as a last resort, as the prompt itself "
            "(underscores become spaces; only meaningful when the name is a natural noun phrase). "
            "Default: the parent directory name of --session-dir."
        ),
    )
    parser.add_argument(
        "--states",
        default=None,
        help="Comma-separated state ids to process (default: every state folder in the session).",
    )
    parser.add_argument(
        "--image-dirname",
        default="images",
        help="Per-state image folder to read; 'images' holds the undistorted views that match the calibration.",
    )
    parser.add_argument(
        "--auto-undistort",
        dest="auto_undistort",
        action="store_true",
        default=True,
        help=(
            "Before reading masks, undistort each state's raw captures into the image folder via the "
            "standalone preprocess.py when they are missing, so no prior calc run is needed (default on)."
        ),
    )
    parser.add_argument(
        "--no-auto-undistort",
        dest="auto_undistort",
        action="store_false",
        help="Disable auto-undistortion; expect <state>/images/ to be pre-populated (e.g. by a calc run).",
    )
    parser.add_argument(
        "--sam3-repo-root",
        default="~/sam3",
        help="SAM3 repository root; inserted into sys.path when the 'sam3' package is not already installed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. Default: let the SAM3 builder resolve it (HF cache).",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for inference.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop instances with SAM confidence below this; when none survive, no mask file is written.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional override of the SAM3 processor's internal mask confidence threshold.",
    )
    parser.add_argument(
        "--instance-mode",
        default="top1",
        choices=["top1", "union"],
        help=(
            "How to combine returned instances: 'top1' keeps the highest-scoring one (the observed case: "
            "SAM3 returns the whole object incl. moving parts as one instance); 'union' ORs every instance "
            "above --min-score (fallback if SAM splits the object)."
        ),
    )
    parser.add_argument(
        "--roi-source",
        default="auto",
        choices=["auto", "off"],
        help=(
            "'auto' localizes the object per state/camera (object-cloud 3D bbox from "
            "multiview/<state>/object_points_roi.ply, ROI sphere from pipeline_manifest.json as loose "
            "fallback) and enables out-of-view skipping, box-overlap instance selection and the projected-hull "
            "guide used by fallback retries; 'off' disables all ROI-aware behavior."
        ),
    )
    parser.add_argument(
        "--object-box-expand",
        type=float,
        default=0.2,
        help=(
            "Expand the object box by this fraction of the input-mesh diagonal (cloud diagonal when the "
            "mesh size is unavailable) on every side; covers texture-poor extremities such as the handle."
        ),
    )
    parser.add_argument(
        "--object-box-min-points",
        type=int,
        default=30,
        help="Minimum object-cloud points required to trust the per-state 3D object box.",
    )
    parser.add_argument(
        "--roi-pad",
        type=float,
        default=1.1,
        help="Padding factor applied to the projected fallback ROI-sphere bbox (not the object box).",
    )
    parser.add_argument(
        "--roi-min-visible-frac",
        type=float,
        default=0.05,
        help="Skip a camera as object_out_of_view when less than this fraction of the ROI bbox is inside the image.",
    )
    parser.add_argument(
        "--roi-min-overlap",
        type=float,
        default=0.5,
        help="Instance filter: at least this fraction of an instance mask must lie inside the padded ROI bbox.",
    )
    parser.add_argument(
        "--hull-guided-box-expand",
        type=float,
        default=0.10,
        help="Expand the projected-hull tight bbox by this fraction of its max side before the SAM3 retry.",
    )
    parser.add_argument(
        "--hull-guided-min-overlap",
        type=float,
        default=0.30,
        help="SAM3 retry candidate gate: minimum fraction of candidate pixels inside the projected hull guide.",
    )
    parser.add_argument(
        "--hull-guided-min-guide-coverage",
        type=float,
        default=0.25,
        help="SAM3 retry candidate gate: minimum fraction of projected hull guide pixels covered by the candidate.",
    )
    parser.add_argument(
        "--hull-projection-grid-resolution",
        type=int,
        default=64,
        help="Longest-axis voxel count used for the projected-hull guide.",
    )
    parser.add_argument(
        "--hull-projection-min-source-masks",
        type=int,
        default=3,
        help="Minimum number of successful SAM masks needed before building the projected-hull guide.",
    )
    parser.add_argument(
        "--hull-projection-min-opportunities",
        type=int,
        default=2,
        help="Minimum source-mask cameras that must see a voxel in the guide hull vote.",
    )
    parser.add_argument(
        "--hull-projection-min-view-fraction",
        type=float,
        default=0.55,
        help="Minimum source-mask vote fraction required for a voxel to enter the guide hull.",
    )
    parser.add_argument(
        "--hull-projection-dilate",
        type=int,
        default=2,
        help="3x3 dilation iterations applied to projected hull guides to hide voxel raster gaps.",
    )
    parser.add_argument(
        "--hull-projection-min-mask-fraction",
        type=float,
        default=0.0005,
        help="Reject a projected hull guide smaller than this image fraction.",
    )
    parser.add_argument(
        "--hull-projection-max-mask-fraction",
        type=float,
        default=0.60,
        help="Reject a projected hull guide larger than this image fraction.",
    )
    parser.add_argument(
        "--fill-holes-max-frac",
        type=float,
        default=0.01,
        help=(
            "Fill enclosed background holes smaller than this fraction of the mask area "
            "(0 disables). Keeps real openings such as an articulated-handle gap."
        ),
    )
    parser.add_argument(
        "--overlay-cameras",
        type=int,
        default=4,
        help="Number of cameras per state exported as [image | mask] inspection panels.",
    )
    return parser.parse_args()


def _resolve_prompt(args: argparse.Namespace, object_name: str):
    if args.prompt:
        return str(args.prompt).strip(), "arg"
    if args.prompt_json:
        path = os.path.expanduser(args.prompt_json)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get(object_name)
        if isinstance(entry, dict):
            entry = entry.get("prompt") or entry.get("text")
        if entry:
            return str(entry).strip(), "prompt-json"
        print(f"[SAM3][WARN] {path} has no prompt for '{object_name}'; falling back to the object name.")
    if object_name:
        cleaned = object_name.replace("_", " ").strip()
        if cleaned:
            print(
                "[SAM3][WARN] using the object name as the concept prompt; this only works when the "
                "name is a natural noun phrase (prefer an explicit --prompt)."
            )
            return cleaned, "object-name"
    raise SystemExit("A prompt is required: pass --prompt, or --prompt-json with a matching --object-name.")


def _ensure_undistorted(session_dir: str, args: argparse.Namespace) -> None:
    """Undistort raw captures into <state>/<image-dirname>/ via the standalone
    preprocess module, so mask generation does not require a prior calc run.

    Best-effort: any failure (module missing, cv2 unavailable, no cam_param) is a
    warning, not fatal - the caller still reports 'no states' if nothing exists.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        import preprocess
    except Exception as exc:
        print(f"[SAM3][WARN] preprocess module unavailable ({exc}); expecting pre-undistorted images.")
        return
    try:
        result = preprocess.undistort_session(session_dir, states=args.states, image_dirname=args.image_dirname)
    except Exception as exc:
        print(f"[SAM3][WARN] auto-undistort failed ({exc}); expecting pre-undistorted images.")
        return
    n_states = len(result)
    n_images = sum(len(serials) for serials in result.values())
    if n_states:
        print(f"[SAM3] auto-undistort: {n_states} states, {n_images} images in <state>/{args.image_dirname}/")


def _discover_states(session_dir: str, image_dirname: str, states_filter) -> list:
    states = []
    for name in sorted(os.listdir(session_dir)):
        state_dir = os.path.join(session_dir, name)
        images_dir = os.path.join(state_dir, image_dirname)
        if not os.path.isdir(images_dir):
            continue
        serials = sorted(
            os.path.splitext(entry)[0]
            for entry in os.listdir(images_dir)
            if os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS
        )
        if not serials:
            continue
        if states_filter is not None and name not in states_filter:
            continue
        states.append({"state_id": name, "state_dir": state_dir, "images_dir": images_dir, "serials": serials})
    return states


def _find_image_path(images_dir: str, serial: str):
    for ext in sorted(IMAGE_EXTENSIONS):
        path = os.path.join(images_dir, f"{serial}{ext}")
        if os.path.exists(path):
            return path
    return None


def _load_roi_spheres(output_dir: str) -> dict:
    """Per-state ROI spheres from the calc pipeline manifest (camera-geometry
    derived, so they exist after any calc run, independent of masks)."""
    path = os.path.join(output_dir, "pipeline_manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        print(f"[SAM3][WARN] could not read {path}: {exc}")
        return {}
    rows = ((manifest.get("object_roi_crop") or {}).get("states")) or []
    spheres = {}
    for row in rows:
        center = row.get("roi_center")
        radius = row.get("roi_radius")
        state_id = str(row.get("state_id"))
        if center is not None and radius:
            spheres[state_id] = (np.asarray(center, dtype=np.float64).reshape(3), float(radius))
    return spheres


def _load_state_cameras(state_dir: str) -> dict:
    """Undistorted camera models from <state>/cam_param/*.json (no paradex)."""
    cam_dir = os.path.join(state_dir, "cam_param")
    cams = {}
    try:
        with open(os.path.join(cam_dir, "intrinsics.json"), "r", encoding="utf-8") as f:
            intrinsics = json.load(f)
        with open(os.path.join(cam_dir, "extrinsics.json"), "r", encoding="utf-8") as f:
            extrinsics = json.load(f)
    except Exception:
        return cams
    for serial, intr in intrinsics.items():
        if serial not in extrinsics:
            continue
        try:
            K = np.asarray(intr["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
            E = np.asarray(extrinsics[serial], dtype=np.float64)
            if E.shape == (4, 4):
                E = E[:3, :]
            elif E.size == 12:
                E = E.reshape(3, 4)
            elif E.size == 16:
                E = E.reshape(4, 4)[:3, :]
            if E.shape != (3, 4):
                continue
        except Exception:
            continue
        cams[serial] = {"K": K, "E": E}
    return cams


def _load_object_points(output_dir: str, state_id: str, min_points: int):
    """Per-state triangulated object-only cloud (ASCII PLY written by the calc
    ROI stage; x y z r g b per line). Mask-independent and object-positioned,
    unlike the ROI sphere."""
    path = os.path.join(output_dir, "multiview", state_id, "object_points_roi.ply")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="ascii", errors="replace") as f:
            if f.readline().strip() != "ply":
                return None
            ascii_ok = False
            count = 0
            while True:
                line = f.readline()
                if not line:
                    return None
                line = line.strip()
                if line.startswith("format"):
                    ascii_ok = "ascii" in line
                elif line.startswith("element vertex"):
                    count = int(line.split()[-1])
                elif line == "end_header":
                    break
            if not ascii_ok or count < max(3, int(min_points)):
                return None
            points = np.loadtxt(f, max_rows=count, usecols=(0, 1, 2), ndmin=2)
    except Exception as exc:
        print(f"[SAM3][WARN] could not read {path}: {exc}")
        return None
    if points.shape[0] < max(3, int(min_points)):
        return None
    return points


def _load_mesh_diagonal(output_dir: str):
    """Input-mesh bbox diagonal from the calc manifest (input_mesh.mesh_stats.extents)."""
    path = os.path.join(output_dir, "pipeline_manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None
    extents = (((manifest.get("input_mesh") or {}).get("mesh_stats")) or {}).get("extents")
    if not extents:
        return None
    diagonal = float(np.linalg.norm(np.asarray(extents, dtype=np.float64)))
    return diagonal if diagonal > 0.0 else None


def _object_box_corners(points: np.ndarray, expand_frac: float, mesh_diag=None) -> np.ndarray:
    """Robust 3D bbox of the object cloud as 8 corner points.

    The cloud is best-effort object-only: for off-center placements the calc ROI
    sphere crop may have been skipped entirely, leaving a scene-wide cloud, and a
    plain percentile bbox then projects to (almost) the full image. So the box is
    anchored on the per-axis median (robust to <50% contamination) and, when the
    input-mesh size is known, only points within 0.75x the mesh diagonal of that
    center define the extent - the object cannot outgrow its own mesh diagonal
    (the opened handle stays within it plus the expand margin).
    """
    center = np.median(points, axis=0)
    kept = points
    if mesh_diag is not None and mesh_diag > 0.0:
        for _ in range(2):
            dist = np.linalg.norm(points - center.reshape(1, 3), axis=1)
            inside = dist <= 0.75 * mesh_diag
            if int(np.count_nonzero(inside)) < 10:
                break
            kept = points[inside]
            center = np.median(kept, axis=0)
    lo = np.percentile(kept, 2.0, axis=0)
    hi = np.percentile(kept, 98.0, axis=0)
    base = mesh_diag if (mesh_diag is not None and mesh_diag > 0.0) else float(np.linalg.norm(hi - lo))
    margin = max(expand_frac, 0.0) * max(base, 1.0e-6)
    lo = lo - margin
    hi = hi + margin
    corners = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                corners.append((x, y, z))
    return np.asarray(corners, dtype=np.float64)


def _project_corners_bbox(cam: dict, corners: np.ndarray, width: int, height: int):
    """Project 3D bbox corners into one camera -> pixel bbox + visibility.

    Returns None when every corner is behind the camera, else the same dict
    shape as _project_roi_bbox.
    """
    E = cam["E"]
    K = cam["K"]
    points_cam = corners @ E[:, :3].T + E[:, 3].reshape(1, 3)
    in_front = points_cam[:, 2] > 1.0e-6
    if not bool(np.any(in_front)):
        return None
    visible_points = points_cam[in_front]
    u = K[0, 0] * visible_points[:, 0] / visible_points[:, 2] + K[0, 2]
    v = K[1, 1] * visible_points[:, 1] / visible_points[:, 2] + K[1, 2]
    x0, y0, x1, y1 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
    cx0, cy0 = max(0.0, x0), max(0.0, y0)
    cx1, cy1 = min(float(width), x1), min(float(height), y1)
    full_area = max((x1 - x0) * (y1 - y0), 1.0e-6)
    if cx1 <= cx0 or cy1 <= cy0:
        visible = 0.0
        clipped = None
    else:
        visible = ((cx1 - cx0) * (cy1 - cy0)) / full_area
        clipped = (int(cx0), int(cy0), max(int(cx1), int(cx0) + 1), max(int(cy1), int(cy0) + 1))
    return {"bbox": (x0, y0, x1, y1), "clipped": clipped, "visible_fraction": float(visible)}


def _project_roi_bbox(cam: dict, center: np.ndarray, radius: float, pad: float, width: int, height: int):
    """Project the ROI sphere into one camera -> padded pixel bbox + visibility.

    Returns None when the sphere center is behind (or inside) the camera, else
    {"bbox": unclipped, "clipped": in-image ints, "visible_fraction": float}.
    """
    E = cam["E"]
    K = cam["K"]
    point_cam = E[:, :3] @ center + E[:, 3]
    z = float(point_cam[2])
    if z <= max(radius * 0.5, 1.0e-6):
        return None
    u = float(K[0, 0] * point_cam[0] / z + K[0, 2])
    v = float(K[1, 1] * point_cam[1] / z + K[1, 2])
    rx = pad * float(K[0, 0]) * radius / z
    ry = pad * float(K[1, 1]) * radius / z
    x0, y0, x1, y1 = u - rx, v - ry, u + rx, v + ry
    cx0, cy0 = max(0.0, x0), max(0.0, y0)
    cx1, cy1 = min(float(width), x1), min(float(height), y1)
    full_area = max((x1 - x0) * (y1 - y0), 1.0e-6)
    if cx1 <= cx0 or cy1 <= cy0:
        visible = 0.0
        clipped = None
    else:
        visible = ((cx1 - cx0) * (cy1 - cy0)) / full_area
        clipped = (int(cx0), int(cy0), max(int(cx1), int(cx0) + 1), max(int(cy1), int(cy0) + 1))
    return {"bbox": (x0, y0, x1, y1), "clipped": clipped, "visible_fraction": float(visible)}


def _project_points_to_pixels(cam: dict, points: np.ndarray, width: int, height: int):
    if points is None or points.shape[0] == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    E = cam["E"]
    K = cam["K"]
    points_cam = points @ E[:, :3].T + E[:, 3].reshape(1, 3)
    z = points_cam[:, 2]
    in_front = z > 1.0e-6
    u = np.zeros((points.shape[0],), dtype=np.float64)
    v = np.zeros((points.shape[0],), dtype=np.float64)
    valid_z = np.abs(z) > 1.0e-12
    u[valid_z] = K[0, 0] * points_cam[valid_z, 0] / z[valid_z] + K[0, 2]
    v[valid_z] = K[1, 1] * points_cam[valid_z, 1] / z[valid_z] + K[1, 2]
    finite = np.isfinite(u) & np.isfinite(v)
    xi = np.zeros((points.shape[0],), dtype=np.int64)
    yi = np.zeros((points.shape[0],), dtype=np.int64)
    xi[finite] = np.rint(u[finite]).astype(np.int64)
    yi[finite] = np.rint(v[finite]).astype(np.int64)
    in_image = finite & (xi >= 0) & (xi < int(width)) & (yi >= 0) & (yi < int(height))
    valid = in_front & in_image
    return valid, xi, yi


def _project_points_to_image_mask(points: np.ndarray, cam: dict, width: int, height: int):
    valid, xi, yi = _project_points_to_pixels(cam, points, width, height)
    mask = np.zeros((int(height), int(width)), dtype=bool)
    if bool(np.any(valid)):
        mask[yi[valid], xi[valid]] = True
    return mask, int(np.count_nonzero(valid))


def _project_points_mask_votes(points: np.ndarray, cam: dict, source_mask: np.ndarray):
    height, width = source_mask.shape[:2]
    valid, xi, yi = _project_points_to_pixels(cam, points, width, height)
    hits = np.zeros((points.shape[0],), dtype=bool)
    if bool(np.any(valid)):
        valid_idx = np.flatnonzero(valid)
        hits[valid_idx] = source_mask[yi[valid_idx], xi[valid_idx]]
    return valid, hits


def _dilate_binary_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations <= 0 or not bool(mask.any()):
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    for _ in range(iterations):
        image = image.filter(ImageFilter.MaxFilter(3))
    return np.asarray(image) > 127


def _clip_mask_to_bbox(mask: np.ndarray, clipped_bbox) -> np.ndarray:
    if clipped_bbox is None:
        return mask
    x0, y0, x1, y1 = clipped_bbox
    bounded = np.zeros_like(mask, dtype=bool)
    bounded[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return bounded


def _bbox_from_mask(mask: np.ndarray, expand_frac: float = 0.0):
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    height, width = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x1 - x0, y1 - y0) * max(0.0, float(expand_frac))))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(int(width), x1 + pad)
    y1 = min(int(height), y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _hull_guided_predict(processor, state_obj, fallback_prompt: str, clipped_bbox, image: Image.Image):
    """Retry with fallback text plus a tight projected-hull box."""
    x0, y0, x1, y1 = clipped_bbox
    box = [
        ((x0 + x1) / 2.0) / float(image.width),
        ((y0 + y1) / 2.0) / float(image.height),
        (x1 - x0) / float(image.width),
        (y1 - y0) / float(image.height),
    ]
    processor.set_text_prompt(prompt=fallback_prompt, state=state_obj)
    output = processor.add_geometric_prompt(box=box, label=True, state=state_obj)
    return _normalize_instances(output, image)


def _mask_guide_overlap(mask: np.ndarray, guide_mask: np.ndarray) -> tuple:
    mask_area = float(np.count_nonzero(mask))
    guide_area = float(np.count_nonzero(guide_mask))
    if mask_area <= 0.0 or guide_area <= 0.0:
        return 0.0, 0.0
    intersection = float(np.count_nonzero(np.asarray(mask, dtype=bool) & np.asarray(guide_mask, dtype=bool)))
    return intersection / mask_area, intersection / guide_area


def _select_hull_guided_mask(masks, scores, args, guide_mask: np.ndarray):
    """Pick a SAM retry candidate using the projected hull only as a guide.

    The candidate must meaningfully overlap the guide, and it must cover enough
    of the guide. This keeps the geometric prompt from accepting large floor or
    background components that happen to touch the prompt box.
    """
    order = np.argsort(-scores)
    eligible = []
    best_diag = {"overlap": 0.0, "guide_coverage": 0.0, "candidate_count": 0}
    min_overlap = float(getattr(args, "hull_guided_min_overlap", 0.30))
    min_coverage = float(getattr(args, "hull_guided_min_guide_coverage", 0.25))
    for i in order:
        i = int(i)
        if float(scores[i]) < float(args.min_score):
            continue
        overlap, coverage = _mask_guide_overlap(masks[i], guide_mask)
        if overlap > best_diag["overlap"] or coverage > best_diag["guide_coverage"]:
            best_diag = {
                "overlap": float(overlap),
                "guide_coverage": float(coverage),
                "candidate_count": int(scores.size),
            }
        if overlap >= min_overlap and coverage >= min_coverage:
            eligible.append((i, overlap, coverage))
    if not eligible:
        return None, 0.0, int(scores.size), 0, best_diag
    if args.instance_mode == "union" and len(eligible) > 1:
        ids = [item[0] for item in eligible]
        union = np.any(masks[ids], axis=0)
        overlap, coverage = _mask_guide_overlap(union, guide_mask)
        diag = {"overlap": float(overlap), "guide_coverage": float(coverage), "candidate_count": int(scores.size)}
        return union, float(scores[eligible[0][0]]), int(scores.size), len(eligible), diag
    i, overlap, coverage = eligible[0]
    diag = {"overlap": float(overlap), "guide_coverage": float(coverage), "candidate_count": int(scores.size)}
    return masks[i], float(scores[i]), int(scores.size), len(eligible), diag


def _build_projected_hull_guides(
    cams: dict,
    source_masks: dict,
    pending_entries: list,
    box_corners,
    args: argparse.Namespace,
) -> tuple:
    """Build projected-hull guides for concept-miss views."""
    diagnostics = {
        "enabled": True,
        "status": "skipped",
        "source_mask_count": 0,
        "occupied_voxels": 0,
        "grid_shape": None,
        "rejections": [],
        "rejections_by_serial": {},
    }
    if box_corners is None:
        diagnostics["status"] = "no_object_box"
        return {}, diagnostics
    source_items = [
        (serial, np.asarray(mask, dtype=bool))
        for serial, mask in source_masks.items()
        if serial in cams and mask is not None and bool(np.asarray(mask, dtype=bool).any())
    ]
    diagnostics["source_mask_count"] = len(source_items)
    min_sources = max(1, int(getattr(args, "hull_projection_min_source_masks", 3)))
    if len(source_items) < min_sources:
        diagnostics["status"] = "too_few_source_masks"
        return {}, diagnostics
    if not pending_entries:
        diagnostics["status"] = "no_pending_views"
        return {}, diagnostics

    corners = np.asarray(box_corners, dtype=np.float64).reshape(-1, 3)
    lo = corners.min(axis=0)
    hi = corners.max(axis=0)
    extents = np.maximum(hi - lo, 1.0e-6)
    longest = float(np.max(extents))
    res = max(8, int(getattr(args, "hull_projection_grid_resolution", 64)))
    counts = np.maximum(4, np.ceil(res * extents / max(longest, 1.0e-6)).astype(np.int64))
    diagnostics["grid_shape"] = [int(v) for v in counts]

    axes = [np.linspace(lo[i], hi[i], int(counts[i])) for i in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    voxels = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    if voxels.shape[0] == 0:
        diagnostics["status"] = "empty_grid"
        return {}, diagnostics

    votes = np.zeros((voxels.shape[0],), dtype=np.float64)
    opportunities = np.zeros((voxels.shape[0],), dtype=np.float64)
    for serial, mask in source_items:
        visible, hits = _project_points_mask_votes(voxels, cams[serial], mask)
        opportunities[visible] += 1.0
        votes[hits] += 1.0

    min_opportunities = max(1, int(getattr(args, "hull_projection_min_opportunities", 2)))
    min_view_fraction = float(getattr(args, "hull_projection_min_view_fraction", 0.55))
    occupied = (opportunities >= float(min_opportunities)) & (
        votes >= min_view_fraction * np.maximum(opportunities, 1.0)
    )
    occupied_count = int(np.count_nonzero(occupied))
    diagnostics["occupied_voxels"] = occupied_count
    if occupied_count == 0:
        diagnostics["status"] = "empty_hull"
        return {}, diagnostics

    occupied_voxels = voxels[occupied]
    projected = {}
    rejections_by_serial = diagnostics["rejections_by_serial"]
    min_fraction = float(getattr(args, "hull_projection_min_mask_fraction", 0.0005))
    max_fraction = float(getattr(args, "hull_projection_max_mask_fraction", 0.60))
    for entry in pending_entries:
        serial = entry["serial"]
        if serial not in cams:
            item = {"serial": serial, "reason": "no_camera"}
            diagnostics["rejections"].append(item)
            rejections_by_serial[serial] = item
            continue
        mask, projected_points = _project_points_to_image_mask(
            occupied_voxels, cams[serial], int(entry["width"]), int(entry["height"])
        )
        if projected_points <= 0:
            item = {"serial": serial, "reason": "no_projected_voxels"}
            diagnostics["rejections"].append(item)
            rejections_by_serial[serial] = item
            continue
        mask = _dilate_binary_mask(mask, int(getattr(args, "hull_projection_dilate", 2)))
        mask = _clip_mask_to_bbox(mask, entry.get("clipped"))
        mask, filled = _fill_small_enclosed_holes(mask, float(getattr(args, "fill_holes_max_frac", 0.0)))
        fraction = float(np.mean(mask)) if mask.size else 0.0
        if fraction < min_fraction:
            item = {"serial": serial, "reason": "mask_too_small", "mask_fraction": fraction}
            diagnostics["rejections"].append(item)
            rejections_by_serial[serial] = item
            continue
        if fraction > max_fraction:
            item = {"serial": serial, "reason": "mask_too_large", "mask_fraction": fraction}
            diagnostics["rejections"].append(item)
            rejections_by_serial[serial] = item
            continue
        projected[serial] = {
            "mask": mask,
            "filled": int(filled) if filled is not None else 0,
            "mask_fraction": fraction,
            "projected_voxels": int(projected_points),
        }

    diagnostics["status"] = "ok" if projected else "no_projected_masks"
    diagnostics["projected_count"] = len(projected)
    return projected, diagnostics


def _load_sam3(repo_root: str, checkpoint, device: str):
    """Build the SAM3 image model + processor. All SAM3 API use lives here and
    in the two predict helpers below.

    API per the public repo (facebookresearch/sam3): build_sam3_image_model() ->
    model, Sam3Processor(model).set_image(pil) -> state,
    .set_text_prompt(prompt, state) / .add_geometric_prompt(box, label, state)
    -> {"masks", "boxes", "scores"}; boxes for the geometric prompt are
    normalized [center_x, center_y, width, height]. Fallback retries use a
    separate text prompt plus a projected-hull box on a fresh image state.
    """
    expanded = os.path.expanduser(repo_root) if repo_root else None
    if expanded and os.path.isdir(expanded) and expanded not in sys.path:
        sys.path.insert(0, expanded)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if checkpoint:
        ckpt = os.path.expanduser(checkpoint)
        try:
            model = build_sam3_image_model(checkpoint_path=ckpt)
        except TypeError:
            model = build_sam3_image_model(ckpt)
    else:
        model = build_sam3_image_model()
    if device:
        try:
            model = model.to(device)
        except Exception as exc:  # keep going on the builder's own placement
            print(f"[SAM3][WARN] could not move the model to '{device}': {exc}")
    return Sam3Processor(model)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_instances(output: dict, image: Image.Image):
    """SAM3 output dict -> (masks [N,H,W] bool, scores [N] float), N possibly 0."""
    masks = _to_numpy(output["masks"])
    scores = _to_numpy(output["scores"]).reshape(-1).astype(np.float64)
    if masks.size == 0 or scores.size == 0:
        return np.zeros((0, image.height, image.width), dtype=bool), np.zeros((0,), dtype=np.float64)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.dtype != bool:
        # float outputs are probabilities (0..1) or logits (signed); pick the threshold accordingly
        threshold = 0.0 if float(masks.min()) < 0.0 else 0.5
        masks = masks > threshold
    if masks.shape[1:] != (image.height, image.width):
        resized = []
        for mask in masks:
            resized.append(
                np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (image.width, image.height), Image.NEAREST
                    )
                )
                > 127
            )
        masks = np.stack(resized, axis=0)
    return masks, scores


def _pcs_predict(processor, state_obj, prompt: str, image: Image.Image):
    output = processor.set_text_prompt(prompt=prompt, state=state_obj)
    return _normalize_instances(output, image)


def _roi_inside_fraction(mask: np.ndarray, clipped_bbox) -> float:
    total = float(mask.sum())
    if total <= 0.0:
        return 0.0
    x0, y0, x1, y1 = clipped_bbox
    return float(mask[y0:y1, x0:x1].sum()) / total


def _select_mask(masks, scores, args, clipped_bbox=None):
    """Pick the object mask from SAM instances.

    Candidates need score >= --min-score and, when an ROI bbox is available,
    >= --roi-min-overlap of their pixels inside it (drops rig/cloth instances).
    Returns (mask, score, instance_count, candidate_count).
    """
    order = np.argsort(-scores)
    eligible = [int(i) for i in order if float(scores[i]) >= args.min_score]
    if clipped_bbox is not None:
        eligible = [
            i for i in eligible if _roi_inside_fraction(masks[i], clipped_bbox) >= args.roi_min_overlap
        ]
    if not eligible:
        return None, 0.0, int(scores.size), 0
    if args.instance_mode == "union" and len(eligible) > 1:
        return np.any(masks[eligible], axis=0), float(scores[eligible[0]]), int(scores.size), len(eligible)
    return masks[eligible[0]], float(scores[eligible[0]]), int(scores.size), len(eligible)


def _fill_small_enclosed_holes(mask: np.ndarray, max_frac: float):
    """Fill background components that touch no image border and are small
    relative to the mask area. Returns (mask, filled_pixel_count); -1 when no
    connected-component backend (cv2/scipy) is available."""
    if max_frac <= 0.0 or not bool(mask.any()):
        return mask, 0
    height, width = mask.shape
    limit = max_frac * float(mask.sum())
    try:
        import cv2

        inverted = (~mask).astype(np.uint8)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
        out = mask.copy()
        filled = 0
        for component_id in range(1, count):
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
            if not touches_border and area <= limit:
                out[labels == component_id] = True
                filled += area
        return out, filled
    except ImportError:
        pass
    try:
        from scipy import ndimage

        labels, count = ndimage.label(~mask)
        out = mask.copy()
        filled = 0
        border_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :]))
        border_labels |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
        for component_id in range(1, count + 1):
            if component_id in border_labels:
                continue
            component = labels == component_id
            area = int(component.sum())
            if area <= limit:
                out[component] = True
                filled += area
        return out, filled
    except ImportError:
        if not _HOLE_FILL_WARNED[0]:
            print("[SAM3][WARN] neither cv2 nor scipy available; enclosed-hole filling skipped.")
            _HOLE_FILL_WARNED[0] = True
        return mask, -1


def _write_mask_png(path: str, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8)) * 255).save(path)


def _write_overlay(path: str, image_np: np.ndarray, mask: np.ndarray, roi_bbox=None) -> None:
    overlay = image_np.copy()
    tint = np.array([255, 200, 0], dtype=np.float64)
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * tint).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    if roi_bbox is not None:
        draw = ImageDraw.Draw(overlay_img)
        draw.rectangle(list(roi_bbox), outline=(255, 0, 255), width=3)
    panel = np.concatenate([image_np, np.asarray(overlay_img)], axis=1)
    Image.fromarray(panel).save(path, quality=90)


def _clean_state_mask_dir(state_mask_dir: str) -> None:
    """Remove masks/overlays from an earlier provider so runs never mix."""
    if not os.path.isdir(state_mask_dir):
        return
    for entry in os.listdir(state_mask_dir):
        if entry.endswith(".png") or (entry.startswith("overlay_") and entry.endswith(".jpg")):
            os.remove(os.path.join(state_mask_dir, entry))


def main() -> None:
    args = parse_args()

    session_dir = os.path.abspath(os.path.expanduser(args.session_dir))
    if not os.path.isdir(session_dir):
        raise SystemExit(f"Session directory not found: {session_dir}")
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or os.path.join(session_dir, "processed")))
    masks_root = os.path.join(output_dir, "object_masks")

    object_name = args.object_name or os.path.basename(os.path.dirname(session_dir))
    prompt, prompt_source = _resolve_prompt(args, object_name)

    states_filter = None
    if args.states:
        states_filter = {token.strip() for token in str(args.states).split(",") if token.strip()}
    if getattr(args, "auto_undistort", True):
        _ensure_undistorted(session_dir, args)
    states = _discover_states(session_dir, args.image_dirname, states_filter)
    if not states:
        raise SystemExit(
            f"No states with a '{args.image_dirname}/' folder under {session_dir}. "
            "Expected raw captures under <state>/raw/images with cam_param, or pre-undistorted "
            "images (with --no-auto-undistort)."
        )

    roi_spheres = {}
    mesh_diag = None
    if args.roi_source != "off":
        roi_spheres = _load_roi_spheres(output_dir)
        if not roi_spheres:
            print(
                "[SAM3][WARN] no ROI spheres in pipeline_manifest.json; sphere visibility fallback "
                "unavailable (per-state object-cloud boxes may still apply)."
            )
        mesh_diag = _load_mesh_diagonal(output_dir)
        if mesh_diag is None:
            print(
                "[SAM3][WARN] no input-mesh extents in pipeline_manifest.json; the object box falls "
                "back to the cloud spread and a contaminated cloud will inflate it."
            )
        else:
            print(f"[SAM3] mesh diagonal = {mesh_diag:.3f} (object-box size anchor)")
    print(f"[SAM3] session={session_dir}")
    print(f"[SAM3] output={masks_root}")
    print(
        f"[SAM3] prompt='{prompt}' (source={prompt_source}) states={len(states)} "
        f"fallback_prompt='{args.fallback_prompt}' roi_sphere_states={len(roi_spheres)}"
    )

    import torch

    processor = _load_sam3(args.sam3_repo_root, args.checkpoint, args.device)
    if args.confidence_threshold is not None:
        processor.set_confidence_threshold(float(args.confidence_threshold))

    csv_rows = []
    manifest_states = {}
    total_masks = 0
    total_skipped = 0
    total_hull_guided = 0
    total_out_of_view = 0
    with torch.inference_mode():
        for state in states:
            state_id = state["state_id"]
            state_mask_dir = os.path.join(masks_root, state_id)
            os.makedirs(state_mask_dir, exist_ok=True)
            _clean_state_mask_dir(state_mask_dir)

            sphere = roi_spheres.get(state_id)
            box_corners = None
            if args.roi_source != "off":
                object_points = _load_object_points(output_dir, state_id, args.object_box_min_points)
                if object_points is not None:
                    box_corners = _object_box_corners(object_points, args.object_box_expand, mesh_diag)
                elif sphere is not None:
                    print(
                        f"[SAM3][WARN] {state_id}: no object-cloud box "
                        f"(multiview/{state_id}/object_points_roi.ply); loose ROI sphere used for "
                        "visibility only, geometric fallback disabled for this state."
                    )
            roi_active = box_corners is not None or sphere is not None
            cams = _load_state_cameras(state["state_dir"]) if roi_active else {}

            mask_count = 0
            hull_guided_count = 0
            out_of_view_count = 0
            score_sum = 0.0
            score_count = 0
            state_rows = []
            source_masks = {}
            pending_entries = []
            for serial_index, serial in enumerate(state["serials"]):
                image_path = _find_image_path(state["images_dir"], serial)
                if image_path is None:
                    continue
                image = Image.open(image_path).convert("RGB")

                roi = None
                box_kind = ""
                roi_available = serial in cams and roi_active
                if serial in cams and box_corners is not None:
                    roi = _project_corners_bbox(cams[serial], box_corners, image.width, image.height)
                    box_kind = "object"
                elif serial in cams and sphere is not None:
                    roi = _project_roi_bbox(
                        cams[serial], sphere[0], sphere[1], args.roi_pad, image.width, image.height
                    )
                    box_kind = "sphere"
                # roi is None with roi_available means the box/sphere is behind the camera.
                roi_visible = roi["visible_fraction"] if roi is not None else (0.0 if roi_available else None)
                clipped = roi["clipped"] if roi is not None else None

                row = {
                    "state_id": state_id,
                    "serial": serial,
                    "provider": "sam3",
                    "prompt_mode": "",
                    "score": "",
                    "instance_count": "",
                    "box_kind": box_kind,
                    "roi_visible_fraction": ("" if roi_visible is None else f"{roi_visible:.3f}"),
                    "mask_fraction": "",
                    "filled_hole_fraction": "",
                    "fallback_source_masks": "",
                    "hull_occupied_voxels": "",
                    "hull_projected_voxels": "",
                    "hull_guide_mask_fraction": "",
                    "fallback_prompt": "",
                    "hull_guided_overlap": "",
                    "hull_guided_coverage": "",
                    "hull_rejection_reason": "",
                    "status": "",
                    "mask_path": "",
                }

                if roi_available and (clipped is None or roi_visible < args.roi_min_visible_frac):
                    # The working volume does not project into this camera (outside the
                    # image or behind it): the object cannot be visible, so running SAM
                    # would only invite false positives.
                    row["status"] = "object_out_of_view"
                    out_of_view_count += 1
                    total_out_of_view += 1
                    state_rows.append(row)
                    continue

                state_obj = processor.set_image(image)
                masks, scores = _pcs_predict(processor, state_obj, prompt, image)
                mask, score, instance_count, candidates = _select_mask(masks, scores, args, clipped)
                prompt_mode = "pcs"

                row["instance_count"] = instance_count
                if mask is None or not bool(mask.any()):
                    row["status"] = "no_detection" if instance_count == 0 else "no_roi_candidate"
                    state_rows.append(row)
                    pending_entries.append(
                        {
                            "serial": serial,
                            "serial_index": serial_index,
                            "image_path": image_path,
                            "width": image.width,
                            "height": image.height,
                            "clipped": clipped,
                            "row": row,
                        }
                    )
                    continue

                mask, filled = _fill_small_enclosed_holes(mask, args.fill_holes_max_frac)
                mask_path = os.path.join(state_mask_dir, f"{serial}.png")
                _write_mask_png(mask_path, mask)
                mask_area = float(mask.sum())
                row["prompt_mode"] = prompt_mode
                row["score"] = f"{score:.4f}"
                row["mask_fraction"] = f"{mask_area / float(mask.size):.6f}"
                row["filled_hole_fraction"] = f"{(filled / mask_area if filled > 0 else 0.0):.6f}"
                row["status"] = "ok"
                row["mask_path"] = mask_path
                state_rows.append(row)
                mask_count += 1
                score_sum += score
                score_count += 1
                total_masks += 1
                source_masks[serial] = mask
                if serial_index < max(0, int(args.overlay_cameras)):
                    _write_overlay(
                        os.path.join(state_mask_dir, f"overlay_{serial}.jpg"),
                        np.asarray(image),
                        mask,
                        roi_bbox=clipped,
                    )

            projected_guides, hull_diag = _build_projected_hull_guides(
                cams, source_masks, pending_entries, box_corners, args
            )
            for entry in pending_entries:
                row = entry["row"]
                serial = entry["serial"]
                projected_info = projected_guides.get(serial)
                if projected_info is None:
                    total_skipped += 1
                    reason = hull_diag.get("status", "unavailable")
                    per_view = (hull_diag.get("rejections_by_serial") or {}).get(serial)
                    if per_view is not None:
                        reason = per_view.get("reason", reason)
                    row["hull_rejection_reason"] = reason
                    print(f"[SAM3][WARN] {state_id}/{serial}: {row['status']} (hull fallback={reason})")
                    continue

                guide_mask = np.asarray(projected_info["mask"], dtype=bool)
                guide_fraction = float(np.mean(guide_mask)) if guide_mask.size else 0.0
                row["fallback_source_masks"] = int(hull_diag.get("source_mask_count", 0))
                row["hull_occupied_voxels"] = int(hull_diag.get("occupied_voxels", 0))
                row["hull_projected_voxels"] = int(projected_info.get("projected_voxels", 0))
                row["hull_guide_mask_fraction"] = f"{guide_fraction:.6f}"

                mask = None
                score = 0.0
                instance_count = ""
                filled = int(projected_info.get("filled", 0))
                prompt_mode = ""
                retry_diag = {}
                prompt_bbox = _bbox_from_mask(
                    guide_mask,
                    float(getattr(args, "hull_guided_box_expand", 0.10)),
                )
                if prompt_bbox is None:
                    row["hull_rejection_reason"] = "empty_guide_bbox"
                else:
                    image = Image.open(entry["image_path"]).convert("RGB")
                    state_obj = processor.set_image(image)
                    fallback_prompt = str(getattr(args, "fallback_prompt", "foreground object"))
                    masks, scores = _hull_guided_predict(
                        processor, state_obj, fallback_prompt, prompt_bbox, image
                    )
                    mask, score, instance_count, candidates, retry_diag = _select_hull_guided_mask(
                        masks, scores, args, guide_mask
                    )
                    row["instance_count"] = instance_count
                    row["fallback_prompt"] = fallback_prompt
                    row["hull_guided_overlap"] = f"{float(retry_diag.get('overlap', 0.0)):.3f}"
                    row["hull_guided_coverage"] = f"{float(retry_diag.get('guide_coverage', 0.0)):.3f}"
                    if mask is not None and bool(mask.any()):
                        prompt_mode = "hull_guided_sam3"
                    else:
                        row["hull_rejection_reason"] = "hull_guided_no_candidate"

                if mask is None or not bool(mask.any()):
                    total_skipped += 1
                    reason = row["hull_rejection_reason"] or "hull_guided_failed"
                    print(f"[SAM3][WARN] {state_id}/{serial}: {row['status']} (hull fallback={reason})")
                    continue

                mask, filled2 = _fill_small_enclosed_holes(mask, args.fill_holes_max_frac)
                if filled2 is not None and int(filled2) > 0:
                    filled = int(filled2)
                mask_path = os.path.join(state_mask_dir, f"{serial}.png")
                _write_mask_png(mask_path, mask)
                mask_area = float(mask.sum())
                row["prompt_mode"] = prompt_mode
                row["score"] = f"{float(score):.4f}"
                row["mask_fraction"] = f"{mask_area / float(mask.size):.6f}"
                row["filled_hole_fraction"] = f"{(filled / mask_area if filled > 0 else 0.0):.6f}"
                row["status"] = "ok"
                row["mask_path"] = mask_path
                mask_count += 1
                total_masks += 1
                hull_guided_count += 1
                total_hull_guided += 1
                if int(entry["serial_index"]) < max(0, int(args.overlay_cameras)):
                    image_np = np.asarray(Image.open(entry["image_path"]).convert("RGB"))
                    _write_overlay(
                        os.path.join(state_mask_dir, f"overlay_{serial}.jpg"),
                        image_np,
                        mask,
                        roi_bbox=prompt_bbox or entry.get("clipped"),
                    )

            csv_rows.extend(state_rows)
            box_extent = None
            if box_corners is not None:
                extent_vec = box_corners.max(axis=0) - box_corners.min(axis=0)
                box_extent = [round(float(v), 4) for v in extent_vec]
            manifest_states[state_id] = {
                "images": len(state["serials"]),
                "masks": mask_count,
                "hull_guided": hull_guided_count,
                "out_of_view": out_of_view_count,
                "box_kind": ("object" if box_corners is not None else ("sphere" if sphere is not None else "none")),
                "box_extent": box_extent,
                "mean_score": (score_sum / score_count) if score_count else None,
                "hull_projection": hull_diag,
            }
            box_note = manifest_states[state_id]["box_kind"]
            if box_extent is not None:
                # World units (calibration frame); sanity check against the mesh size.
                box_note += "(" + "x".join(f"{v:.2f}" for v in box_extent) + ")"
            print(
                f"[SAM3] {state_id}: masks={mask_count}/{len(state['serials'])} "
                f"hull_guided={hull_guided_count} out_of_view={out_of_view_count} box={box_note}"
            )

    os.makedirs(masks_root, exist_ok=True)
    stats_path = os.path.join(masks_root, "mask_stats.csv")
    fieldnames = [
        "state_id",
        "serial",
        "provider",
        "prompt_mode",
        "score",
        "instance_count",
        "box_kind",
        "roi_visible_fraction",
        "mask_fraction",
        "filled_hole_fraction",
        "fallback_source_masks",
        "hull_occupied_voxels",
        "hull_projected_voxels",
        "hull_guide_mask_fraction",
        "fallback_prompt",
        "hull_guided_overlap",
        "hull_guided_coverage",
        "hull_rejection_reason",
        "status",
        "mask_path",
    ]
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    manifest = {
        "schema": "paradex.articulated_object.masks_manifest.v0",
        "provider": "sam3",
        "prompt": prompt,
        "prompt_source": prompt_source,
        "object_name": object_name,
        "checkpoint": args.checkpoint,
        "sam3_repo_root": args.sam3_repo_root,
        "device": args.device,
        "min_score": args.min_score,
        "confidence_threshold": args.confidence_threshold,
        "instance_mode": args.instance_mode,
        "roi": {
            "source": args.roi_source,
            "sphere_states": len(roi_spheres),
            "object_box_expand": args.object_box_expand,
            "object_box_min_points": args.object_box_min_points,
            "pad": args.roi_pad,
            "min_visible_frac": args.roi_min_visible_frac,
            "min_overlap": args.roi_min_overlap,
        },
        "hull_guided_fallback": {
            "fallback_prompt": args.fallback_prompt,
            "grid_resolution": args.hull_projection_grid_resolution,
            "min_source_masks": args.hull_projection_min_source_masks,
            "min_opportunities": args.hull_projection_min_opportunities,
            "min_view_fraction": args.hull_projection_min_view_fraction,
            "dilate": args.hull_projection_dilate,
            "min_mask_fraction": args.hull_projection_min_mask_fraction,
            "max_mask_fraction": args.hull_projection_max_mask_fraction,
            "guided_box_expand": args.hull_guided_box_expand,
            "guided_min_overlap": args.hull_guided_min_overlap,
            "guided_min_guide_coverage": args.hull_guided_min_guide_coverage,
        },
        "fill_holes_max_frac": args.fill_holes_max_frac,
        "image_dirname": args.image_dirname,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "states": manifest_states,
        "totals": {
            "states": len(states),
            "masks": total_masks,
            "skipped": total_skipped,
            "hull_guided": total_hull_guided,
            "out_of_view": total_out_of_view,
        },
    }
    manifest_path = os.path.join(masks_root, "masks_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        f"[SAM3] wrote {total_masks} masks (hull_guided={total_hull_guided}, "
        f"out_of_view={total_out_of_view}, "
        f"skipped={total_skipped}) -> {masks_root}"
    )
    print(f"[SAM3] manifest: {manifest_path}")
    print("[SAM3] consume with: calculate.py ... --object-mask-source external (or auto)")


if __name__ == "__main__":
    main()
