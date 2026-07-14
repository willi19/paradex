#!/usr/bin/env python3
"""Standalone image preprocessing for the articulated-object pipeline.

Currently this does one thing: undistort the raw captured images into each
state's calibrated ``images/`` folder, using only the per-state ``cam_param``
that the capture step already wrote (``original_intrinsics``, ``dist_params``,
``intrinsics_undistort``). It is dependency-light (cv2 + numpy, both imported
lazily) and free of paradex imports, so it runs standalone in either the calc
(Python 3.8) or the SAM3 (Python 3.12) environment:

    python preprocess.py --session-dir /path/to/capture/<object>/<session>

``generate_masks_sam3.py`` calls ``undistort_session`` on demand when a state
has raw images but no undistorted ones, so mask generation no longer depends on
a prior calc run.

Room to grow (not implemented yet): the working-volume center (camera optical
axis convergence) for a centered-object capture, which would replace the ROI
that ``generate_masks_sam3.py`` currently reads from the calc manifest. That is
why this module is named ``preprocess`` rather than ``undistort``.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _imread_color(path: str):
    """Read an image as BGR uint8, robust to non-ASCII paths (no cv2.imread)."""
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_png(path: str, image) -> None:
    """Write a PNG, robust to non-ASCII paths (no cv2.imwrite)."""
    import cv2

    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"failed to encode image for {path}")
    buf.tofile(path)


def _list_serials(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        return []
    serials = []
    for name in os.listdir(images_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTENSIONS:
            serials.append(stem)
    return sorted(set(serials))


def _load_intrinsics(state_dir: str) -> dict:
    with open(os.path.join(state_dir, "cam_param", "intrinsics.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def undistort_state(state_dir: str, force: bool = False, image_dirname: str = "images") -> List[str]:
    """Undistort ``<state_dir>/raw/images/*`` into ``<state_dir>/<image_dirname>/*.png``.

    Uses ``cv2.undistort`` with the per-camera ``original_intrinsics`` /
    ``dist_params`` / ``intrinsics_undistort`` from ``cam_param/intrinsics.json``.
    Existing outputs are kept unless ``force``. Cameras without intrinsics are
    skipped with a warning. Returns the serials present in the output folder.
    """
    import cv2
    import numpy as np

    raw_images_dir = os.path.join(state_dir, "raw", "images")
    out_dir = os.path.join(state_dir, image_dirname)
    if not os.path.isdir(raw_images_dir):
        return _list_serials(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        intrinsics = _load_intrinsics(state_dir)
    except FileNotFoundError:
        print(f"[PREP][WARN] {state_dir}: cam_param/intrinsics.json missing; cannot undistort.")
        return _list_serials(out_dir)

    for name in sorted(os.listdir(raw_images_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        out_path = os.path.join(out_dir, f"{stem}.png")
        if os.path.exists(out_path) and not force:
            continue
        if stem not in intrinsics:
            print(f"[PREP][WARN] {state_dir}: no intrinsics for camera {stem}, skipping.")
            continue
        image = _imread_color(os.path.join(raw_images_dir, name))
        if image is None:
            print(f"[PREP][WARN] {state_dir}: failed to read {name}")
            continue
        intr = intrinsics[stem]
        original_k = np.asarray(intr["original_intrinsics"], dtype=np.float64).reshape(3, 3)
        new_k = np.asarray(intr["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(intr["dist_params"], dtype=np.float64).reshape(-1)
        _imwrite_png(out_path, cv2.undistort(image, original_k, dist, None, new_k))

    return _list_serials(out_dir)


def ensure_state_undistorted(state_dir: str, image_dirname: str = "images", force: bool = False) -> List[str]:
    """Undistort a single state only when its output folder is empty (or ``force``)."""
    existing = _list_serials(os.path.join(state_dir, image_dirname))
    if existing and not force:
        return existing
    return undistort_state(state_dir, force=force, image_dirname=image_dirname)


def _discover_state_dirs(session_dir: str, states_filter: Optional[set]) -> List[str]:
    dirs = []
    for name in sorted(os.listdir(session_dir)):
        state_dir = os.path.join(session_dir, name)
        if not os.path.isdir(os.path.join(state_dir, "raw", "images")):
            continue
        if states_filter is not None and name not in states_filter:
            continue
        dirs.append(state_dir)
    return dirs


def undistort_session(
    session_dir: str,
    states=None,
    force: bool = False,
    image_dirname: str = "images",
) -> Dict[str, List[str]]:
    """Undistort every capture state under a session. Returns ``{state_id: serials}``.

    A state whose ``images/`` folder is already populated is left untouched
    (per-image existence check), so calling this on an already-processed session
    is cheap. Only states that have a ``raw/images`` folder are considered.
    """
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    states_filter = None
    if states:
        if isinstance(states, str):
            states = [token.strip() for token in states.split(",") if token.strip()]
        states_filter = set(states)
    result: Dict[str, List[str]] = {}
    for state_dir in _discover_state_dirs(session_dir, states_filter):
        state_id = os.path.basename(state_dir)
        result[state_id] = undistort_state(state_dir, force=force, image_dirname=image_dirname)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Undistort raw multiview captures into the calibrated <state>/images folder.",
    )
    parser.add_argument(
        "--session-dir",
        required=True,
        help="Capture session directory containing <state>/raw/images and <state>/cam_param.",
    )
    parser.add_argument(
        "--states",
        default=None,
        help="Comma-separated state ids to process (default: every state in the session).",
    )
    parser.add_argument(
        "--image-dirname",
        default="images",
        help="Per-state output folder for the undistorted images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-undistort even when the output images already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = undistort_session(
        args.session_dir, states=args.states, force=args.force, image_dirname=args.image_dirname
    )
    total = sum(len(serials) for serials in result.values())
    print(f"[PREP] undistorted {len(result)} states, {total} images -> <state>/{args.image_dirname}/")


if __name__ == "__main__":
    main()
