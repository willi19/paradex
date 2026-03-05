import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

from paradex.visualization.visualizer.viser import ViserViewer


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "1", "y", "on"):
        return True
    if v in ("no", "false", "f", "0", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{v}'")


def parse_camera_ids(raw: Optional[str]) -> Optional[Set[str]]:
    if raw is None:
        return None
    values = [x.strip() for x in raw.split(",")]
    values = [x for x in values if x]
    return set(values)


def resolve_camparam_dir(episode_root: Optional[str], camparam_dir: Optional[str]) -> Path:
    if camparam_dir:
        return Path(camparam_dir).expanduser().resolve()
    if episode_root:
        return (Path(episode_root).expanduser().resolve() / "cam_param")
    raise ValueError("Either --episode-root or --camparam-dir is required.")


def resolve_c2r_path(episode_root: Optional[str], c2r_path: Optional[str]) -> Optional[Path]:
    if c2r_path:
        return Path(c2r_path).expanduser().resolve()
    if episode_root:
        candidate = Path(episode_root).expanduser().resolve() / "C2R.npy"
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_c2r(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path, allow_pickle=True), dtype=float)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=float)
        out[:3, :] = arr
        return out
    if arr.size == 12:
        out = np.eye(4, dtype=float)
        out[:3, :] = arr.reshape(3, 4)
        return out
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"Unsupported C2R shape: {arr.shape} (expected 4x4 or 3x4)")


def parse_cam_from_world(extrinsic_value) -> np.ndarray:
    arr = np.asarray(extrinsic_value, dtype=float)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=float)
        out[:3, :] = arr
        return out
    if arr.shape == (4, 4):
        return arr
    if arr.size == 12:
        out = np.eye(4, dtype=float)
        out[:3, :] = arr.reshape(3, 4)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {arr.shape} (expected 3x4 or 4x4)")


def validate_rotation_matrix(R: np.ndarray, det_tol: float = 1e-2, orth_tol: float = 1e-2) -> Tuple[bool, str]:
    det = np.linalg.det(R)
    orth_err = np.linalg.norm(R.T @ R - np.eye(3), ord="fro")
    if abs(det - 1.0) > det_tol:
        return False, f"det(R)={det:.6f}"
    if orth_err > orth_tol:
        return False, f"orth_error={orth_err:.6e}"
    return True, "ok"


def orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_fixed = U @ Vt
    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1.0
        R_fixed = U @ Vt
    return R_fixed


def build_intrinsic_for_viewer(serial: str, payload: Dict) -> Dict:
    if "intrinsics_undistort" in payload:
        K = np.asarray(payload["intrinsics_undistort"], dtype=float)
    elif "original_intrinsics" in payload:
        K = np.asarray(payload["original_intrinsics"], dtype=float)
    else:
        raise ValueError(f"{serial}: missing 'intrinsics_undistort'/'original_intrinsics'")

    if K.shape != (3, 3):
        raise ValueError(f"{serial}: intrinsic matrix shape {K.shape} is not 3x3")

    width = payload.get("width")
    height = payload.get("height")
    if width is None or height is None:
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        width = int(round(cx * 2.0))
        height = int(round(cy * 2.0))
        print(f"[WARN] {serial}: missing width/height, fallback to width={width}, height={height}")

    return {
        "intrinsics_undistort": K.tolist(),
        "width": int(width),
        "height": int(height),
    }


def color_from_name(name: str) -> Tuple[int, int, int]:
    seed = int(np.frombuffer(name.encode("utf-8"), dtype=np.uint8).sum())
    rng = np.random.default_rng(seed)
    return tuple(int(x) for x in rng.integers(low=70, high=255, size=3))


def iter_target_serials(
    intrinsics: Dict,
    extrinsics: Dict,
    selected_ids: Optional[Set[str]],
) -> Iterable[str]:
    k_intr = set(intrinsics.keys())
    k_extr = set(extrinsics.keys())
    common = k_intr & k_extr
    missing_intr = sorted(k_extr - k_intr)
    missing_extr = sorted(k_intr - k_extr)
    if missing_intr:
        print(f"[WARN] Missing intrinsics for {len(missing_intr)} cameras: {missing_intr}")
    if missing_extr:
        print(f"[WARN] Missing extrinsics for {len(missing_extr)} cameras: {missing_extr}")

    if selected_ids is not None:
        unknown = sorted(selected_ids - (k_intr | k_extr))
        if unknown:
            print(f"[WARN] Requested camera IDs not found: {unknown}")
        common = common & selected_ids

    return sorted(common)


def main():
    parser = argparse.ArgumentParser(description="Visualize camera system in viser from cam_param.")
    parser.add_argument("--episode-root", type=str, default=None, help="Path containing cam_param/ directory.")
    parser.add_argument("--camparam-dir", type=str, default=None, help="Direct path to cam_param directory.")
    parser.add_argument("--frustum-size", type=float, default=0.08, help="Camera frustum depth/size.")
    parser.add_argument("--show-axes", type=str2bool, default=True, help="Show camera axes.")
    parser.add_argument("--show-labels", type=str2bool, default=True, help="Show camera serial labels.")
    parser.add_argument("--show-floor", type=str2bool, default=True, help="Show floor/grid.")
    parser.add_argument("--floor-height", type=float, default=0.0, help="Floor height (z in world frame).")
    parser.add_argument(
        "--robot-frame",
        type=str2bool,
        default=False,
        help="Use robot frame as world frame by applying C2R.npy.",
    )
    parser.add_argument(
        "--c2r-path",
        type=str,
        default=None,
        help="Path to C2R.npy (default: <episode-root>/C2R.npy when available).",
    )
    parser.add_argument("--camera-ids", type=str, default=None, help="Comma-separated camera IDs.")
    parser.add_argument(
        "--fix-rotation",
        type=str2bool,
        default=False,
        help="If true, repair invalid rotation matrices with SVD; default is skip.",
    )
    parser.add_argument("--dry-run", type=str2bool, default=False, help="Validate and print only, do not open viewer.")
    args = parser.parse_args()

    camparam_dir = resolve_camparam_dir(args.episode_root, args.camparam_dir)
    intr_path = camparam_dir / "intrinsics.json"
    extr_path = camparam_dir / "extrinsics.json"
    print(f"[INFO] cam_param directory: {camparam_dir}")

    intrinsics = load_json(intr_path)
    extrinsics = load_json(extr_path)
    c2r = None
    if args.robot_frame:
        c2r_resolved = resolve_c2r_path(args.episode_root, args.c2r_path)
        if c2r_resolved is None:
            raise FileNotFoundError(
                "Robot-frame mode requires C2R.npy. "
                "Provide --c2r-path or set --episode-root with C2R.npy inside."
            )
        c2r = load_c2r(c2r_resolved)
        print(f"[INFO] robot-frame mode on: using C2R from {c2r_resolved}")
    selected_ids = parse_camera_ids(args.camera_ids)
    serials = list(iter_target_serials(intrinsics, extrinsics, selected_ids))
    if not serials:
        raise RuntimeError("No valid camera IDs to visualize after filtering.")

    viewer = None if args.dry_run else ViserViewer()
    if viewer is not None and args.show_floor:
        viewer.add_floor(height=args.floor_height)
    num_added = 0
    num_skipped = 0
    sample_logged = False
    for serial in serials:
        try:
            intrinsic_view = build_intrinsic_for_viewer(serial, intrinsics[serial])
            cam_from_world = parse_cam_from_world(extrinsics[serial])
            if c2r is not None:
                # cam_from_robot = cam_from_world @ world_from_robot(C2R)
                cam_from_world = cam_from_world @ c2r
            R_cw = cam_from_world[:3, :3]
            valid, reason = validate_rotation_matrix(R_cw)
            if not valid:
                if args.fix_rotation:
                    print(f"[WARN] {serial}: invalid rotation ({reason}), repairing via SVD.")
                    cam_from_world[:3, :3] = orthonormalize_rotation(R_cw)
                else:
                    print(f"[WARN] {serial}: invalid rotation ({reason}), skipped. Use --fix-rotation true to repair.")
                    num_skipped += 1
                    continue

            world_from_cam = np.linalg.inv(cam_from_world)
            if not sample_logged:
                print(f"[INFO] sample camera {serial}")
                print(f"       cam_from_world.t = {cam_from_world[:3, 3].round(6).tolist()}")
                print(f"       world_from_cam.t = {world_from_cam[:3, 3].round(6).tolist()}")
                sample_logged = True

            if viewer is not None:
                viewer.add_camera(
                    name=serial,
                    extrinsic=world_from_cam,
                    intrinsic=intrinsic_view,
                    color=color_from_name(serial),
                    size=args.frustum_size,
                    show_axes=args.show_axes,
                )
                if args.show_labels:
                    viewer.server.scene.add_label(f"/cameras/{serial}_frame/label", serial)
            num_added += 1
        except Exception as e:
            print(f"[WARN] {serial}: skipped due to error: {e}")
            num_skipped += 1

    print(f"[INFO] cameras added={num_added}, skipped={num_skipped}")
    if num_added == 0:
        raise RuntimeError("All cameras were skipped; nothing to visualize.")
    if viewer is not None:
        viewer.start_viewer()


if __name__ == "__main__":
    main()
