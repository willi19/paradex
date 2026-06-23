import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from paradex.visualization.visualizer.viser import ViserViewer


def parse_cameras_txt(path: Path) -> Dict[int, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cameras.txt: {path}")

    out: Dict[int, Dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sp = line.split()
        if len(sp) < 8:
            continue

        cam_id = int(sp[0])
        model = sp[1]
        width = int(float(sp[2]))
        height = int(float(sp[3]))
        params = np.array(list(map(float, sp[4:])), dtype=np.float64)

        if model not in ("OPENCV", "PINHOLE"):
            continue
        if len(params) < 4:
            continue

        fx, fy, cx, cy = params[:4]
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        out[cam_id] = {
            "intrinsics_undistort": K.tolist(),
            "width": width,
            "height": height,
            "model": model,
        }
    return out


def parse_images_txt(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing images.txt: {path}")

    out: Dict[str, Dict] = {}
    lines = path.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        sp = line.split()
        if len(sp) < 10 or not sp[0].isdigit():
            i += 1
            continue

        qw, qx, qy, qz = map(float, sp[1:5])
        tx, ty, tz = map(float, sp[5:8])
        cam_id = int(sp[8])
        name = sp[9]

        R_cw = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_cw = np.array([tx, ty, tz], dtype=np.float64)

        cam_from_world = np.eye(4, dtype=np.float64)
        cam_from_world[:3, :3] = R_cw
        cam_from_world[:3, 3] = t_cw

        out[name] = {
            "camera_id": cam_id,
            "cam_from_world": cam_from_world,
        }

        # Next line is POINTS2D; skip it if exists.
        i += 2

    return out


def color_for_serial(serial: str) -> Tuple[int, int, int]:
    if serial == "25452062":
        return (255, 90, 90)
    if serial == "25452066":
        return (90, 170, 255)
    return (90, 90, 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize COLMAP camera poses in viser.")
    parser.add_argument(
        "--model-txt-dir",
        type=str,
        default="/home/temp_id/shared_data/capture/eccv2026/hand_taeyun/right/apple/0/colmap_joint_opt_allcams_00001/model_txt",
        help="Directory containing cameras.txt and images.txt",
    )
    parser.add_argument("--frustum-size", type=float, default=0.06)
    parser.add_argument("--show-axes", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_txt_dir).expanduser().resolve()
    cameras = parse_cameras_txt(model_dir / "cameras.txt")
    images = parse_images_txt(model_dir / "images.txt")

    vis = ViserViewer()
    vis.add_floor(height=0.0)

    added = 0
    for image_name, item in sorted(images.items()):
        cam_id = item["camera_id"]
        if cam_id not in cameras:
            continue

        serial = image_name.rsplit(".", 1)[0]
        world_from_cam = np.linalg.inv(item["cam_from_world"])

        vis.add_camera(
            name=serial,
            extrinsic=world_from_cam,
            intrinsic=cameras[cam_id],
            color=color_for_serial(serial),
            size=float(args.frustum_size),
            show_axes=bool(args.show_axes),
        )

        if not args.hide_labels:
            vis.server.scene.add_label(f"/cameras/{serial}_frame/label", serial)
        added += 1

    print(f"[VIS] model_dir={model_dir}")
    print(f"[VIS] added cameras={added}")
    print("[VIS] open viewer: http://localhost:8080")
    vis.start_viewer()


if __name__ == "__main__":
    main()
