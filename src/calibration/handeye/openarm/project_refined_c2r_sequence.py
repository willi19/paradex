import argparse
import os
from typing import List

import cv2
import numpy as np
import tqdm

from paradex.calibration.utils import handeye_calib_path_openarm, load_camparam
from paradex.image.image_dict import ImageDict
from paradex.robot.utils import get_robot_urdf_path
from paradex.utils.file_io import find_latest_directory
from paradex.visualization.robot import RobotModule


def _side_root(name: str, side: str) -> str:
    return os.path.join(handeye_calib_path_openarm, name, side)


def _valid_indices(root_dir: str) -> List[str]:
    indices: List[str] = []
    if not os.path.isdir(root_dir):
        return indices
    for idx in sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else x):
        p = os.path.join(root_dir, idx)
        if not os.path.isdir(p):
            continue
        if not os.path.exists(os.path.join(p, "qpos.npy")):
            continue
        has_img = os.path.isdir(os.path.join(p, "undistort", "images")) or os.path.isdir(
            os.path.join(p, "images")
        )
        if not has_img:
            continue
        indices.append(idx)
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Calibration directory name.")
    parser.add_argument("--arm", type=str, default="openarm")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    parser.add_argument(
        "--c2r-file",
        type=str,
        default="C2R_refined.npy",
        help="C2R file under <calib_root>/0. Fallback to C2R.npy when missing.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output folder for grid images. Default: <calib_root>/<side>/projection_refined_grid",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in sorted capture list.")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive), -1 for all.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    args = parser.parse_args()

    if args.name is None:
        args.name = find_latest_directory(handeye_calib_path_openarm)

    root_path = os.path.join(handeye_calib_path_openarm, args.name)
    zero_root = os.path.join(root_path, "0")
    side_root = _side_root(args.name, args.side)

    c2r_path = os.path.join(zero_root, args.c2r_file)
    if not os.path.exists(c2r_path):
        c2r_path = os.path.join(zero_root, "C2R.npy")
    if not os.path.exists(c2r_path):
        raise FileNotFoundError(f"No C2R file found in {zero_root}")
    c2r = np.load(c2r_path).astype(np.float64)
    print(f"Using C2R: {c2r_path}")

    intrinsic, extrinsic = load_camparam(zero_root)
    rm = RobotModule(get_robot_urdf_path(arm_name=args.arm))

    indices = _valid_indices(side_root)
    if not indices:
        raise RuntimeError(f"No valid sequence frames in {side_root}")

    s = max(0, int(args.start))
    e = len(indices) if int(args.end) < 0 else min(len(indices), int(args.end))
    if e <= s:
        raise ValueError(f"Invalid range start={s}, end={e}, total={len(indices)}")
    st = max(1, int(args.stride))
    indices = indices[s:e:st]

    out_dir = args.out_dir or os.path.join(side_root, "projection_refined_grid")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving grids to: {out_dir}")
    print(f"Frames: {len(indices)}")

    for idx in tqdm.tqdm(indices, desc="Project refined C2R"):
        idx_dir = os.path.join(side_root, idx)
        img_path = os.path.join(idx_dir, "undistort")
        if not os.path.isdir(os.path.join(img_path, "images")):
            img_path = idx_dir

        img_dict = ImageDict.from_path(img_path)
        img_dict.set_camparam(intrinsic, extrinsic)

        qpos = np.load(os.path.join(idx_dir, "qpos.npy"))
        rm.update_cfg(qpos)
        robot_mesh = rm.get_robot_mesh()
        robot_mesh.apply_transform(c2r)

        overlay = img_dict.project_mesh(robot_mesh, color=(0, 255, 0), alpha=0.35)
        grid = overlay.merge()
        cv2.putText(
            grid,
            f"idx={idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(os.path.join(out_dir, f"{idx}.png"), grid)

    print("Done.")


if __name__ == "__main__":
    main()

