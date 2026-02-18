import argparse
import os
from typing import List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from paradex.calibration.utils import handeye_calib_path_openarm, load_camparam
from paradex.image.grid import make_image_grid
from paradex.image.image_dict import ImageDict
from paradex.robot.utils import get_robot_urdf_path
from paradex.utils.file_io import find_latest_directory
from paradex.visualization.robot import RobotModule


def _side_root(name: str, side: str) -> str:
    return os.path.join(handeye_calib_path_openarm, name, side)


def _valid_indices(root_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(root_dir):
        return out
    for idx in sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else x):
        p = os.path.join(root_dir, idx)
        if not os.path.isdir(p):
            continue
        if not os.path.exists(os.path.join(p, "qpos.npy")):
            continue
        und = os.path.join(p, "undistort", "images")
        raw = os.path.join(p, "images")
        if not os.path.isdir(und) and not os.path.isdir(raw):
            continue
        out.append(idx)
    return out


def _sample_indices(indices: List[str], n: int) -> List[str]:
    if len(indices) <= n:
        return indices
    sel = np.linspace(0, len(indices) - 1, n).round().astype(int)
    return [indices[i] for i in sel]


def _delta_T(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", [droll, dpitch, dyaw]).as_matrix()
    T[:3, 3] = np.array([dx, dy, dz], dtype=np.float64)
    return T


def _render_grid(
    image_dicts: List[ImageDict],
    robot_meshes: List,
    c2r: np.ndarray,
    indices: List[str],
    trans_step_m: float,
    rot_step_deg: float,
) -> np.ndarray:
    panels = []
    for idx, img_dict, mesh_robot in zip(indices, image_dicts, robot_meshes):
        mesh_cam = mesh_robot.copy()
        mesh_cam.apply_transform(c2r)
        overlay = img_dict.project_mesh(mesh_cam, color=(0, 255, 0), alpha=0.35)
        cam_grid = overlay.merge()
        cv2.putText(
            cam_grid,
            f"idx={idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(cam_grid)

    grid = make_image_grid(panels)
    t = c2r[:3, 3]
    euler = R.from_matrix(c2r[:3, :3]).as_euler("xyz", degrees=True)
    status1 = f"t[m]=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})  rpy[deg]=({euler[0]:+.2f},{euler[1]:+.2f},{euler[2]:+.2f})"
    status2 = f"step: trans={trans_step_m*1000:.1f}mm rot={rot_step_deg:.2f}deg | save:v reset:r quit:ESC"
    cv2.putText(grid, status1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(grid, status2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2, cv2.LINE_AA)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Calibration directory name under handeye_calib_path_openarm")
    parser.add_argument("--arm", type=str, default="openarm")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--n-frames", type=int, default=4, help="How many capture indices to display in the panel grid")
    parser.add_argument("--trans-step-mm", type=float, default=2.0)
    parser.add_argument("--rot-step-deg", type=float, default=0.5)
    parser.add_argument("--save-name", type=str, default="C2R_refined.npy")
    parser.add_argument(
        "--perturb-frame",
        type=str,
        default="robot",
        choices=["robot", "camera"],
        help="Apply keyboard perturbation in robot frame (post-multiply) or camera frame (pre-multiply).",
    )
    args = parser.parse_args()

    if args.name is None:
        args.name = find_latest_directory(handeye_calib_path_openarm)

    root_path = os.path.join(handeye_calib_path_openarm, args.name)
    side_root = _side_root(args.name, args.side)
    zero_root = os.path.join(root_path, "0")
    c2r_path = os.path.join(zero_root, "C2R.npy")
    if not os.path.exists(c2r_path):
        raise FileNotFoundError(f"C2R not found: {c2r_path}")

    intrinsic, extrinsic = load_camparam(zero_root)
    indices = _valid_indices(side_root)
    if not indices:
        raise RuntimeError(f"No valid capture indices in {side_root}")
    indices = _sample_indices(indices, max(1, args.n_frames))

    rm = RobotModule(get_robot_urdf_path(arm_name=args.arm))
    image_dicts: List[ImageDict] = []
    robot_meshes: List = []
    for idx in indices:
        idx_dir = os.path.join(side_root, idx)
        img_path = os.path.join(idx_dir, "undistort")
        if not os.path.isdir(os.path.join(img_path, "images")):
            img_path = idx_dir
        img_dict = ImageDict.from_path(img_path)
        img_dict.set_camparam(intrinsic, extrinsic)
        image_dicts.append(img_dict)

        qpos = np.load(os.path.join(idx_dir, "qpos.npy"))
        rm.update_cfg(qpos)
        robot_meshes.append(rm.get_robot_mesh())

    c2r_init = np.load(c2r_path).astype(np.float64)
    c2r = c2r_init.copy()

    trans_step_m = args.trans_step_mm / 1000.0
    rot_step_deg = float(args.rot_step_deg)
    win = "C2R Refine (OpenARM)"

    print("Controls:")
    print(f"  perturb frame: {args.perturb_frame}")
    print("  trans: q/a(+x/-x), w/s(+y/-y), e/d(+z/-z)")
    print("  rot  : u/j(+rx/-rx), i/k(+ry/-ry), o/l(+rz/-rz)")
    print("  step : ]/[ increase/decrease, reset:r, save:v, quit:ESC")

    while True:
        grid = _render_grid(image_dicts, robot_meshes, c2r, indices, trans_step_m, rot_step_deg)
        cv2.imshow(win, grid)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord("r"):
            c2r = c2r_init.copy()
            continue
        if key == ord("v"):
            save_path = os.path.join(zero_root, args.save_name)
            np.save(save_path, c2r)
            print(f"saved: {save_path}")
            continue

        if key == ord("]"):
            trans_step_m *= 1.25
            rot_step_deg *= 1.25
            continue
        if key == ord("["):
            trans_step_m /= 1.25
            rot_step_deg /= 1.25
            continue

        dt = np.zeros(3, dtype=np.float64)
        dr = np.zeros(3, dtype=np.float64)
        if key == ord("q"):
            dt[0] += trans_step_m
        elif key == ord("a"):
            dt[0] -= trans_step_m
        elif key == ord("w"):
            dt[1] += trans_step_m
        elif key == ord("s"):
            dt[1] -= trans_step_m
        elif key == ord("e"):
            dt[2] += trans_step_m
        elif key == ord("d"):
            dt[2] -= trans_step_m
        elif key == ord("u"):
            dr[0] += np.deg2rad(rot_step_deg)
        elif key == ord("j"):
            dr[0] -= np.deg2rad(rot_step_deg)
        elif key == ord("i"):
            dr[1] += np.deg2rad(rot_step_deg)
        elif key == ord("k"):
            dr[1] -= np.deg2rad(rot_step_deg)
        elif key == ord("o"):
            dr[2] += np.deg2rad(rot_step_deg)
        elif key == ord("l"):
            dr[2] -= np.deg2rad(rot_step_deg)
        else:
            continue

        delta = _delta_T(
            dx=dt[0], dy=dt[1], dz=dt[2], droll=dr[0], dpitch=dr[1], dyaw=dr[2]
        )
        if args.perturb_frame == "camera":
            c2r = delta @ c2r
        else:
            # Robot-space perturbation around the current robot frame.
            c2r = c2r @ delta

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
