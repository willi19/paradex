import argparse
import os
import json
import numpy as np
import cv2

from paradex.utils.file_io import find_latest_directory
from paradex.calibration.utils import (
    cam_param_dir, extrinsic_dir, load_current_intrinsic,
)
from paradex.image.aruco import (
    boardinfo_dict, _charuco_board_cache, get_charuco_detector,
)


def build_id_to_3d():
    get_charuco_detector()  # ensures setLegacyPattern is applied
    out = {}
    offset = 0
    for b_id, cfg in boardinfo_dict.items():
        obj_pts = _charuco_board_cache[b_id].getChessboardCorners().reshape(-1, 3)
        n = (cfg["numX"] - 1) * (cfg["numY"] - 1)
        out[b_id] = {local_id + offset: obj_pts[local_id] for local_id in range(n)}
        offset += n
    return out


def discover_serials(root_dir):
    serials = set()
    for idx in sorted(os.listdir(root_dir)):
        md = os.path.join(root_dir, idx, "markers_2d")
        if not os.path.isdir(md):
            continue
        for f in os.listdir(md):
            if f.endswith("_corner.npy"):
                serials.add(f.split("_")[0])
    return sorted(serials)


def collect_views(root_dir, serial1, serial2):
    id_to_3d = build_id_to_3d()
    objp_list, p1_list, p2_list = [], [], []

    for idx in sorted(os.listdir(root_dir)):
        md = os.path.join(root_dir, idx, "markers_2d")
        f1c = os.path.join(md, f"{serial1}_corner.npy")
        f1i = os.path.join(md, f"{serial1}_id.npy")
        f2c = os.path.join(md, f"{serial2}_corner.npy")
        f2i = os.path.join(md, f"{serial2}_id.npy")
        if not all(os.path.exists(p) for p in (f1c, f1i, f2c, f2i)):
            continue

        c1, i1 = np.load(f1c), np.load(f1i)
        c2, i2 = np.load(f2c), np.load(f2i)
        if len(i1) == 0 or len(i2) == 0:
            continue

        map1 = {int(mid): k for k, mid in enumerate(i1)}
        map2 = {int(mid): k for k, mid in enumerate(i2)}

        for b_id, mapping in id_to_3d.items():
            keys = set(mapping.keys())
            common = sorted(keys & set(map1.keys()) & set(map2.keys()))
            if len(common) < 6:
                continue

            objp = np.asarray([mapping[mid] for mid in common], dtype=np.float32)
            p1 = np.asarray([c1[map1[mid]] for mid in common], dtype=np.float32)
            p2 = np.asarray([c2[map2[mid]] for mid in common], dtype=np.float32)

            objp_list.append(objp)
            p1_list.append(p1)
            p2_list.append(p2)

    return objp_list, p1_list, p2_list


def run_stereo_calibration(name):
    root_dir = os.path.join(extrinsic_dir, name)
    intrinsics_dict = load_current_intrinsic()

    serials = discover_serials(root_dir)
    if len(serials) != 2:
        raise ValueError(
            f"stereo calibration expects exactly 2 cameras, found {len(serials)}: {serials}"
        )
    s1, s2 = serials
    for s in serials:
        if s not in intrinsics_dict:
            raise KeyError(f"intrinsic for {s} not found in shared_data/intrinsic/")

    K1 = intrinsics_dict[s1]["original_intrinsics"]
    d1 = intrinsics_dict[s1]["dist_params"]
    K2 = intrinsics_dict[s2]["original_intrinsics"]
    d2 = intrinsics_dict[s2]["dist_params"]
    w, h = intrinsics_dict[s1]["width"], intrinsics_dict[s1]["height"]

    objp, p1, p2 = collect_views(root_dir, s1, s2)
    if len(objp) < 5:
        raise ValueError(
            f"insufficient stereo views: {len(objp)} (target >= 10 across diverse poses)"
        )
    print(f"[{name}] cameras: {s1}, {s2}  | views: {len(objp)}  | "
          f"corners: {sum(len(o) for o in objp)}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objp, p1, p2, K1, d1, K2, d2, (w, h),
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=criteria,
    )
    print(f"stereo RMS reproj error: {rms:.4f} px   |T| = {np.linalg.norm(T):.4f} m")

    extrinsics = {
        s1: np.hstack([np.eye(3), np.zeros((3, 1))]).tolist(),
        s2: np.hstack([R, T.reshape(3, 1)]).tolist(),
    }

    out_dir = os.path.join(cam_param_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "extrinsics.json"), "w") as f:
        json.dump(extrinsics, f, indent=4)

    intrinsics_out = {}
    for s in serials:
        intr = intrinsics_dict[s]
        intrinsics_out[s] = {
            "original_intrinsics": np.asarray(intr["original_intrinsics"]).tolist(),
            "intrinsics_undistort": np.asarray(intr["intrinsics_undistort"]).tolist(),
            "dist_params": np.asarray(intr["dist_params"]).tolist(),
            "height": intr["height"],
            "width": intr["width"],
        }
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics_out, f, indent=4)

    print(f"saved -> {out_dir}/extrinsics.json , intrinsics.json")
    return rms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2-camera extrinsic calibration via cv2.stereoCalibrate (charuco)."
    )
    parser.add_argument("--name", type=str, default=None,
                        help="session under shared_data/extrinsic/. defaults to latest.")
    args = parser.parse_args()

    name = args.name or find_latest_directory(extrinsic_dir)
    run_stereo_calibration(name)
