"""Compare xArm kinematic calibration before/after using captured handeye data.

Re-solves the hand-eye problem twice on an existing
``~/shared_data/handeye_calibration/<session>/`` capture: once with the
nominal URDF (``rsc/robot/xarm.urdf.original``) and once with the
calibrated URDF (``rsc/robot/xarm.urdf``). Captured charuco data, qpos,
and eef.npy are reused as-is — only the FK and the solver are rerun.

Prerequisites:
- ``src/calibration/xarm_kinematic_calibration.py`` has been run, so
  ``rsc/robot/xarm.urdf.original`` exists.
- ``src/calibration/handeye/calculate.py`` has been run at least once on
  the target session (charuco_3d_corners.npy / charuco_3d_ids.npy cached).

Usage:
    python src/validate/calibration/compare_xarm_kinematic_calib.py
    python src/validate/calibration/compare_xarm_kinematic_calib.py --name 20251011_194400
    python src/validate/calibration/compare_xarm_kinematic_calib.py --no_overlay
"""
import argparse
import os

import numpy as np
import tqdm

from paradex.calibration.Tsai_Lenz import solve_ax_xb
from paradex.calibration.utils import handeye_calib_path, load_camparam
from paradex.image.aruco import find_common_indices
from paradex.image.image_dict import ImageDict
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.utils.file_io import find_latest_directory
from paradex.visualization.robot import RobotModule


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_NOMINAL_URDF = os.path.join(REPO_ROOT, "rsc/robot/xarm.urdf.original")
DEFAULT_CALIB_URDF = os.path.join(REPO_ROOT, "rsc/robot/xarm.urdf")
EEF_LINK = "link6"


def _load_session(name):
    root_dir = os.path.join(handeye_calib_path, name)
    index_list = sorted(os.listdir(root_dir), key=lambda s: int(s) if s.isdigit() else s)
    index_list = [i for i in index_list if os.path.isdir(os.path.join(root_dir, i))]
    return root_dir, index_list


def _compute_motion_wrt_cam(root_dir, index_list):
    motion_wrt_cam = []
    fit_errs_mm = []
    charuco_ids = [
        np.load(os.path.join(root_dir, i, "charuco_3d_ids.npy")) for i in index_list
    ]
    charuco_cor = [
        np.load(os.path.join(root_dir, i, "charuco_3d_corners.npy")) for i in index_list
    ]
    for i in range(1, len(index_list)):
        common, common_prev = find_common_indices(charuco_ids[i], charuco_ids[i - 1])
        cor = charuco_cor[i][common]
        cor_prev = charuco_cor[i - 1][common_prev]
        M = SOLVE_XA_B(cor, cor_prev)
        motion_wrt_cam.append(M)
        err = cor_prev - (M[:3, :3] @ cor.T).T - M[:3, 3]
        fit_errs_mm.append(float(np.mean(np.linalg.norm(err, axis=1)) * 1000))
    return motion_wrt_cam, fit_errs_mm


def _eval_variant(name_tag, urdf_path, root_dir, index_list, motion_wrt_cam, overlay):
    print(f"\n========== {name_tag}  ({urdf_path}) ==========")
    if not os.path.exists(urdf_path):
        print(f"  [missing URDF] {urdf_path}")
        return None

    # FK per index
    rw = RobotWrapper(urdf_path)
    eef_fk_list = []
    for idx in index_list:
        qpos = np.load(os.path.join(root_dir, idx, "qpos.npy"))
        eef = rw.compute_forward_kinematics(qpos, link_list=[EEF_LINK])[EEF_LINK]
        eef_fk_list.append(eef)

    # Robot motion (URDF-dependent)
    motion_wrt_robot = [
        eef_fk_list[i - 1] @ np.linalg.inv(eef_fk_list[i])
        for i in range(1, len(index_list))
    ]

    # Solve AX = XB
    robot_wrt_cam = solve_ax_xb(motion_wrt_cam, motion_wrt_robot, verbose=False)

    # AX-XB residual (URDF-independent ground truth: pure motion consistency)
    axxb_trans = []
    for A, B in zip(motion_wrt_cam, motion_wrt_robot):
        diff = (A @ robot_wrt_cam) - (robot_wrt_cam @ B)
        axxb_trans.append(float(np.linalg.norm(diff[:3, 3]) * 1000))

    # FK error vs robot's reported TCP (partially circular but useful)
    fk_trans = []
    fk_rot = []
    for idx, eef in zip(index_list, eef_fk_list):
        eef_robot_path = os.path.join(root_dir, idx, "eef.npy")
        if not os.path.exists(eef_robot_path):
            continue
        eef_from_robot = np.load(eef_robot_path)
        err = np.linalg.inv(eef) @ eef_from_robot
        fk_trans.append(float(np.linalg.norm(err[:3, 3]) * 1000))
        cosv = (np.trace(err[:3, :3]) - 1) / 2.0
        cosv = max(-1.0, min(1.0, cosv))
        fk_rot.append(float(np.degrees(np.arccos(cosv))))

    # Marker std (each charuco point's location in EEF frame, across captures)
    marker_obs = {}
    for idx, eef in zip(index_list, eef_fk_list):
        ids = np.load(os.path.join(root_dir, idx, "charuco_3d_ids.npy"))
        cor = np.load(os.path.join(root_dir, idx, "charuco_3d_corners.npy"))
        eef_inv = np.linalg.inv(eef)
        cam_inv = np.linalg.inv(robot_wrt_cam)
        for mid, c in zip(ids, cor):
            c_h = np.array([c[0], c[1], c[2], 1.0])
            wrt_eef = (eef_inv @ cam_inv @ c_h)[:3]
            marker_obs.setdefault(int(mid), []).append(wrt_eef)

    marker_stds = []
    for mid, obs in marker_obs.items():
        if len(obs) < 2:
            continue
        marker_stds.append(np.std(np.array(obs), axis=0))
    marker_stds = np.array(marker_stds) * 1000 if marker_stds else np.zeros((0, 3))

    summary = {
        "variant": name_tag,
        "C2R": np.linalg.inv(robot_wrt_cam),
        "robot_wrt_cam": robot_wrt_cam,
        "axxb_trans_mm": np.array(axxb_trans),
        "fk_trans_mm": np.array(fk_trans),
        "fk_rot_deg": np.array(fk_rot),
        "marker_std_mm": marker_stds,
    }

    print(f"  AX-XB residual (mm):  mean={summary['axxb_trans_mm'].mean():.3f}  "
          f"median={np.median(summary['axxb_trans_mm']):.3f}  "
          f"max={summary['axxb_trans_mm'].max():.3f}")
    print(f"  FK vs robot TCP (mm): mean={summary['fk_trans_mm'].mean():.3f}  "
          f"median={np.median(summary['fk_trans_mm']):.3f}  "
          f"max={summary['fk_trans_mm'].max():.3f}")
    print(f"  FK rot error (deg):   mean={summary['fk_rot_deg'].mean():.4f}  "
          f"max={summary['fk_rot_deg'].max():.4f}")
    if len(marker_stds):
        per_marker_norm = np.linalg.norm(marker_stds, axis=1)
        print(f"  Marker std |xyz| (mm): mean={per_marker_norm.mean():.3f}  "
              f"median={np.median(per_marker_norm):.3f}  "
              f"max={per_marker_norm.max():.3f}  "
              f"(N={len(marker_stds)})")

    # Save tagged outputs
    np.save(os.path.join(root_dir, index_list[0], f"C2R_{name_tag}.npy"), robot_wrt_cam)

    if overlay:
        _render_overlay(name_tag, urdf_path, root_dir, index_list, eef_fk_list,
                        robot_wrt_cam, marker_obs)

    return summary


def _render_overlay(name_tag, urdf_path, root_dir, index_list, eef_fk_list,
                    robot_wrt_cam, marker_obs):
    print(f"  rendering overlay to {{index}}/debug_{name_tag}/ ...")
    rm = RobotModule(urdf_path)
    intrinsic, extrinsic = load_camparam(os.path.join(root_dir, index_list[0]))

    # mean marker position in EEF frame (same recipe as calculate.py:debug)
    marker_wrt_eef = []
    for mid, obs in marker_obs.items():
        marker_wrt_eef.append(np.mean(np.array(obs), axis=0))
    marker_wrt_eef = np.array(marker_wrt_eef) if marker_wrt_eef else np.zeros((0, 3))

    img_dict = None
    for idx, eef in tqdm.tqdm(list(zip(index_list, eef_fk_list)),
                              desc=f"overlay-{name_tag}"):
        undist_path = os.path.join(root_dir, idx, "undistort")
        if not os.path.isdir(undist_path):
            print(f"    [skip {idx}] no undistort/ — run calculate.py first")
            continue
        if img_dict is None:
            img_dict = ImageDict.from_path(undist_path)
            img_dict.set_camparam(intrinsic, extrinsic)
        else:
            img_dict.update_path(undist_path)

        qpos = np.load(os.path.join(root_dir, idx, "qpos.npy"))
        rm.update_cfg(qpos)
        mesh = rm.get_robot_mesh()
        mesh.apply_transform(robot_wrt_cam)
        overlay_dict = img_dict.project_mesh(mesh)

        if len(marker_wrt_eef):
            mh = np.ones((marker_wrt_eef.shape[0], 4))
            mh[:, :3] = marker_wrt_eef
            wrt_cam = (robot_wrt_cam @ eef @ mh.T).T[:, :3]
            overlay_dict.draw_keypoint(overlay_dict.project_pointcloud(wrt_cam),
                                       (255, 0, 0))  # predicted (red)

        observed = np.load(os.path.join(root_dir, idx, "charuco_3d_corners.npy"))
        overlay_dict.draw_keypoint(overlay_dict.project_pointcloud(observed),
                                   (0, 0, 255))      # observed (blue)

        overlay_dict.save(os.path.join(root_dir, idx, f"debug_{name_tag}"))


def _print_side_by_side(summaries):
    valid = [s for s in summaries if s is not None]
    if len(valid) < 2:
        return
    a, b = valid[0], valid[1]

    def _row(label, va, vb, fmt="{:.3f}"):
        delta = vb - va
        delta_str = ("+" if delta >= 0 else "") + fmt.format(delta)
        return f"  {label:<28}  {fmt.format(va):>10}    {fmt.format(vb):>10}    {delta_str:>10}"

    print("\n========== SUMMARY (lower is better) ==========")
    print(f"  {'metric':<28}  {a['variant']:>10}    {b['variant']:>10}    {'Δ (b-a)':>10}")
    print(_row("AX-XB residual mean (mm)",
               a["axxb_trans_mm"].mean(), b["axxb_trans_mm"].mean()))
    print(_row("AX-XB residual median (mm)",
               np.median(a["axxb_trans_mm"]), np.median(b["axxb_trans_mm"])))
    print(_row("AX-XB residual max (mm)",
               a["axxb_trans_mm"].max(), b["axxb_trans_mm"].max()))
    print(_row("FK trans mean (mm)",
               a["fk_trans_mm"].mean(), b["fk_trans_mm"].mean()))
    print(_row("FK rot mean (deg)",
               a["fk_rot_deg"].mean(), b["fk_rot_deg"].mean(), fmt="{:.4f}"))
    if len(a["marker_std_mm"]) and len(b["marker_std_mm"]):
        an = np.linalg.norm(a["marker_std_mm"], axis=1)
        bn = np.linalg.norm(b["marker_std_mm"], axis=1)
        print(_row("Marker std |xyz| mean (mm)", an.mean(), bn.mean()))
        print(_row("Marker std |xyz| median (mm)", np.median(an), np.median(bn)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default=None, help="Handeye session timestamp")
    p.add_argument("--nominal_urdf", type=str, default=DEFAULT_NOMINAL_URDF)
    p.add_argument("--calibrated_urdf", type=str, default=DEFAULT_CALIB_URDF)
    p.add_argument("--no_overlay", action="store_true",
                   help="Skip overlay rendering (overlay is on by default)")
    args = p.parse_args()
    overlay = not args.no_overlay

    name = args.name or find_latest_directory(handeye_calib_path)
    root_dir, index_list = _load_session(name)
    print(f"Session: {name}  ({len(index_list)} captures)")

    # Sanity-check that charuco data is cached
    missing = [i for i in index_list
               if not os.path.exists(os.path.join(root_dir, i, "charuco_3d_corners.npy"))]
    if missing:
        raise SystemExit(
            f"Missing charuco_3d_corners.npy in {len(missing)} indices "
            f"(first: {missing[0]}). Run calculate.py once first to cache it."
        )

    motion_wrt_cam, cam_fit = _compute_motion_wrt_cam(root_dir, index_list)
    print(f"Cam motion fit (mm): mean={np.mean(cam_fit):.3f}  max={np.max(cam_fit):.3f}  "
          "(URDF-independent — sanity check on charuco data)")

    summaries = [
        _eval_variant("nominal", args.nominal_urdf, root_dir, index_list,
                      motion_wrt_cam, overlay),
        _eval_variant("calibrated", args.calibrated_urdf, root_dir, index_list,
                      motion_wrt_cam, overlay),
    ]
    _print_side_by_side(summaries)
    print(f"\nC2R saved as {root_dir}/0/C2R_nominal.npy and C2R_calibrated.npy")
    if overlay:
        print(f"Overlay images in {root_dir}/{{index}}/debug_nominal/ and debug_calibrated/")


if __name__ == "__main__":
    main()
