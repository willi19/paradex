"""Retarget collab_bundle MANO predictions to allegro_v5 right hand.

Steps:
  1. Load predictions.npz (MANO params in camera 23029839 = frame-0 cam frame).
  2. Forward MANO -> 21 joints / vertices per frame in cam0 space.
  3. Transform joints/verts cam0 -> current calib world via E[23029839]^-1.
  4. Run dex-retargeting (vector) using local (wrist-relative) joints to get
     allegro_v5 qpos per frame.
  5. Save qpos + transformed joints/verts.

Output (under --output-dir):
  joints_world_{side}.npy        (T, 21, 3) in current calib world frame
  verts_world_{side}.npy         (T, 778, 3)
  qpos_{side}.npy                (T, 16)
  joint_names_{side}.json
  meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from paradex.calibration.utils import load_current_camparam


SOURCE_SERIAL = "23029839"

OPERATOR2MANO_RIGHT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32
)
OPERATOR2MANO_LEFT = np.array(
    [[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float32
)

ALLEGRO_V5_DIR = Path("/home/temp_id/paradex/rsc/robot/allegro_v5")

# dex-retargeting yaml templates for allegro_v5.
#
# allegro_v5 URDF anchors palm_link at the MCP plane (joints 0_0/4_0/8_0/12_0
# attach at near-zero offsets from palm_link). The standard allegro config
# uses a separate `wrist` link as origin, but v5 doesn't have one. To keep
# retargeting math consistent with the URDF kinematics, we use MANO middle MCP
# (idx 9) as the common origin instead of MANO wrist (idx 0).
#
# scaling_factor scales the *human* reference vectors before matching with
# robot vectors. Larger scaling_factor -> longer target -> fingers extend more
# (less curl). Useful when the operator's hand is bigger than the robot.
_DEFAULT_VECTOR_SCALING = 1.0

_VECTOR_TEMPLATE = """retargeting:
  type: vector
  urdf_path: allegro_v5_right.urdf
  target_origin_link_names: [palm_link, palm_link, palm_link, palm_link]
  target_task_link_names: [link_15_0_tip, link_3_0_tip, link_7_0_tip, link_11_0_tip]
  scaling_factor: {scaling_factor}
  target_link_human_indices: [[9, 9, 9, 9], [4, 8, 12, 16]]
  low_pass_alpha: {low_pass_alpha}
"""

# Position retargeting: palm_link at MCP center (idx 9), tips at 4/8/12/16.
_POSITION_TEMPLATE = """retargeting:
  type: position
  urdf_path: allegro_v5_right.urdf
  target_link_names: [palm_link, link_15_0_tip, link_3_0_tip, link_7_0_tip, link_11_0_tip]
  target_link_human_indices: [9, 4, 8, 12, 16]
  low_pass_alpha: {low_pass_alpha}
"""

_DEFAULT_LOW_PASS_ALPHA = 0.2


def _e_inv_4x4(extrinsic_3x4: np.ndarray) -> np.ndarray:
    E = np.eye(4, dtype=np.float64)
    E[:3, :] = np.asarray(extrinsic_3x4, dtype=np.float64)
    return np.linalg.inv(E)


def _transform_pts(pts: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    R = T_4x4[:3, :3]
    t = T_4x4[:3, 3]
    return pts @ R.T + t


# MANO mesh vertex indices for fingertips (standard smplx VERTEX_IDS['mano']).
_MANO_TIP_VERT_IDX = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 672,
}

# Reorder smplx MANO 16 joints + 5 appended tips -> HAMER 21-joint convention:
#   0 wrist
#   1..4 thumb (mcp, pip, dip, tip)
#   5..8 index, 9..12 middle, 13..16 ring, 17..20 pinky
# smplx MANO J_regressor order: 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky,
# 10-12=ring, 13-15=thumb. Appended tips order matches _MANO_TIP_VERT_IDX:
# 16=thumb, 17=index, 18=middle, 19=ring, 20=pinky.
_MANO_TO_HAMER = [
    0,
    13, 14, 15, 16,  # thumb
    1, 2, 3, 17,     # index
    4, 5, 6, 18,     # middle
    10, 11, 12, 19,  # ring
    7, 8, 9, 20,     # pinky
]


def _forward_mano(
    bundle: dict, side: str, mano_dir: str, T: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (joints[T,21,3] HAMER convention, verts[T,778,3], faces[F,3])
    in cam0 frame.
    """
    import torch
    import smplx

    is_rhand = side == "right"
    pkl = "MANO_RIGHT.pkl" if is_rhand else "MANO_LEFT.pkl"
    m = smplx.create(
        os.path.join(mano_dir, pkl),
        "mano",
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=True,
    )
    if not is_rhand:
        m_r = smplx.create(
            os.path.join(mano_dir, "MANO_RIGHT.pkl"),
            "mano",
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True,
        )
        if torch.sum(torch.abs(m.shapedirs[:, 0, :] - m_r.shapedirs[:, 0, :])) < 1:
            m.shapedirs[:, 0, :] *= -1

    betas = bundle[f"{side}_betas"]
    transl = bundle[f"{side}_trans"]
    orient = bundle[f"{side}_root_orient"]
    pose = bundle[f"{side}_hand_pose"]

    with torch.no_grad():
        out = m(
            global_orient=torch.tensor(orient, dtype=torch.float32),
            hand_pose=torch.tensor(pose, dtype=torch.float32),
            betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(T, -1),
            transl=torch.tensor(transl, dtype=torch.float32),
        )
    joints16 = out.joints.detach().cpu().numpy().astype(np.float64)   # (T, 16, 3)
    verts = out.vertices.detach().cpu().numpy().astype(np.float64)    # (T, 778, 3)

    tips = np.stack(
        [verts[:, _MANO_TIP_VERT_IDX[k], :] for k in ("thumb", "index", "middle", "ring", "pinky")],
        axis=1,
    )  # (T, 5, 3)
    joints21 = np.concatenate([joints16, tips], axis=1)               # (T, 21, 3)
    joints_hamer = joints21[:, _MANO_TO_HAMER, :]
    faces = np.asarray(m.faces, dtype=np.int32)
    return joints_hamer, verts, faces


def _estimate_mano_wrist_frame(keypoint_3d: np.ndarray) -> np.ndarray:
    """SVD-based hand frame, matches dex-retargeting reference impl."""
    points = keypoint_3d[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(centered)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-8)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


def _build_retargeter(side: str, retarget_type: str, work_dir: Path,
                      scaling_factor: float = _DEFAULT_VECTOR_SCALING,
                      low_pass_alpha: float = _DEFAULT_LOW_PASS_ALPHA):
    from dex_retargeting.retargeting_config import RetargetingConfig

    if side != "right":
        raise NotImplementedError("allegro_v5 left URDF not available yet")
    if retarget_type not in ("vector", "position"):
        raise ValueError(f"unknown retarget type: {retarget_type}")

    if retarget_type == "vector":
        yaml_str = _VECTOR_TEMPLATE.format(
            scaling_factor=float(scaling_factor),
            low_pass_alpha=float(low_pass_alpha),
        )
    else:
        yaml_str = _POSITION_TEMPLATE.format(low_pass_alpha=float(low_pass_alpha))
    RetargetingConfig.set_default_urdf_dir(str(ALLEGRO_V5_DIR))
    yml_path = work_dir / f"allegro_v5_right_{retarget_type}.yml"
    yml_path.write_text(yaml_str)
    return RetargetingConfig.load_from_file(str(yml_path)).build()


def _retarget_frames(joints_cam0: np.ndarray, valid: np.ndarray, side: str, rt) -> np.ndarray:
    """Run vector retargeting per frame; returns (T, n_dof) qpos. Uses MANO
    21-joint local (wrist-relative, operator frame) coords — frame-independent,
    so we can use cam0 joints directly without going through world.
    """
    op = OPERATOR2MANO_RIGHT if side == "right" else OPERATOR2MANO_LEFT
    indices = rt.optimizer.target_link_human_indices
    rtype = rt.optimizer.retargeting_type
    n_fixed = len(rt.optimizer.idx_pin2fixed)
    fixed_qpos = np.zeros((n_fixed,), dtype=np.float32)

    T = joints_cam0.shape[0]
    n_dof = len(rt.optimizer.robot.dof_joint_names)
    qpos_all = np.full((T, n_dof), np.nan, dtype=np.float32)

    last = None
    for t in range(T):
        if not bool(valid[t]):
            if last is not None:
                qpos_all[t] = last
            continue
        kp3 = joints_cam0[t].astype(np.float32)
        kp3 = kp3 - kp3[0:1, :]
        wrist_rot = _estimate_mano_wrist_frame(kp3)
        joint_pos = kp3 @ wrist_rot @ op
        if rtype == "POSITION":
            ref = joint_pos[indices, :]
        else:
            ref = joint_pos[indices[1, :], :] - joint_pos[indices[0, :], :]
        q = rt.retarget(ref, fixed_qpos=fixed_qpos)
        last = np.asarray(q, dtype=np.float32)
        qpos_all[t] = last
    return qpos_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle",
        default=str(Path(__file__).parent / "collab_bundle" / "predictions.npz"),
    )
    ap.add_argument("--output-dir", default=str(Path(__file__).parent / "out_taeksoo"))
    ap.add_argument(
        "--mano-model-dir",
        default="/home/temp_id/hamer-mediapipe/_DATA/data/mano",
    )
    ap.add_argument("--calib-name", default=None,
                    help="cam_param subdir; defaults to latest.")
    ap.add_argument("--source-serial", default=SOURCE_SERIAL,
                    help="Camera serial whose frame the bundle lives in (frame-0 cam).")
    ap.add_argument("--sides", nargs="+", default=["right"], choices=["right", "left"])
    ap.add_argument("--retarget-type", default="vector", choices=["vector", "position"])
    ap.add_argument("--scaling-factor", type=float, default=_DEFAULT_VECTOR_SCALING,
                    help="Vector retargeting: scales human reference vectors. "
                    "Higher -> fingers more extended (less curl).")
    ap.add_argument("--low-pass-alpha", type=float, default=_DEFAULT_LOW_PASS_ALPHA,
                    help="dex-retargeting low-pass alpha. "
                    "Smaller -> stronger filter / slower response (e.g. 0.05).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = dict(np.load(args.bundle, allow_pickle=True))
    T = int(bundle["right_trans"].shape[0])
    print(f"[taeksoo] bundle T={T} sides={args.sides}")

    intrinsic, extrinsic = load_current_camparam(args.calib_name)
    if args.source_serial not in extrinsic:
        raise KeyError(
            f"source serial {args.source_serial} not in current calibration "
            f"({sorted(extrinsic.keys())[:5]}...)"
        )
    E_world_from_cam0 = _e_inv_4x4(extrinsic[args.source_serial])
    print(f"[taeksoo] cam0={args.source_serial} -> world transform loaded")

    meta = {
        "bundle": str(args.bundle),
        "source_serial": args.source_serial,
        "calib_name": args.calib_name,
        "T": T,
        "world_from_cam0": E_world_from_cam0.tolist(),
        "sides": {},
    }

    for side in args.sides:
        valid_key = f"pred_valid_{side}"
        valid = bundle[valid_key] if valid_key in bundle else bundle[f"{side}_valid"]
        valid = np.asarray(valid, dtype=bool)

        joints_cam0, verts_cam0, faces = _forward_mano(
            bundle, side, args.mano_model_dir, T
        )
        joints_world = np.stack(
            [_transform_pts(joints_cam0[t], E_world_from_cam0) for t in range(T)]
        )
        verts_world = np.stack(
            [_transform_pts(verts_cam0[t], E_world_from_cam0) for t in range(T)]
        )
        np.save(out_dir / f"joints_cam0_{side}.npy", joints_cam0)
        np.save(out_dir / f"joints_world_{side}.npy", joints_world)
        np.save(out_dir / f"verts_world_{side}.npy", verts_world)
        np.save(out_dir / f"faces_{side}.npy", faces)
        np.save(out_dir / f"valid_{side}.npy", valid)
        print(f"[taeksoo] {side}: saved joints/verts (T={T}, valid={int(valid.sum())})")

        rt = _build_retargeter(
            side, args.retarget_type, out_dir,
            scaling_factor=args.scaling_factor,
            low_pass_alpha=args.low_pass_alpha,
        )
        joint_names = list(rt.optimizer.robot.dof_joint_names)
        qpos = _retarget_frames(joints_cam0, valid, side, rt)
        np.save(out_dir / f"qpos_{side}.npy", qpos)
        (out_dir / f"joint_names_{side}.json").write_text(json.dumps(joint_names))
        print(f"[taeksoo] {side}: qpos shape={qpos.shape} ({len(joint_names)} dof)")

        meta["sides"][side] = {
            "qpos": f"qpos_{side}.npy",
            "joint_names": f"joint_names_{side}.json",
            "joints_world": f"joints_world_{side}.npy",
            "verts_world": f"verts_world_{side}.npy",
            "faces": f"faces_{side}.npy",
            "valid": f"valid_{side}.npy",
            "n_valid": int(valid.sum()),
            "n_dof": len(joint_names),
        }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[taeksoo] done -> {out_dir}")


if __name__ == "__main__":
    main()
