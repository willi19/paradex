"""Single-frame HaMeR + triangulation + MANO fit (with betas optimized).

Run this inside the hamer-mp conda env. Imports helpers from /home/temp_id/hamer-mediapipe.

Inputs (under --scene_dir):
  frames/<cam_id>/<frame_name>
  cam_param/intrinsics.json
  cam_param/extrinsics.json

Output (under --out_path, default scene_dir/hand/mano_fit_betas.json):
  {global_orient, hand_pose, betas, transl, joints, verts, loss}  in world frame
"""
import os
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HAMER_REPO = Path(os.environ.get("HAMER_REPO", "/home/temp_id/hamer-mediapipe"))
sys.path.insert(0, str(HAMER_REPO))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402

from demo_mediapipe import resolve_checkpoint  # noqa: E402
from hamer.configs import CACHE_DIR_HAMER  # noqa: E402
from hamer.models import download_models, load_hamer  # noqa: E402
from hamer.models.mano_wrapper import MANO  # noqa: E402
from run_multiview_frames_triangulate import (  # noqa: E402
    build_cams_for_frame,
    load_cam_params,
    load_ego_cam_params,
    lm_refine_frame,
    process_one_camera,
    triangulate_frame,
)
from fit_mano_to_kpts_3d import (  # noqa: E402
    BETAS_INIT,
    analytical_ik,
    mano_template_joints,
    orthonormalize_rotmat,
)


class _Args:
    """Mimic the argparse namespace process_one_camera expects."""
    one_euro = False
    one_euro_min_cutoff = 0.003
    one_euro_beta = 300.0
    one_euro_d_cutoff = 30.0
    one_euro_max_gap = 3
    pre_median_window = 1
    fps = 30.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True, type=str)
    p.add_argument("--frame_name", default="00001.png", type=str)
    p.add_argument("--hand", choices=["right", "left", "both"], default="both")
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--checkpoint", default=None, type=str)
    p.add_argument("--out_path", default=None, type=str,
                   help="Default: <scene_dir>/hand/mano_fit_betas.json")
    p.add_argument("--min_views", type=int, default=2)
    p.add_argument("--ransac_thresh_px", type=float, default=8.0)
    p.add_argument("--kpt_score_thr", type=float, default=0.0)
    p.add_argument("--lm_refine", action="store_true", default=True)
    p.add_argument("--no_lm_refine", action="store_false", dest="lm_refine")
    p.add_argument("--lm_huber_delta", type=float, default=4.0)
    p.add_argument("--lm_max_iters", type=int, default=50)

    p.add_argument("--iters_transl", type=int, default=800)
    p.add_argument("--iters_orient", type=int, default=800)
    p.add_argument("--iters_pose", type=int, default=1000)
    p.add_argument("--iters_betas", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lambda_pose_prior", type=float, default=1e-3)
    p.add_argument("--lambda_betas_prior", type=float, default=1e-2)
    p.add_argument("--root_weight", type=float, default=5.0,
                   help="Weight on wrist+MCP joints (0,5,9,13,17) in MANO fit data term.")
    return p.parse_args()


def fit_with_betas(mano, kpts_3d_np: np.ndarray, device: torch.device, args) -> dict:
    """Fit MANO (correct chirality model passed in) to 3D keypoints in world frame."""
    kpts_for_fit = kpts_3d_np.copy()
    recon = torch.tensor(kpts_for_fit, dtype=torch.float32, device=device).unsqueeze(0)
    betas_init = torch.tensor(BETAS_INIT, dtype=torch.float32, device=device).unsqueeze(0)
    template_joints = mano_template_joints(mano, betas_init, device)

    pose_R = analytical_ik(template_joints, kpts_for_fit)
    init_go = torch.tensor(pose_R[0:1], dtype=torch.float32, device=device).unsqueeze(0)
    init_hp = torch.tensor(pose_R[1:16], dtype=torch.float32, device=device).unsqueeze(0)

    global_orient = nn.Parameter(init_go.clone())
    hand_pose = nn.Parameter(init_hp.clone())
    transl = nn.Parameter(torch.tensor(kpts_for_fit[0], dtype=torch.float32, device=device).unsqueeze(0))
    betas = nn.Parameter(betas_init.clone())
    ik_hp = init_hp.detach()

    def forward():
        return mano.forward(
            global_orient=global_orient.float(),
            hand_pose=hand_pose.float(),
            betas=betas.float(),
            transl=transl.float(),
            pose2rot=False, use_pca=False,
        )

    root_idx = [0, 5, 9, 13, 17]
    weights = torch.ones(21, dtype=torch.float32, device=device)
    for ri in root_idx:
        weights[ri] = float(args.root_weight)
    weights = weights.view(1, 21, 1)

    def step(opt, n_iters, stage):
        for _ in range(n_iters):
            out = forward()
            if stage == "transl":
                data = torch.mean((out.joints - recon)[:, root_idx, :] ** 2)
            else:
                diff = (out.joints - recon) ** 2
                data = torch.mean(weights * diff)
            loss = data
            if stage in ("pose", "betas") and args.lambda_pose_prior > 0:
                loss = loss + args.lambda_pose_prior * torch.mean((hand_pose - ik_hp) ** 2)
            if stage == "betas" and args.lambda_betas_prior > 0:
                loss = loss + args.lambda_betas_prior * torch.mean(betas ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                global_orient.copy_(orthonormalize_rotmat(global_orient))
                hand_pose.copy_(orthonormalize_rotmat(hand_pose))

    if args.iters_transl > 0:
        step(torch.optim.Adam([transl], lr=args.lr), args.iters_transl, "transl")
    if args.iters_orient > 0:
        step(torch.optim.Adam([global_orient, transl], lr=args.lr), args.iters_orient, "orient")
    if args.iters_pose > 0:
        step(torch.optim.Adam([global_orient, transl, hand_pose], lr=args.lr), args.iters_pose, "pose")
    if args.iters_betas > 0:
        step(torch.optim.Adam([global_orient, transl, hand_pose, betas], lr=args.lr), args.iters_betas, "betas")

    with torch.no_grad():
        out = forward()
        joints_np = out.joints[0].detach().cpu().numpy()
        verts_np = out.vertices[0].detach().cpu().numpy()
        final_loss = float(torch.mean((out.joints - recon) ** 2).item())

    return {
        "global_orient": global_orient.detach().cpu().numpy(),
        "hand_pose": hand_pose.detach().cpu().numpy(),
        "betas": betas.detach().cpu().numpy(),
        "transl": transl.detach().cpu().numpy(),
        "joints": joints_np,
        "verts": verts_np,
        "loss": final_loss,
    }


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    out_path = Path(args.out_path) if args.out_path else (scene_dir / "hand" / "mano_fit_betas.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load cam params
    cams_static = load_cam_params(scene_dir)
    cams_ego = load_ego_cam_params(scene_dir)
    ego_ids = set(cams_ego.keys())
    known_ids = set(cams_static.keys()) | ego_ids

    frames_root = scene_dir / "frames"
    cam_dirs_all = sorted([p for p in frames_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    cam_ids_used = [p.name for p in cam_dirs_all if p.name in known_ids and p.name not in ego_ids]
    if not cam_ids_used:
        raise RuntimeError(f"no usable cams in {scene_dir}")
    cam_frame_dirs = {cid: frames_root / cid for cid in cam_ids_used}
    print(f"[INFO] cams={len(cam_ids_used)} frame={args.frame_name}")

    # HaMeR + MediaPipe
    download_models(CACHE_DIR_HAMER)
    ckpt = resolve_checkpoint(args.checkpoint)
    hamer_model, hamer_cfg = load_hamer(ckpt)
    hamer_model = hamer_model.to(device).eval()
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5,
    )
    mano_models = {}
    for hand_label, is_rhand in (("right", True), ("left", False)):
        mano_models[hand_label] = MANO(
            model_path=str(HAMER_REPO / "_DATA" / "data" / "mano"),
            is_rhand=is_rhand,
            gender="neutral", num_hand_joints=15,
            mean_params=str(HAMER_REPO / "_DATA" / "data" / "mano_mean_params.npz"),
            create_body_pose=False,
        ).to(device).eval()

    hands_to_run = ["right", "left"] if args.hand == "both" else [args.hand]
    out_per_hand = {}
    pco_args = _Args()
    for hand in hands_to_run:
        mano = mano_models[hand]
        print(f"\n========== HAND={hand} ==========")
        # Phase A: per-cam HaMeR for the single frame
        per_cam_results = {}
        for i, cid in enumerate(cam_ids_used, start=1):
            desc = f"[{hand} {i}/{len(cam_ids_used)}] cam {cid}"
            res = process_one_camera(
                cam_frame_dirs[cid], [args.frame_name], hamer_model, hamer_cfg,
                device, mp_hands, hand, pco_args, progress_desc=desc,
            )
            per_cam_results[cid] = res

        # Phase B: triangulate
        cams_fn = build_cams_for_frame(cams_static, cams_ego, args.frame_name)
        per_cam_2d = {}
        per_cam_score = {}
        for cid in cam_ids_used:
            if cid not in cams_fn:
                continue
            r = per_cam_results[cid].get(args.frame_name)
            if r is not None and r["detected"]:
                per_cam_2d[cid] = r["keypoints"]
                per_cam_score[cid] = r["score"]
        if len(per_cam_2d) < args.min_views:
            print(f"[WARN] {hand}: only {len(per_cam_2d)} cams detected; skipping")
            out_per_hand[hand] = {"detected": False, "n_views_detected": len(per_cam_2d)}
            continue
        X, inliers = triangulate_frame(
            per_cam_2d, per_cam_score, cams_fn,
            min_views=args.min_views, ransac_thresh_px=args.ransac_thresh_px,
            score_thr=args.kpt_score_thr,
        )
        if not np.all(np.isfinite(X)):
            print(f"[WARN] {hand}: triangulation has NaNs; skipping")
            out_per_hand[hand] = {"detected": False, "n_views_detected": len(per_cam_2d)}
            continue
        if args.lm_refine:
            X = lm_refine_frame(X, inliers, per_cam_2d, cams_fn,
                                huber_delta=args.lm_huber_delta, max_iters=args.lm_max_iters)
            if not np.all(np.isfinite(X)):
                print(f"[WARN] {hand}: lm_refine produced NaNs, fallback to RANSAC result")
        n_inl_per_joint = [len(c) for c in inliers]
        print(f"[{hand}] triangulated 21 kpts, mean depth={np.mean(X[:, 2]):.3f}, "
              f"inliers/joint min/mean/max={min(n_inl_per_joint)}/{np.mean(n_inl_per_joint):.1f}/{max(n_inl_per_joint)}")

        # Phase C: MANO fit with betas — uses chirality-correct MANO model
        fit = fit_with_betas(mano, X, device, args)
        print(f"[{hand}] mano fit loss={fit['loss']:.6e}")

        betas_t = torch.tensor(fit["betas"], dtype=torch.float32, device=device)
        if betas_t.dim() == 1:
            betas_t = betas_t.unsqueeze(0)
        J0_betas = mano_template_joints(mano, betas_t, device)[0]

        faces_h = np.asarray(mano.faces, dtype=np.int32).tolist() if hasattr(mano, "faces") else None
        out_per_hand[hand] = {
            "detected": True,
            "frame_name": args.frame_name,
            "kpts_3d_world": X.tolist(),
            "global_orient": fit["global_orient"].tolist(),
            "hand_pose": fit["hand_pose"].tolist(),
            "betas": fit["betas"].tolist(),
            "transl": fit["transl"].tolist(),
            "joints_world": fit["joints"].tolist(),
            "verts_world": fit["verts"].tolist(),
            "J0_betas": J0_betas.tolist(),
            "mano_faces": faces_h,
            "loss": fit["loss"],
            "n_views_detected": len(per_cam_2d),
        }

    mp_hands.close()

    out = {"frame_name": args.frame_name, "hands": out_per_hand}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\n[DONE] saved {out_path}")


if __name__ == "__main__":
    main()
