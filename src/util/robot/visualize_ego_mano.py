"""Project MANO hand mesh onto ego camera views and create a side-by-side grid image."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# hamer-mp imports
HAMER_ROOT = Path("/home/temp_id/hamer-mediapipe")
if str(HAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(HAMER_ROOT))

from hamer.models.mano_wrapper import MANO


def load_mano_model(device="cpu"):
    data_root = HAMER_ROOT / "_DATA" / "data"
    mano = MANO(
        model_path=str(data_root / "mano"),
        gender="male",
        num_hand_joints=15,
        mean_params=str(data_root / "mano_mean_params.npz"),
        create_body_pose=False,
    ).to(device)
    mano.eval()
    return mano


def load_fit(fit_path):
    with open(fit_path) as f:
        data = json.load(f)
    fit = data["fit"]
    global_orient = torch.tensor(fit["global_orient"], dtype=torch.float32)
    hand_pose = torch.tensor(fit["hand_pose"], dtype=torch.float32)
    betas = torch.tensor(fit["betas"], dtype=torch.float32)
    transl = torch.tensor(fit["transl"], dtype=torch.float32)
    return global_orient, hand_pose, betas, transl


def get_mano_mesh(mano, global_orient, hand_pose, betas, transl):
    with torch.no_grad():
        out = mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            transl=transl,
            pose2rot=False,
        )
    verts = out.vertices[0].cpu().numpy()  # (778, 3)
    joints = out.joints[0].cpu().numpy()   # (21, 3)
    faces = mano.faces.astype(np.int32)    # (1538, 3)
    return verts, joints, faces


def project_points(pts_3d, K, T_cw):
    """Project 3D world points to 2D pixel coords. K: 3x3, T_cw: 4x4 cam_from_world."""
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    pts_cam = (T_cw[:3] @ pts_h.T).T  # (N, 3)
    z = pts_cam[:, 2]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d, z


def draw_mesh_wireframe(img, pts_2d, z, faces, color=(0, 255, 0), alpha=0.6):
    overlay = img.copy()
    visible = z > 0
    for f in faces:
        if not (visible[f[0]] and visible[f[1]] and visible[f[2]]):
            continue
        tri = pts_2d[f].astype(np.int32)
        cv2.polylines(overlay, [tri], isClosed=True, color=color, thickness=1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_joints(img, pts_2d, z, radius=4, color=(0, 0, 255)):
    for i in range(len(pts_2d)):
        if z[i] <= 0:
            continue
        x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        cv2.circle(img, (x, y), radius, color, -1)
        cv2.putText(img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-root", default="/home/temp_id/shared_data/capture/eccv2026/hand_taeyun/right/apple/0")
    ap.add_argument("--fit-json", default="")
    ap.add_argument("--ego-ids", default="25452062,25452066")
    ap.add_argument("--ego-intrinsic-id", default="22645026")
    ap.add_argument("--frame", default="00001")
    ap.add_argument("--output", default="ego_mano_projection.jpg")
    args = ap.parse_args()

    episode_root = Path(args.episode_root)

    # Load fit JSON
    if args.fit_json:
        fit_path = Path(args.fit_json)
    else:
        fit_path = episode_root / "single_frame_fit_warm_start_one_euro" / f"{args.frame}.json"
    print(f"[FIT] {fit_path}")

    # Load cam params
    cam_param = episode_root / "cam_param"
    intr = json.loads((cam_param / "intrinsics.json").read_text())
    K_ego = np.array(intr[args.ego_intrinsic_id]["intrinsics_undistort"], dtype=np.float64)

    # Load ego extrinsics from PnP results
    ego_pnp_dir = episode_root / "ego_pnp"
    ego_ids = [s.strip() for s in args.ego_ids.split(",")]

    # Load MANO
    mano = load_mano_model()
    global_orient, hand_pose, betas, transl = load_fit(fit_path)
    verts, joints, faces = get_mano_mesh(mano, global_orient, hand_pose, betas, transl)
    print(f"[MANO] vertices={verts.shape}, joints={joints.shape}, faces={faces.shape}")

    # Project onto each ego view
    panels = []
    for eid in ego_ids:
        T_path = ego_pnp_dir / f"{eid}_cam_from_world.npy"
        if not T_path.exists():
            print(f"[SKIP] {T_path} not found")
            continue
        T_cw = np.load(str(T_path))

        img_path = episode_root / "video_extracted" / eid / f"{args.frame}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[SKIP] {img_path} not found")
            continue

        # Project mesh and joints
        verts_2d, verts_z = project_points(verts, K_ego, T_cw)
        joints_2d, joints_z = project_points(joints, K_ego, T_cw)

        # Draw
        img = draw_mesh_wireframe(img, verts_2d, verts_z, faces, color=(0, 255, 0), alpha=0.6)
        img = draw_joints(img, joints_2d, joints_z, radius=5, color=(0, 0, 255))

        # Add label
        cv2.putText(img, eid, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        panels.append(img)
        print(f"[OK] {eid}: {(verts_z > 0).sum()}/{len(verts_z)} vertices visible")

    if not panels:
        print("[FAIL] No panels")
        return

    # Side-by-side grid
    grid = np.concatenate(panels, axis=1)
    out_path = episode_root / args.output
    cv2.imwrite(str(out_path), grid)
    print(f"[SAVED] {out_path} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
