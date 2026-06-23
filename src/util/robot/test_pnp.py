"""Ego camera extrinsic calibration via PnP — all frames.

All exo cameras are already calibrated (cam_param/). For each frame:
  1. DKM-match ego vs all exo.
  2. Cluster ego 2D points seen in >=2 exo views.
  3. Multi-view DLT triangulate 3D points from exo cameras.
  4. PnP RANSAC (AP3P, min 4 inliers) -> ego pose.
  5. MANO projection overlay saved per frame.
  6. Viser trajectory visualization at the end.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

HAMER_ROOT = Path("/home/temp_id/hamer-mediapipe")
if str(HAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(HAMER_ROOT))


# ---------------------------------------------------------------------------
# Camera parameter loaders
# ---------------------------------------------------------------------------

def load_cam_params(cam_param_dir: Path):
    intr = json.loads((cam_param_dir / "intrinsics.json").read_text())
    extr = json.loads((cam_param_dir / "extrinsics.json").read_text())
    cameras = {}
    for cam_id, v in intr.items():
        K = np.array(v["intrinsics_undistort"], dtype=np.float64)
        cameras[cam_id] = {
            "intrinsics_undistort": K.tolist(),
            "width": v["width"], "height": v["height"],
        }
    exo_images = {}
    for cam_id in extr:
        if cam_id not in cameras:
            continue
        T = np.eye(4, dtype=np.float64)
        T[:3] = np.array(extr[cam_id], dtype=np.float64)
        exo_images[f"{cam_id}.jpg"] = {"camera_id": cam_id, "cam_from_world": T}
    return cameras, exo_images


# ---------------------------------------------------------------------------
# Feature Matchers
# ---------------------------------------------------------------------------

class DKMMatcher:
    def __init__(self, device="cuda"):
        from dkm.models import DKMv3_indoor
        self.device = device
        self.model = DKMv3_indoor(device=device)
        print(f"[DKM] DKMv3_indoor device={device}")

    def match(self, p0, p1, conf=0.5, num_samples=10000):
        dense_matches, dense_certainty = self.model.match(str(p0), str(p1))
        matches, certainty = self.model.sample(dense_matches, dense_certainty, num=num_samples)
        cert = certainty.cpu().numpy()
        mask = cert >= conf
        m = matches[mask].cpu().numpy()
        img0 = cv2.imread(str(p0))
        img1 = cv2.imread(str(p1))
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        kp0 = np.stack([(m[:, 0] + 1) / 2 * w0, (m[:, 1] + 1) / 2 * h0], axis=1)
        kp1 = np.stack([(m[:, 2] + 1) / 2 * w1, (m[:, 3] + 1) / 2 * h1], axis=1)
        return kp0, kp1


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def triangulate_multiview(Ks, Ts, pts2d):
    n = len(Ks)
    A = np.zeros((2 * n, 4), dtype=np.float64)
    for i in range(n):
        P = Ks[i] @ Ts[i][:3]
        x, y = pts2d[i]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def check_reproj(pt3d, K, T, pt2d, thresh):
    ph = np.append(pt3d, 1.0)
    if (T[:3] @ ph)[2] <= 0:
        return False, float("inf")
    proj = K @ (T[:3] @ ph)
    proj = proj[:2] / proj[2]
    err = np.linalg.norm(proj - pt2d)
    return err < thresh, err


def triangulation_angle_deg(Ts, pt3d):
    centers = [np.linalg.inv(T)[:3, 3] for T in Ts]
    min_angle = 180.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d1 = pt3d - centers[i]
            d2 = pt3d - centers[j]
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 < 1e-9 or n2 < 1e-9:
                return 0.0
            cos_a = np.clip(np.dot(d1, d2) / (n1 * n2), -1, 1)
            min_angle = min(min_angle, np.degrees(np.arccos(cos_a)))
    return min_angle


def robust_triangulate(Ks, Ts, pts2d, reproj_thresh=3.0, min_angle=1.5):
    n = len(Ks)
    if n < 2:
        return None, None
    pt3d = triangulate_multiview(Ks, Ts, pts2d)
    if pt3d is None:
        return None, None
    errs = []
    for i in range(n):
        _, e = check_reproj(pt3d, Ks[i], Ts[i], pts2d[i], float("inf"))
        errs.append(e)
    errs = np.array(errs)
    mask = np.ones(n, dtype=bool)
    for _ in range(max(0, n - 2)):
        worst = np.argmax(errs * mask)
        if errs[worst] <= reproj_thresh:
            break
        mask[worst] = False
        if mask.sum() < 2:
            break
        idxs = np.where(mask)[0]
        pt3d = triangulate_multiview(
            [Ks[i] for i in idxs], [Ts[i] for i in idxs],
            [pts2d[i] for i in idxs])
        if pt3d is None:
            return None, None
        for i in idxs:
            _, errs[i] = check_reproj(pt3d, Ks[i], Ts[i], pts2d[i], float("inf"))
    inlier_idxs = np.where(mask)[0]
    for i in inlier_idxs:
        ok, _ = check_reproj(pt3d, Ks[i], Ts[i], pts2d[i], reproj_thresh)
        if not ok:
            return None, None
    if triangulation_angle_deg([Ts[i] for i in inlier_idxs], pt3d) < min_angle:
        return None, None
    for i in inlier_idxs:
        if (Ts[i][:3] @ np.append(pt3d, 1.0))[2] <= 0:
            return None, None
    return pt3d, mask


# ---------------------------------------------------------------------------
# PnP
# ---------------------------------------------------------------------------

def _match_and_triangulate(matcher, ego_path, exo_paths, exo_images, cameras,
                           conf, ego_radius, triang_reproj):
    from scipy.spatial import cKDTree
    observations = []
    for xn, xp in exo_paths.items():
        ego_pts, exo_pts = matcher.match(ego_path, xp, conf)
        if len(ego_pts) < 5:
            continue
        for k in range(len(ego_pts)):
            observations.append((ego_pts[k], xn, exo_pts[k]))
    if len(observations) < 10:
        return [], []
    ego_2d_all = np.array([o[0] for o in observations])
    tree = cKDTree(ego_2d_all)
    visited = set()
    clusters = []
    for idx in range(len(observations)):
        if idx in visited:
            continue
        neighbors = tree.query_ball_point(ego_2d_all[idx], r=ego_radius)
        seen = {}
        for n in neighbors:
            en = observations[n][1]
            if en not in seen:
                seen[en] = n
        cluster = list(seen.values())
        if len(cluster) >= 2:
            clusters.append(cluster)
            visited.update(cluster)
    corr_2d, corr_3d = [], []
    for cluster in clusters:
        ego_2ds, view_Ks, view_Ts, view_pts = [], [], [], []
        for idx in cluster:
            ego_2d, en, exo_2d = observations[idx]
            cid = exo_images[en]["camera_id"]
            ego_2ds.append(ego_2d)
            view_Ks.append(np.array(cameras[cid]["intrinsics_undistort"], dtype=np.float64))
            view_Ts.append(exo_images[en]["cam_from_world"])
            view_pts.append(exo_2d)
        pt3d, inlier_mask = robust_triangulate(
            view_Ks, view_Ts, view_pts, reproj_thresh=triang_reproj, min_angle=1.5)
        if pt3d is None:
            continue
        inlier_ego_2ds = [ego_2ds[i] for i in range(len(ego_2ds)) if inlier_mask[i]]
        corr_2d.append(np.mean(inlier_ego_2ds, axis=0))
        corr_3d.append(pt3d)
    return corr_2d, corr_3d


def _run_pnp(pts_2d, pts_3d, K, reproj, name=""):
    best = None
    for flag in [cv2.SOLVEPNP_AP3P, cv2.SOLVEPNP_EPNP]:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, None,
            reprojectionError=reproj, iterationsCount=100000, flags=flag,
        )
        if not ok or inliers is None or len(inliers) < 4:
            continue
        inl = inliers.ravel()
        use_rvec, use_tvec = rvec.copy(), tvec.copy()
        for _ in range(5):
            use_rvec, use_tvec = cv2.solvePnPRefineLM(
                pts_3d[inl], pts_2d[inl], K, None,
                rvec=use_rvec.copy(), tvec=use_tvec.copy(),
            )
            proj_all, _ = cv2.projectPoints(pts_3d, use_rvec, use_tvec, K, None)
            errs_all = np.linalg.norm(pts_2d - proj_all.reshape(-1, 2), axis=1)
            new_inl = np.where(errs_all < reproj)[0]
            if len(new_inl) < 4:
                break
            if set(new_inl) == set(inl):
                inl = new_inl
                break
            inl = new_inl
        proj, _ = cv2.projectPoints(pts_3d[inl], use_rvec, use_tvec, K, None)
        err = np.linalg.norm(pts_2d[inl] - proj.reshape(-1, 2), axis=1).mean()
        if err > reproj * 3:
            continue
        if best is None or len(inl) > best[1] or (len(inl) == best[1] and err < best[2]):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = cv2.Rodrigues(use_rvec)[0]
            T[:3, 3] = use_tvec.ravel()
            best = (T, len(inl), err)
    if best is None:
        return None
    T, ni, err = best
    pos = np.linalg.inv(T)[:3, 3]
    print(f"  [PnP] {name}: {ni} inliers, reproj={err:.2f}px, pos=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}]")
    return best


def calibrate_ego_frame(matcher, frame_name, img_dir, ego_ids, ego_K,
                        cameras, exo_images, conf, ego_radius,
                        triang_reproj, pnp_reproj, stereo_max_dist=0.10):
    """Calibrate ego cameras for a single frame. Returns dict {ego_id: T_4x4}."""
    from scipy.spatial import cKDTree

    exo_paths = {}
    for xn in sorted(exo_images.keys()):
        stem = xn.replace(".jpg", "")
        p = img_dir / stem / frame_name
        if p.exists():
            exo_paths[xn] = p

    ego_data = {}
    for eid in sorted(ego_ids):
        ego_path = img_dir / eid / frame_name
        if not ego_path.exists():
            continue
        corr_2d, corr_3d = _match_and_triangulate(
            matcher, ego_path, exo_paths, exo_images, cameras,
            conf, ego_radius, triang_reproj)
        if len(corr_2d) < 4:
            continue
        pts_2d = np.array(corr_2d, dtype=np.float64)
        pts_3d = np.array(corr_3d, dtype=np.float64)
        pnp = _run_pnp(pts_2d, pts_3d, ego_K, pnp_reproj, eid)
        ego_data[eid] = {"pts_2d": pts_2d, "pts_3d": pts_3d, "pnp": pnp}

    names = sorted(ego_data.keys())
    results = {}

    if len(names) == 2:
        n0, n1 = names
        pnp0, pnp1 = ego_data[n0].get("pnp"), ego_data[n1].get("pnp")
        if pnp0 is not None and pnp1 is not None:
            anchor = n0 if pnp0[1] > pnp1[1] else n1
        elif pnp0 is not None:
            anchor = n0
        elif pnp1 is not None:
            anchor = n1
        else:
            return results

        other = n1 if anchor == n0 else n0
        T_anchor = ego_data[anchor]["pnp"][0]
        results[anchor] = T_anchor

        # Stereo transfer
        p_anc = img_dir / anchor / frame_name
        p_oth = img_dir / other / frame_name
        if p_anc.exists() and p_oth.exists():
            pts_anc, pts_oth = matcher.match(p_anc, p_oth, conf)
            if len(pts_anc) >= 10:
                da = ego_data[anchor]
                tree_anc = cKDTree(pts_anc)
                transfer_2d, transfer_3d = [], []
                for i in range(len(da["pts_2d"])):
                    dist_nn, idx_nn = tree_anc.query(da["pts_2d"][i], k=1)
                    if dist_nn < ego_radius:
                        transfer_2d.append(pts_oth[idx_nn])
                        transfer_3d.append(da["pts_3d"][i])
                od = ego_data.get(other, {})
                all_2d = list(transfer_2d) + list(od.get("pts_2d", []))
                all_3d = list(transfer_3d) + list(od.get("pts_3d", []))
                if len(all_2d) >= 4:
                    pnp_o = _run_pnp(np.array(all_2d), np.array(all_3d),
                                     ego_K, pnp_reproj, f"{other}(stereo)")
                    if pnp_o is not None:
                        T_o = pnp_o[0]
                        pos_a = np.linalg.inv(T_anchor)[:3, 3]
                        pos_o = np.linalg.inv(T_o)[:3, 3]
                        dist = np.linalg.norm(pos_a - pos_o)
                        if dist <= stereo_max_dist:
                            results[other] = T_o

        if other not in results and ego_data.get(other, {}).get("pnp") is not None:
            results[other] = ego_data[other]["pnp"][0]
    else:
        for name in names:
            if ego_data[name]["pnp"] is not None:
                results[name] = ego_data[name]["pnp"][0]

    return results


# ---------------------------------------------------------------------------
# MANO projection
# ---------------------------------------------------------------------------

def load_mano_model(device="cpu"):
    from hamer.models.mano_wrapper import MANO
    data_root = HAMER_ROOT / "_DATA" / "data"
    mano = MANO(
        model_path=str(data_root / "mano"),
        gender="male", num_hand_joints=15,
        mean_params=str(data_root / "mano_mean_params.npz"),
        create_body_pose=False,
    ).to(device)
    mano.eval()
    return mano


def get_mano_mesh(mano, fit):
    with torch.no_grad():
        out = mano(
            global_orient=torch.tensor(fit["global_orient"], dtype=torch.float32),
            hand_pose=torch.tensor(fit["hand_pose"], dtype=torch.float32),
            betas=torch.tensor(fit["betas"], dtype=torch.float32),
            transl=torch.tensor(fit["transl"], dtype=torch.float32),
            pose2rot=False,
        )
    return out.vertices[0].cpu().numpy(), out.joints[0].cpu().numpy(), mano.faces.astype(np.int32)


def project_points(pts_3d, K, T_cw):
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    pts_cam = (T_cw[:3] @ pts_h.T).T
    z = pts_cam[:, 2]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d, z


_nvdiff_glctx = None

def _get_nvdiff_glctx():
    global _nvdiff_glctx
    if _nvdiff_glctx is None:
        import nvdiffrast.torch as dr
        _nvdiff_glctx = dr.RasterizeCudaContext()
    return _nvdiff_glctx


def draw_mesh_overlay(img, verts, faces, K, T_cw, mesh_color=(1.0, 1.0, 0.9)):
    """Render MANO mesh with nvdiffrast + simple shading, composite onto image."""
    import nvdiffrast.torch as dr

    device = torch.device('cuda')
    h, w = img.shape[:2]
    glctx = _get_nvdiff_glctx()

    # Transform verts to camera space
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_cam = (T_cw[:3] @ verts_h.T).T  # (V, 3)

    # Build clip-space projection matrix from K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    near, far = 0.01, 10.0
    # OpenGL-style projection: flip y,z for GL convention, then project
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = 2 * fx / w
    proj[0, 2] = 1 - 2 * cx / w
    proj[1, 1] = 2 * fy / h
    proj[1, 2] = 2 * cy / h - 1
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1.0

    # Flip y,z for OpenGL convention
    verts_gl = verts_cam.copy()
    verts_gl[:, 1] *= -1
    verts_gl[:, 2] *= -1

    # To clip space
    verts_clip = (proj @ np.hstack([verts_gl, np.ones((len(verts_gl), 1))]).T).T

    verts_clip_t = torch.tensor(verts_clip, dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    faces_t = torch.tensor(faces.astype(np.int32).copy(), dtype=torch.int32, device=device).contiguous()

    # Rasterize
    rast, _ = dr.rasterize(glctx, verts_clip_t, faces_t, resolution=[h, w])

    # Compute face normals in camera space for shading
    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v1)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    face_normals = face_normals / norms

    # Per-vertex normals (average of adjacent face normals)
    vert_normals = np.zeros_like(verts_cam)
    for i in range(3):
        np.add.at(vert_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vert_normals = vert_normals / norms
    # Flip to GL convention
    vert_normals_gl = vert_normals.copy()
    vert_normals_gl[:, 1] *= -1
    vert_normals_gl[:, 2] *= -1

    normals_t = torch.tensor(vert_normals_gl, dtype=torch.float32, device=device).unsqueeze(0)
    normals_interp, _ = dr.interpolate(normals_t, rast, faces_t)
    normals_interp = torch.nn.functional.normalize(normals_interp, dim=-1)

    # 3-point lighting (Raymond-style)
    light_dirs = []
    for phi, theta in zip([0, 2*np.pi/3, 4*np.pi/3], [np.pi/6]*3):
        d = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        light_dirs.append(d / np.linalg.norm(d))
    ambient = 0.3
    diffuse = 0.0
    for ld in light_dirs:
        ld_t = torch.tensor(ld, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        diffuse = diffuse + torch.clamp(torch.sum(normals_interp * ld_t, dim=-1, keepdim=True), 0, 1) / 3.0

    shade = ambient + 0.7 * diffuse  # (1, H, W, 1)
    color = torch.tensor(mesh_color, dtype=torch.float32, device=device).view(1, 1, 1, 3)
    shaded = (shade * color).clamp(0, 1)

    # Alpha mask from rasterization
    alpha = (rast[..., 3:4] > 0).float()

    # nvdiffrast output is bottom-up (OpenGL), flip to top-down (OpenCV)
    shaded = torch.flip(shaded, [1])
    alpha = torch.flip(alpha, [1])

    # Composite
    img_t = torch.tensor(img[:, :, ::-1].copy(), dtype=torch.float32, device=device).unsqueeze(0) / 255.0  # BGR->RGB
    out = shaded * alpha + img_t * (1 - alpha)
    out_np = (out[0].cpu().numpy() * 255).astype(np.uint8)
    return out_np[:, :, ::-1].copy()  # RGB->BGR


def render_mano_projection(frame_name, ego_ids, ego_Ts, ego_K, fit, mano,
                           img_dir, out_path):
    """Render MANO mesh on ego views, save side-by-side image."""
    verts, joints, faces = get_mano_mesh(mano, fit)
    panels = []
    for eid in ego_ids:
        if eid not in ego_Ts:
            continue
        img = cv2.imread(str(img_dir / eid / frame_name))
        if img is None:
            continue
        img = draw_mesh_overlay(img, verts, faces, ego_K, ego_Ts[eid])
        cv2.putText(img, eid, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        panels.append(img)
    if panels:
        grid = np.concatenate(panels, axis=1)
        cv2.imwrite(str(out_path), grid)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_trajectory(exo_images, cameras, ego_intrinsic_id,
                         all_ego_Ts, ego_ids, frustum_size=0.05):
    """Show exo cameras + ego camera trajectory in viser."""
    from paradex.visualization.visualizer.viser import ViserViewer

    vis = ViserViewer(scene_title="ego_trajectory")
    vis.add_floor(height=0.0)

    # Exo cameras (static, green)
    for name, info in sorted(exo_images.items()):
        cid = info["camera_id"]
        if cid not in cameras:
            continue
        vis.add_camera(
            name=f"exo_{name.split('.')[0]}",
            extrinsic=np.linalg.inv(info["cam_from_world"]),
            intrinsic=cameras[cid], color=(0, 220, 0),
            size=frustum_size, show_axes=False)

    # Ego camera trajectories
    colors = [(220, 0, 0), (0, 0, 220)]
    for ci, eid in enumerate(sorted(ego_ids)):
        color = colors[ci % len(colors)]
        positions = []
        for frame_idx, ego_Ts in enumerate(all_ego_Ts):
            if eid not in ego_Ts:
                continue
            T_cw = ego_Ts[eid]
            T_wc = np.linalg.inv(T_cw)
            pos = T_wc[:3, 3]
            positions.append(pos)

            # Add camera frustum for sampled frames (every 10th + first + last)
            if frame_idx == 0 or frame_idx == len(all_ego_Ts) - 1 or frame_idx % 10 == 0:
                vis.add_camera(
                    name=f"ego_{eid}/frame_{frame_idx:05d}",
                    extrinsic=T_wc,
                    intrinsic=cameras[ego_intrinsic_id],
                    color=color, size=frustum_size * 0.5, show_axes=False)

        # Draw trajectory line
        if len(positions) >= 2:
            positions = np.array(positions)
            color_norm = tuple(c / 255.0 for c in color)
            vis.server.scene.add_spline_catmull_rom(
                f"/trajectory/{eid}",
                positions=positions,
                color=color_norm,
                line_width=3.0,
            )
        print(f"[VIS] {eid}: {len(positions)} frames in trajectory")

    vis.start_viewer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-root", default="/home/temp_id/shared_data/capture/eccv2026/hand_taeyun/right/apple/0")
    ap.add_argument("--ego-ids", default="25452062,25452066")
    ap.add_argument("--ego-intrinsic-id", default="22645026")
    ap.add_argument("--match-conf", type=float, default=0.9)
    ap.add_argument("--ego-radius", type=float, default=10.0)
    ap.add_argument("--triang-reproj", type=float, default=3.0)
    ap.add_argument("--pnp-reproj", type=float, default=2.0)
    ap.add_argument("--start-frame", type=int, default=1)
    ap.add_argument("--end-frame", type=int, default=-1)
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--vis-frustum-size", type=float, default=0.05)
    args = ap.parse_args()

    episode_root = Path(args.episode_root).expanduser().resolve()
    cam_param_dir = episode_root / "cam_param"
    img_dir = episode_root / "video_extracted"
    fit_dir = episode_root / "single_frame_fit_warm_start_one_euro"

    cameras, exo_images = load_cam_params(cam_param_dir)
    ego_K = np.array(cameras[args.ego_intrinsic_id]["intrinsics_undistort"], dtype=np.float64)
    ego_ids = [s.strip() for s in args.ego_ids.split(",")]

    # Determine frame range
    fit_files = sorted(fit_dir.glob("*.json"))
    if not fit_files:
        print("[ERROR] No fit JSON files found")
        return
    max_frame = int(fit_files[-1].stem)
    start = args.start_frame
    end = args.end_frame if args.end_frame > 0 else max_frame
    print(f"[INFO] exo={len(exo_images)}, frames={start}-{end}, ego={ego_ids}")

    # Init matcher
    matcher = DKMMatcher(device="cuda")

    # Init MANO
    mano = load_mano_model()

    # Output dirs
    out_pnp = episode_root / "ego_pnp"
    out_pnp.mkdir(parents=True, exist_ok=True)
    out_proj = episode_root / "ego_mano_projections"
    out_proj.mkdir(parents=True, exist_ok=True)

    # Process all frames
    all_ego_Ts = []  # list of {ego_id: T_4x4} per frame
    for fi in range(start, end + 1):
        frame_name = f"{fi:05d}.jpg"
        fit_path = fit_dir / f"{fi:05d}.json"

        print(f"\n{'='*60}")
        print(f"[FRAME {fi:05d}]")

        # Check ego images exist
        ego_exists = all((img_dir / eid / frame_name).exists() for eid in ego_ids)
        if not ego_exists:
            print(f"  Ego images missing, skip")
            all_ego_Ts.append({})
            continue

        # PnP calibration
        ego_Ts = calibrate_ego_frame(
            matcher, frame_name, img_dir, ego_ids, ego_K,
            cameras, exo_images,
            conf=args.match_conf,
            ego_radius=args.ego_radius,
            triang_reproj=args.triang_reproj,
            pnp_reproj=args.pnp_reproj,
        )

        # Save PnP results
        for eid, T in ego_Ts.items():
            np.save(str(out_pnp / f"{eid}_{fi:05d}_cam_from_world.npy"), T)

        # Summary
        for eid in ego_ids:
            if eid in ego_Ts:
                pos = np.linalg.inv(ego_Ts[eid])[:3, 3]
                print(f"  {eid}: pos=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}]")
            else:
                print(f"  {eid}: FAILED")

        if len(ego_Ts) == 2:
            positions = [np.linalg.inv(ego_Ts[e])[:3, 3] for e in sorted(ego_Ts.keys())]
            print(f"  stereo dist={np.linalg.norm(positions[0] - positions[1]):.4f}m")

        # MANO projection
        if fit_path.exists() and ego_Ts:
            with open(fit_path) as f:
                fit = json.load(f)["fit"]
            render_mano_projection(
                frame_name, ego_ids, ego_Ts, ego_K, fit, mano,
                img_dir, out_proj / f"{fi:05d}.jpg")

        all_ego_Ts.append(ego_Ts)

    # Final summary
    print(f"\n{'='*60}")
    n_ok = sum(1 for ts in all_ego_Ts if len(ts) == len(ego_ids))
    n_partial = sum(1 for ts in all_ego_Ts if 0 < len(ts) < len(ego_ids))
    n_fail = sum(1 for ts in all_ego_Ts if len(ts) == 0)
    print(f"[DONE] {n_ok} full / {n_partial} partial / {n_fail} failed out of {len(all_ego_Ts)} frames")
    print(f"[SAVED] PnP: {out_pnp}")
    print(f"[SAVED] Projections: {out_proj}")

    # Visualize trajectory
    if args.vis:
        visualize_trajectory(
            exo_images, cameras, args.ego_intrinsic_id,
            all_ego_Ts, ego_ids, args.vis_frustum_size)


if __name__ == "__main__":
    main()
