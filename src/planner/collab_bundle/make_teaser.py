"""Render teaser 3D video: predicted camera + hands + MoGe-2 point cloud of frame 0.

Given a prediction NPZ (from infer_regress.py) and the source video, produces
an MP4 of the 3D scene from a pulled-back ego viewpoint showing:
  - predicted L/R MANO hand meshes (frame-0 camera space)
  - predicted ego-camera frustum trail (uses pred_cam focal)
  - frame-0 colored point cloud from MoGe-2 depth

Everything lives in frame-0 camera space (identity at frame 0). Output is just
the right panel (3D view); composite the left panel (input image + caption)
yourself.

Usage:
  python data_processing/make_teaser.py \
    --npz temp/egodex__part5_stack_unstack_tupperware__142_resized_clip00320_pred.npz \
    --video /workspace/project/datasets/egodex_jhb_vclab_taeksoo/egodex/clips/part5_stack_unstack_tupperware/142_resized_clip00320.mp4 \
    --output temp/teaser_right.mp4
"""

import argparse
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import numpy as np
import pyrender
import trimesh
import torch
from scipy.spatial.transform import Rotation


def make_frustum_corners(c2w, scale=0.05, aspect=1.0, fov_y=1.0):
    half_h = np.tan(fov_y / 2) * scale
    half_w = half_h * aspect
    corners_cam = np.array([
        [0, 0, 0],
        [-half_w, -half_h, scale],
        [half_w, -half_h, scale],
        [half_w, half_h, scale],
        [-half_w, half_h, scale],
    ])
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return (R @ corners_cam.T).T + t


def frustum_wireframe(corners, radius=0.0012):
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    paths = [trimesh.creation.cylinder(radius=radius, segment=[corners[i], corners[j]])
             for i, j in edges]
    return trimesh.util.concatenate(paths)


def run_moge_frame0(frame_rgb, fov_x_deg=None):
    """Run MoGe-2 on a single RGB frame. Returns (depth HxW, fx, fy, cx, cy).

    If `fov_x_deg` is provided, MoGe calibrates depth to that horizontal FOV.
    Use this when you have GT intrinsics — especially for out-of-distribution
    setups (wide FOV, square images) where MoGe's own focal estimate is off.
    """
    from moge.model.v2 import MoGeModel
    print("Loading MoGe-2...")
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl').cuda().eval()

    img_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        if fov_x_deg is not None:
            result = model.infer(img_t.cuda(), fov_x=float(fov_x_deg))
        else:
            result = model.infer(img_t.cuda())

    depth = result['depth'].cpu().numpy()
    # MoGe returns intrinsics normalized to [0,1] w.r.t. (W, H).
    K_norm = result['intrinsics'].cpu().numpy()
    H, W = frame_rgb.shape[:2]
    fx = float(K_norm[0, 0] * W)
    fy = float(K_norm[1, 1] * H)
    cx = float(K_norm[0, 2] * W)
    cy = float(K_norm[1, 2] * H)

    del model
    torch.cuda.empty_cache()
    print(f"MoGe-2: depth {depth.shape}, fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
          + (f"  [fov_x hint={fov_x_deg:.1f}°]" if fov_x_deg is not None else ""))
    return depth, fx, fy, cx, cy


def unproject_points(rgb, depth, fx, fy, cx, cy, stride=2, max_depth=None):
    """Unproject a depth map to colored 3D points in camera space (OpenCV: +Z forward)."""
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing='ij')
    ys = ys.reshape(-1)
    xs = xs.reshape(-1)
    z = depth[ys, xs]

    valid = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        valid &= z < max_depth
    ys, xs, z = ys[valid], xs[valid], z[valid]

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    cols = rgb[ys, xs].astype(np.uint8)
    return pts, cols


def render_teaser(npz_path, video_path, output_path,
                  render_size=(720, 1280), fps=24, pc_stride=2,
                  point_size=4.0, max_depth=3.0,
                  view_back=0.35, view_up=0.18, view_right=0.10,
                  obj_models_dir=None, obj_pred_npz=None,
                  rescale_pcd_to_hands=False):
    """Render teaser 3D video from prediction NPZ + source video.

    Can be called from other scripts (e.g., infer_e2e.py) or via CLI.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── Load NPZ (supports both decoder predictions and raw HaWoR format) ──
    data = np.load(npz_path, allow_pickle=True)
    cam_traj = data['cam_traj']                  # (T, 7) = trans + quat
    left_trans = data['left_trans']
    left_root_orient = data['left_root_orient']
    left_hand_pose = data['left_hand_pose']
    left_betas = data['left_betas']
    left_valid = data['pred_valid_left'] if 'pred_valid_left' in data else data['left_valid']
    right_trans = data['right_trans']
    right_root_orient = data['right_root_orient']
    right_hand_pose = data['right_hand_pose']
    right_betas = data['right_betas']
    right_valid = data['pred_valid_right'] if 'pred_valid_right' in data else data['right_valid']

    # MANO canonical wrist joint position (flat_hand_mean=True, betas=0, zero pose/orient).
    # Used to correctly transform MANO transl params across coordinate frames:
    # MANO rotates verts around the wrist joint, not the origin, so the correct
    # transform is wrist-pivoted, not trans-pivoted.
    # (Same constants used in videox_fun/utils/hand_action_utils.py)
    _CANONICAL_WRIST = {
        'right': np.array([0.09566991, 0.00638343, 0.0061863], dtype=np.float64),
        'left':  np.array([-0.09566991, 0.00638343, 0.0061863], dtype=np.float64),
    }

    if 'pred_cam' in data:
        # Decoder prediction format — hands already in frame-0 cam space
        pred_cam = data['pred_cam']
        T = pred_cam.shape[0]
    else:
        # Raw HaWoR format — hands in world space, need to convert
        T = cam_traj.shape[0]
        cam_scale = float(data.get('cam_scale', 1.0))
        focal_ratio = float(data['cam_focal']) / (float(data['cam_center'][0]) * 2)

        # Build c2w matrices and w2c_0
        c2w_all = np.zeros((T, 4, 4), dtype=np.float64)
        c2w_all[:, 3, 3] = 1.0
        for t in range(T):
            c2w_all[t, :3, :3] = Rotation.from_quat(cam_traj[t, 3:7]).as_matrix()
            c2w_all[t, :3, 3] = cam_traj[t, :3] * cam_scale
        w2c_0 = np.linalg.inv(c2w_all[0])

        # Convert hands from world space to frame-0 cam space (wrist-pivoted transform)
        for side, trans_key, orient_key in [
            ('left', 'left_trans', 'left_root_orient'),
            ('right', 'right_trans', 'right_root_orient'),
        ]:
            trans_w = data[trans_key].copy()    # (T, 3) MANO transl param in world
            orient_w = data[orient_key].copy()  # (T, 3) axis-angle world space
            R_w2c = w2c_0[:3, :3]
            t_w2c = w2c_0[:3, 3]
            wrist_canon = _CANONICAL_WRIST[side]
            for t in range(T):
                # Wrist joint position in world space = MANO transl + canonical wrist
                wrist_world = trans_w[t] + wrist_canon
                # Transform the wrist JOINT (not transl) to cam-0 space
                wrist_cam = R_w2c @ wrist_world + t_w2c
                # MANO transl param in cam space = wrist_cam - canonical_wrist
                trans_w[t] = wrist_cam - wrist_canon
                # Transform orientation
                R_hand_w = Rotation.from_rotvec(orient_w[t]).as_matrix()
                R_hand_c = R_w2c @ R_hand_w
                orient_w[t] = Rotation.from_matrix(R_hand_c).as_rotvec()
            if side == 'left':
                left_trans, left_root_orient = trans_w, orient_w
            else:
                right_trans, right_root_orient = trans_w, orient_w

        # Build pred_cam from cam_traj (relative to frame 0)
        pred_cam = np.zeros((T, 10), dtype=np.float32)
        for t in range(T):
            rel = w2c_0 @ c2w_all[t]
            R = rel[:3, :3]
            pred_cam[t, :3] = R[:, 0]
            pred_cam[t, 3:6] = R[:, 1]
            pred_cam[t, 6:9] = rel[:3, 3]
            pred_cam[t, 9] = focal_ratio

    print(f"Loaded NPZ: {T} frames, "
          f"left_valid={int(left_valid.sum())}/{T}, right_valid={int(right_valid.sum())}/{T}")

    # ── Load video (frame 0 for MoGe) ──────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(frames) > 0, f"No frames read from {video_path}"
    video = np.stack(frames)
    H, W = video.shape[1:3]
    print(f"Video: {video.shape}")

    # ── Build camera c2w trajectory in FRAME-0 camera space ───────────────
    # The point cloud (MoGe) and hand meshes live in frame-0 camera space, so
    # the camera trail must match that frame. Two paths:
    #   • pred_cam available → use predicted cam directly (already frame-0-rel)
    #   • raw HaWoR only     → normalize raw cam_traj via w2c_0 (world → cam_0)
    # Only egodex's raw cam_traj happens to be already at identity at frame 0;
    # other datasets store cam_traj in an arbitrary world frame and need this
    # normalization, otherwise the camera trail floats away from the scene.
    cam_c2w = np.zeros((T, 4, 4), dtype=np.float32)
    cam_c2w[:, 3, 3] = 1.0
    if 'pred_cam' in data:
        # Decoder pred_cam[t] = [rot_6d(6), trans(3), focal(1)], frame-0 cam space.
        for t in range(T):
            r6d = pred_cam[t, :6]
            a1, a2 = r6d[:3], r6d[3:6]
            b1 = a1 / (np.linalg.norm(a1) + 1e-8)
            b2 = a2 - np.dot(b1, a2) * b1
            b2 = b2 / (np.linalg.norm(b2) + 1e-8)
            cam_c2w[t, :3, :3] = np.stack([b1, b2, np.cross(b1, b2)], axis=-1)
            cam_c2w[t, :3, 3] = pred_cam[t, 6:9]
    else:
        # Raw HaWoR: apply scale + normalize to frame-0 camera space.
        cam_scale = float(data.get('cam_scale', 1.0))
        c2w_world = np.zeros((T, 4, 4), dtype=np.float32)
        c2w_world[:, 3, 3] = 1.0
        for t in range(T):
            c2w_world[t, :3, :3] = Rotation.from_quat(cam_traj[t, 3:7]).as_matrix()
            c2w_world[t, :3, 3] = cam_traj[t, :3] * cam_scale
        w2c_0_norm = np.linalg.inv(c2w_world[0])
        for t in range(T):
            cam_c2w[t] = w2c_0_norm @ c2w_world[t]

    # Predicted focal from pred_cam focal_ratio (median across frames).
    focal_ratio = float(np.median(pred_cam[:, 9]))
    pred_focal_px = focal_ratio * W

    # ── MoGe-2 on frame 0 → colored point cloud ────────────────────────────
    # Feed the GT horizontal FOV (from the NPZ's intrinsics) to MoGe so its
    # depth is calibrated to our camera model. Without this, MoGe estimates
    # its own focal from pixels — for unusual FOVs (hot3d 96°, vitra 110°) or
    # square aspect (hot3d 352×352), MoGe's estimate is off and the resulting
    # point cloud is scaled inconsistently with our hand/cam coords.
    gt_fov_x_deg = None
    if 'cam_focal' in data and 'cam_center' in data:
        gt_focal_px = float(data['cam_focal'])
        gt_W = float(data['cam_center'][0]) * 2
        gt_fov_x_deg = float(np.degrees(2.0 * np.arctan(gt_W / (2.0 * gt_focal_px))))
        print(f"GT intrinsics: focal={gt_focal_px:.1f}px, W={gt_W:.0f}, "
              f"fov_x={gt_fov_x_deg:.1f}°")
    depth0, fx_moge, fy_moge, cx_moge, cy_moge = run_moge_frame0(
        video[0], fov_x_deg=gt_fov_x_deg)
    # Unproject with MoGe's returned intrinsics (which now match GT FOV if we
    # provided fov_x) — self-consistent with the depth calibration.
    pc_pts, pc_cols = unproject_points(
        video[0], depth0, fx_moge, fy_moge, cx_moge, cy_moge,
        stride=pc_stride, max_depth=max_depth)
    print(f"Point cloud: {pc_pts.shape[0]} points (unprojected with focal={fx_moge:.1f})")

    if rescale_pcd_to_hands:
        # MoGe-derived metric vs HaWoR/MegaSaM metric (where hand poses live)
        # often disagree. Calibrate the PCD to MegaSaM by projecting the right
        # wrist (cam0 frame, MegaSaM metric) to the depth image, looking up
        # MoGe's depth at that pixel, and rescaling pc_pts by the ratio.
        candidates = []
        for trans, label in (
                (right_trans[0] if right_valid[0] else None, 'right'),
                (left_trans[0] if left_valid[0] else None, 'left')):
            if trans is None: continue
            wz = float(trans[2])
            if wz <= 0: continue
            u = trans[0] / wz * fx_moge + cx_moge
            v = trans[1] / wz * fy_moge + cy_moge
            ui, vi = int(round(u)), int(round(v))
            Hd, Wd = depth0.shape[:2]
            if not (0 <= ui < Wd and 0 <= vi < Hd): continue
            zm = float(depth0[vi, ui])
            if zm <= 0: continue
            candidates.append((label, wz, zm, (ui, vi)))
        if candidates:
            label, wz, zm, (ui, vi) = candidates[0]
            r_pcd = wz / zm
            pc_pts = pc_pts * r_pcd
            print(f"Rescaled PCD to {label}-hand metric: ratio={r_pcd:.4f} "
                  f"(wrist z = {wz:.3f} m, MoGe depth at ({ui},{vi}) = {zm:.3f} m)")
        else:
            print("WARN: --rescale_pcd_to_hands requested but no usable hand wrist; PCD left in MoGe metric.")
    # FOVy for frustum: derived from predicted focal and video height.
    pred_fov_y = 2.0 * np.arctan(H / (2.0 * pred_focal_px))
    pred_aspect = W / H
    print(f"Predicted focal_ratio={focal_ratio:.3f} → focal={pred_focal_px:.1f}px, "
          f"fov_y={np.degrees(pred_fov_y):.1f}°")

    # ── MANO forward ───────────────────────────────────────────────────────
    import smplx
    mano_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'HaWoR', '_DATA', 'data', 'mano')
    mano_l = smplx.create(os.path.join(mano_dir, 'MANO_LEFT.pkl'),
                          'mano', is_rhand=False, use_pca=False, flat_hand_mean=True)
    mano_r = smplx.create(os.path.join(mano_dir, 'MANO_RIGHT.pkl'),
                          'mano', is_rhand=True, use_pca=False, flat_hand_mean=True)
    if torch.sum(torch.abs(mano_l.shapedirs[:, 0, :] - mano_r.shapedirs[:, 0, :])) < 1:
        mano_l.shapedirs[:, 0, :] *= -1

    with torch.no_grad():
        left_out = mano_l(
            global_orient=torch.tensor(left_root_orient, dtype=torch.float32),
            hand_pose=torch.tensor(left_hand_pose, dtype=torch.float32),
            betas=torch.tensor(left_betas, dtype=torch.float32).unsqueeze(0).expand(T, -1),
            transl=torch.tensor(left_trans, dtype=torch.float32),
        )
        right_out = mano_r(
            global_orient=torch.tensor(right_root_orient, dtype=torch.float32),
            hand_pose=torch.tensor(right_hand_pose, dtype=torch.float32),
            betas=torch.tensor(right_betas, dtype=torch.float32).unsqueeze(0).expand(T, -1),
            transl=torch.tensor(right_trans, dtype=torch.float32),
        )
    left_verts = left_out.vertices.detach().numpy()
    right_verts = right_out.vertices.detach().numpy()
    left_faces = mano_l.faces
    right_faces = mano_r.faces

    # ── Object mesh + trajectory (optional) ───────────────────────────────
    obj_mesh_template = None
    obj_traj = None  # (T, 4, 4) poses in frame-0 cam space
    obj_id_fallback = None
    obj_name_fallback = None
    obj_pred = None  # (T, 9) rot6d + trans, frame-0 cam space

    if 'obj_pred' in data:
        obj_pred = data['obj_pred']
        print(f"Using object trajectory from 'obj_pred': {obj_pred.shape}")
    elif 'obj_gt' in data:
        obj_pred = data['obj_gt']
        print(f"Using object trajectory from 'obj_gt': {obj_pred.shape}")
    elif 'obj_poses' in data and len(data['obj_poses']) > 0:
        # Dataset NPZ format: (N_obj, T, 9) world-space 6D-cols + trans.
        # Pick the first visible object and convert to frame-0 cam space.
        obj_poses_w = data['obj_poses']
        n_obj_total = obj_poses_w.shape[0]
        obj_vis = (data['obj_visibility'] if 'obj_visibility' in data
                    else np.ones((n_obj_total, obj_poses_w.shape[1])))
        oi = 0
        for i in range(n_obj_total):
            if obj_vis[i, 0] > 0.5:
                oi = i
                break
        # Recompute w2c_0 here so this path also works when pred_cam was provided.
        cam_scale_local = float(data.get('cam_scale', 1.0))
        c2w_0_local = np.eye(4, dtype=np.float64)
        c2w_0_local[:3, :3] = Rotation.from_quat(cam_traj[0, 3:7]).as_matrix()
        c2w_0_local[:3, 3] = cam_traj[0, :3] * cam_scale_local
        w2c_0_local = np.linalg.inv(c2w_0_local)
        T_obj = obj_poses_w.shape[1]
        obj_pred_arr = np.zeros((T_obj, 9), dtype=np.float32)
        for t in range(T_obj):
            # Dataset format: obj_poses 6D = first two COLUMNS of R (per dataset_egodex.py).
            r6d = obj_poses_w[oi, t, :6]
            a1, a2 = r6d[:3], r6d[3:6]
            b1 = a1 / (np.linalg.norm(a1) + 1e-8)
            b2 = a2 - np.dot(b1, a2) * b1
            b2 = b2 / (np.linalg.norm(b2) + 1e-8)
            R_w = np.stack([b1, b2, np.cross(b1, b2)], axis=-1)  # cols
            t_w = obj_poses_w[oi, t, 6:9]
            T_wo = np.eye(4); T_wo[:3, :3] = R_w; T_wo[:3, 3] = t_w
            T_c0o = w2c_0_local @ T_wo
            # Output ROWS convention to match the decoder/infer_e2e.py format.
            obj_pred_arr[t, :3] = T_c0o[0, :3]
            obj_pred_arr[t, 3:6] = T_c0o[1, :3]
            obj_pred_arr[t, 6:9] = T_c0o[:3, 3]
        obj_pred = obj_pred_arr
        if 'obj_ids' in data:
            obj_id_fallback = int(data['obj_ids'][oi])
        if 'obj_names' in data:
            obj_name_fallback = str(data['obj_names'][oi])
        print(f"Using object trajectory from 'obj_poses' (world→cam0, "
              f"oi={oi}/{n_obj_total}): {obj_pred.shape}")

    if obj_pred is not None:
        # 6D convention: ROWS (matches pytorch3d.matrix_to_rotation_6d, the decoder
        # output, and the world→cam0 fallback above). Reconstruct as rows of R.
        obj_traj = np.zeros((T, 4, 4), dtype=np.float32)
        obj_traj[:, 3, 3] = 1.0
        for t in range(T):
            r6d = obj_pred[t, :6]
            a1, a2 = r6d[:3], r6d[3:6]
            b1 = a1 / (np.linalg.norm(a1) + 1e-8)
            b2 = a2 - np.dot(b1, a2) * b1
            b2 = b2 / (np.linalg.norm(b2) + 1e-8)
            obj_traj[t, :3, :3] = np.stack([b1, b2, np.cross(b1, b2)], axis=0)
            obj_traj[t, :3, 3] = obj_pred[t, 6:9]

        # Load mesh if available
        obj_id = int(data['obj_id']) if 'obj_id' in data else obj_id_fallback
        obj_name_str = (str(data['obj_name']) if 'obj_name' in data
                        else obj_name_fallback)
        if obj_models_dir and (obj_id is not None or obj_name_str):
            candidates = []
            if obj_id is not None:
                candidates += [
                    (os.path.join(obj_models_dir, f"obj_{obj_id:06d}.glb"), 1.0),
                    (os.path.join(obj_models_dir, f"obj_{obj_id:06d}.ply"), 1.0),
                    (os.path.join(obj_models_dir, f"obj_{obj_id:06d}.obj"), 1.0),
                    (os.path.join(obj_models_dir, f"{obj_id:03d}_cm.obj"), 0.01),
                ]
            if obj_name_str:
                candidates.append(
                    (os.path.join(obj_models_dir, obj_name_str, "model.obj"), 1.0))
            for path, scale in candidates:
                if os.path.exists(path):
                    obj_mesh_template = trimesh.load(path, force='mesh')
                    obj_mesh_template.apply_scale(scale)
                    print(f"Loaded object mesh: {path} (scale={scale})")
                    break
        if obj_mesh_template is None and obj_traj is not None:
            # Fallback: small cube
            obj_mesh_template = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
            print("Using fallback cube for object")

    # ── Optional overlay object trajectory (e.g. predicted obj over GT scene) ──
    # Reads `obj_pred` (T, 9) from a second NPZ and renders it in a different
    # color alongside the primary obj_traj. Both are assumed to be in the same
    # frame-0 cam space (small anchor offset between pred/GT runs is acceptable
    # for visual comparison).
    obj_traj_overlay = None
    obj_mesh_template_overlay = None
    if obj_pred_npz is not None:
        ov_data = np.load(obj_pred_npz, allow_pickle=True)
        if 'obj_pred' in ov_data:
            ov_obj = ov_data['obj_pred']
            n_ov = min(T, ov_obj.shape[0])
            obj_traj_overlay = np.zeros((T, 4, 4), dtype=np.float32)
            obj_traj_overlay[:, 3, 3] = 1.0
            for t in range(n_ov):
                # ROWS convention (decoder output via pytorch3d.matrix_to_rotation_6d).
                r6d = ov_obj[t, :6]
                a1, a2 = r6d[:3], r6d[3:6]
                b1 = a1 / (np.linalg.norm(a1) + 1e-8)
                b2 = a2 - np.dot(b1, a2) * b1
                b2 = b2 / (np.linalg.norm(b2) + 1e-8)
                obj_traj_overlay[t, :3, :3] = np.stack([b1, b2, np.cross(b1, b2)], axis=0)
                obj_traj_overlay[t, :3, 3] = ov_obj[t, 6:9]
            print(f"Overlay object trajectory from {obj_pred_npz}: {obj_traj_overlay.shape}")

            # Load a separate mesh template for the overlay if its NPZ identifies
            # a different object (different obj_id/obj_name from the primary).
            ov_obj_id = int(ov_data['obj_id']) if 'obj_id' in ov_data else None
            ov_obj_name = str(ov_data['obj_name']) if 'obj_name' in ov_data else None
            if obj_models_dir and (ov_obj_id is not None or ov_obj_name):
                ov_candidates = []
                if ov_obj_id is not None:
                    ov_candidates += [
                        (os.path.join(obj_models_dir, f"obj_{ov_obj_id:06d}.glb"), 1.0),
                        (os.path.join(obj_models_dir, f"obj_{ov_obj_id:06d}.ply"), 1.0),
                        (os.path.join(obj_models_dir, f"obj_{ov_obj_id:06d}.obj"), 1.0),
                        (os.path.join(obj_models_dir, f"{ov_obj_id:03d}_cm.obj"), 0.01),
                    ]
                if ov_obj_name:
                    ov_candidates.append(
                        (os.path.join(obj_models_dir, ov_obj_name, "model.obj"), 1.0))
                for path, sc in ov_candidates:
                    if os.path.exists(path):
                        obj_mesh_template_overlay = trimesh.load(path, force='mesh')
                        obj_mesh_template_overlay.apply_scale(sc)
                        print(f"Loaded overlay object mesh: {path} (scale={sc})")
                        break
            if obj_mesh_template_overlay is None:
                print(f"Overlay mesh not found (obj_id={ov_obj_id}, name={ov_obj_name!r}); "
                      f"reusing primary mesh.")
        else:
            print(f"Warning: --obj_pred_npz {obj_pred_npz} has no 'obj_pred' key, skipping overlay")

    # ── View camera: pull back from frame-0 with a small offset ───────────
    # cam_c2w[0] ≈ identity, so frame-0 forward is +Z, up is -Y, right is +X.
    R0 = cam_c2w[0, :3, :3]
    t0 = cam_c2w[0, :3, 3]
    fwd0 = R0[:, 2]
    up0 = -R0[:, 1]
    right0 = R0[:, 0]
    view_pos = t0 - fwd0 * view_back + up0 * view_up + right0 * view_right

    # Look-at target: centroid of hands + scene center.
    targets = []
    if left_valid.any():
        targets.append(left_trans[left_valid].mean(axis=0))
    if right_valid.any():
        targets.append(right_trans[right_valid].mean(axis=0))
    # Include point cloud centroid so the scene stays in frame.
    if pc_pts.shape[0] > 0:
        targets.append(pc_pts.mean(axis=0))
    look_at = np.mean(targets, axis=0) if targets else (t0 + fwd0 * 0.5)

    forward = look_at - view_pos
    forward /= np.linalg.norm(forward) + 1e-8
    up = up0 - np.dot(up0, forward) * forward
    up /= np.linalg.norm(up) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)

    # pyrender camera pose (OpenGL: +X right, +Y up, -Z forward).
    def _make_view_c2w(pos, look_target, up_hint):
        fwd = look_target - pos
        fwd /= np.linalg.norm(fwd) + 1e-8
        u = up_hint - np.dot(up_hint, fwd) * fwd
        u /= np.linalg.norm(u) + 1e-8
        r = np.cross(fwd, u)
        r /= np.linalg.norm(r) + 1e-8
        u = np.cross(r, fwd)
        c2w = np.eye(4)
        c2w[:3, 0] = r
        c2w[:3, 1] = u
        c2w[:3, 2] = -fwd
        c2w[:3, 3] = pos
        return c2w

    view_c2w_front = _make_view_c2w(view_pos, look_at, up)

    # Second view: 90° right around the look_at point
    orbit_radius = np.linalg.norm(view_pos - look_at)

    # ── Static scene pieces (built once) ──────────────────────────────────
    pc_mesh = pyrender.Mesh.from_points(pc_pts, colors=pc_cols)

    # Camera trail: small spheres every 4 frames + path cylinders.
    trail_parts = []
    trail_pts = cam_c2w[:, :3, 3]
    for i in range(0, T, 4):
        s = trimesh.creation.uv_sphere(radius=0.004)
        s.apply_translation(trail_pts[i])
        s.visual.vertex_colors = [255, 220, 80, 255]
        trail_parts.append(s)
    for i in range(T - 1):
        seg = np.linalg.norm(trail_pts[i + 1] - trail_pts[i])
        if seg > 1e-6:
            c = trimesh.creation.cylinder(radius=0.0012, segment=[trail_pts[i], trail_pts[i + 1]])
            c.visual.vertex_colors = [255, 220, 80, 255]
            trail_parts.append(c)
    trail_mesh = trimesh.util.concatenate(trail_parts) if trail_parts else None

    # ── Render helper ─────────────────────────────────────────────────────
    rH, rW = render_size
    renderer = pyrender.OffscreenRenderer(rW, rH)
    renderer._renderer.point_size = point_size

    # Side view position
    horiz_right = right0.copy()
    horiz_right[1] = 0; horiz_right /= np.linalg.norm(horiz_right) + 1e-8
    side_pos = look_at + horiz_right * orbit_radius
    side_pos[1] = view_pos[1]
    view_c2w_side = _make_view_c2w(side_pos, look_at, up)

    def _render_pass(view_c2w, writer, label=""):
        for frame in range(T):
            scene = pyrender.Scene(bg_color=[20, 20, 22, 255],
                                   ambient_light=[0.55, 0.55, 0.55])
            scene.add(pc_mesh)
            if trail_mesh is not None:
                scene.add(pyrender.Mesh.from_trimesh(trail_mesh))

            corners = make_frustum_corners(cam_c2w[frame], scale=0.06,
                                           aspect=pred_aspect, fov_y=pred_fov_y)
            frust = frustum_wireframe(corners, radius=0.0016)
            frust.visual.vertex_colors = [255, 100, 100, 255]
            scene.add(pyrender.Mesh.from_trimesh(frust))

            if left_valid[frame]:
                lm = trimesh.Trimesh(vertices=left_verts[frame], faces=left_faces, process=False)
                lm.visual.vertex_colors = np.full((len(left_verts[frame]), 4),
                                                  [90, 130, 230, 255], dtype=np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(lm))
            if right_valid[frame]:
                rm = trimesh.Trimesh(vertices=right_verts[frame], faces=right_faces, process=False)
                rm.visual.vertex_colors = np.full((len(right_verts[frame]), 4),
                                                  [230, 110, 110, 255], dtype=np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(rm))

            # Object mesh (primary — green)
            if obj_mesh_template is not None and obj_traj is not None:
                om = obj_mesh_template.copy()
                om.visual.vertex_colors = np.full((len(om.vertices), 4),
                                                  [80, 220, 80, 200], dtype=np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(om), pose=obj_traj[frame])

            # Overlay object mesh (a *second* object, or pred-over-GT — orange).
            # Uses its own mesh if --obj_pred_npz pointed at a different obj_id;
            # otherwise reuses the primary mesh template.
            ov_template = (obj_mesh_template_overlay
                           if obj_mesh_template_overlay is not None
                           else obj_mesh_template)
            if ov_template is not None and obj_traj_overlay is not None:
                om2 = ov_template.copy()
                om2.visual.vertex_colors = np.full((len(om2.vertices), 4),
                                                   [255, 165, 0, 200], dtype=np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(om2), pose=obj_traj_overlay[frame])

            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            scene.add(light, pose=view_c2w)
            cam_node = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=rW / rH)
            scene.add(cam_node, pose=view_c2w)

            color_img, _ = renderer.render(scene)
            writer.write(cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

            if frame % 20 == 0:
                print(f"  {label} Frame {frame}/{T}")

    # ── Write video: front view, then side view ───────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (rW, rH))

    _render_pass(view_c2w_front, writer, label="Front")
    _render_pass(view_c2w_side, writer, label="Side")

    writer.release()
    renderer.delete()
    print(f"Saved: {output_path} ({2*T} frames: front + side)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--render_size", type=int, nargs=2, default=[720, 1280])
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--pc_stride", type=int, default=2)
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--max_depth", type=float, default=3.0)
    parser.add_argument("--view_back", type=float, default=0.35)
    parser.add_argument("--view_up", type=float, default=0.18)
    parser.add_argument("--view_right", type=float, default=0.10)
    parser.add_argument("--obj_models_dir", type=str, default=None,
                        help="Dir with object meshes (TACO: *_cm.obj, HOT3D: obj_*.glb)")
    parser.add_argument("--obj_pred_npz", type=str, default=None,
                        help="Second NPZ to read 'obj_pred' from for overlay (orange) "
                             "alongside the primary green object. Useful for visualizing "
                             "predicted vs GT object trajectory in the same scene/camera.")
    parser.add_argument("--rescale_pcd_to_hands", action="store_true",
                        help="Rescale the MoGe-derived point cloud to HaWoR/MegaSaM "
                             "metric using the right wrist as a reference. Fixes "
                             "the situation where hands+obj live in MegaSaM metric "
                             "but the PCD lives in MoGe metric.")
    args = parser.parse_args()
    render_teaser(args.npz, args.video, args.output,
                  render_size=tuple(args.render_size), fps=args.fps,
                  pc_stride=args.pc_stride, point_size=args.point_size,
                  max_depth=args.max_depth, view_back=args.view_back,
                  view_up=args.view_up, view_right=args.view_right,
                  obj_models_dir=args.obj_models_dir,
                  obj_pred_npz=args.obj_pred_npz,
                  rescale_pcd_to_hands=args.rescale_pcd_to_hands)


if __name__ == "__main__":
    main()
