import argparse
import json
import os
from typing import Dict, Tuple, Optional

import trimesh
import cv2
import numpy as np

from paradex.calibration.utils import load_camparam
from paradex.image.grid import make_image_grid
from paradex.image.overlay import overlay_mask
import torch
import pytorch_kinematics as pk
try:
    import nvdiffrast.torch as dr
except Exception:
    dr = None
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map
from paradex.utils.path import rsc_path, shared_dir
from paradex.visualization.robot import RobotModule


def inspire_state_to_qpos_sil(state: np.ndarray) -> np.ndarray:
    """Map Inspire 6-dim state (0~1000) to 12-dim hand qpos in radians."""
    action = state.astype(np.float64).reshape(1, 6)
    qpos = np.zeros((1, 12), dtype=np.float64)

    # thumb_1_joint
    mask = action[:, 5] <= 100
    qpos[:, 0] = (
        7e-9 * action[:, 5] ** 3
        - 1e-5 * action[:, 5] ** 2
        - 0.073 * action[:, 5]
        + 75.866
    ) * np.pi / 180.0
    qpos[mask, 0] = 68.5 * np.pi / 180.0

    # thumb_2_joint
    qpos[:, 1] = (
        2e-8 * action[:, 4] ** 3
        - 5e-5 * action[:, 4] ** 2
        - 0.005 * action[:, 4]
        + 31.407
    ) * np.pi / 180.0

    # index_1_joint
    qpos[:, 4] = (
        -4e-8 * action[:, 3] ** 3
        + 3e-5 * action[:, 3] ** 2
        - 0.0704 * action[:, 3]
        + 83.572
    ) * np.pi / 180.0
    # middle_1_joint
    qpos[:, 6] = (
        -4e-8 * action[:, 2] ** 3
        + 3e-5 * action[:, 2] ** 2
        - 0.0704 * action[:, 2]
        + 83.572
    ) * np.pi / 180.0
    # ring_1_joint
    qpos[:, 8] = (
        -4e-8 * action[:, 1] ** 3
        + 3e-5 * action[:, 1] ** 2
        - 0.0704 * action[:, 1]
        + 83.572
    ) * np.pi / 180.0
    # little_1_joint
    qpos[:, 10] = (
        -4e-8 * action[:, 0] ** 3
        + 3e-5 * action[:, 0] ** 2
        - 0.0704 * action[:, 0]
        + 83.572
    ) * np.pi / 180.0

    # thumb_3_joint
    qpos[:, 2] = (
        3e-11 * action[:, 4] ** 4
        - 4e-8 * action[:, 4] ** 3
        + 9e-6 * action[:, 4] ** 2
        - 0.025 * action[:, 4]
        + 28.197
    ) * np.pi / 180.0
    # thumb_4_joint
    qpos[:, 3] = (
        8e-9 * action[:, 4] ** 3
        - 5e-6 * action[:, 4] ** 2
        - 0.0267 * action[:, 4]
        + 24.189
    ) * np.pi / 180.0

    # index_2_joint
    qpos[:, 5] = 1.57 * (1.0 - action[:, 3] / 1000.0)
    # middle_2_joint
    qpos[:, 7] = 1.57 * (1.0 - action[:, 2] / 1000.0)
    # ring_2_joint
    qpos[:, 9] = 1.57 * (1.0 - action[:, 1] / 1000.0)
    # little_2_joint
    qpos[:, 11] = 1.57 * (1.0 - action[:, 0] / 1000.0)

    return qpos[0]

def intr_opencv_to_opengl_proj(K, width, height, near=0.2, far=2.0):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = 1 - 2 * cx / width
    proj[1, 2] = 2 * cy / height - 1
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1
    return proj

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    if t_mtx.dim() == 2:
        t_mtx = t_mtx[None, ...]
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1], device=pos.device)], axis=1)
    return torch.einsum("nj,bij->bni", posw, t_mtx).contiguous()

def mesh_to_obj_dict(mesh, device="cuda"):
    if isinstance(mesh, trimesh.Trimesh):
        verts = mesh.vertices
        faces = mesh.faces
        if mesh.visual.kind == "vertex" and hasattr(mesh.visual, "vertex_colors"):
            vtx_col = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            vtx_col = np.zeros_like(verts)
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")
    verts = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)
    faces = torch.tensor(faces, dtype=torch.int32, device=device)
    vtx_col = torch.tensor(vtx_col, dtype=torch.float32, device=device)
    if vtx_col.max() > 1.0:
        vtx_col = vtx_col / 255.0
    return {"verts": verts, "faces": faces, "vtx_col": vtx_col, "col_idx": faces.clone()}

def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / (torch.norm(axis) + 1e-8)
    x, y, z = axis[0], axis[1], axis[2]
    zero = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    sa = torch.sin(angle)
    ca = torch.cos(angle)
    R = I + sa * K + (1.0 - ca) * (K @ K)
    return R

def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, device=R.device, dtype=R.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

class TorchBatchRenderer:
    def __init__(self, intrinsics, extrinsics, near=0.01, far=2.0):
        if dr is None:
            raise RuntimeError("nvdiffrast is not available")
        self.glctx = dr.RasterizeCudaContext()
        serial_list = list(intrinsics.keys())
        serial_list.sort()
        self.serial_list = serial_list
        cam_intrinsics = [intrinsics[serial]["intrinsics_undistort"] for serial in serial_list]
        cam_extrinsics = [extrinsics[serial] for serial in serial_list]
        width = intrinsics[serial_list[0]]["width"]
        height = intrinsics[serial_list[0]]["height"]
        self.width, self.height = width, height
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cam_extrs = []
        for cam_extrinsic in cam_extrinsics:
            org_extr = np.eye(4)
            org_extr[:3, :] = cam_extrinsic
            cam_extrs.append(torch.tensor(org_extr, device=self.device).float())
        self.cam_extrs_t = torch.stack(cam_extrs)
        self.intr_opengl = torch.stack(
            [
                torch.tensor(
                    intr_opencv_to_opengl_proj(cam_intrinsic, width, height, near=near, far=far)
                ).to(self.device)
                for cam_intrinsic in cam_intrinsics
            ]
        )
        self.flip_z = torch.tensor(np.diag([1, -1, -1, 1]).astype(np.float32)).to(self.device)
        self.near = near
        self.far = far

    def render_multi(self, mesh_list):
        obj_dicts = mesh_list
        mtx = self.intr_opengl @ self.flip_z @ self.cam_extrs_t

        stacked_pos = []
        stacked_pos_idx = []
        stacked_col = []
        accumulated_vertex_numb = 0
        for obj_dict in obj_dicts:
            verts = obj_dict["verts"]
            pos = verts[0] if verts.dim() == 3 else verts
            stacked_pos.append(pos)
            pos_idx = obj_dict["faces"].clone()
            pos_idx += accumulated_vertex_numb
            stacked_pos_idx.append(pos_idx)
            vtx_col = obj_dict.get("vtx_col")
            if vtx_col is None:
                vtx_col = torch.zeros_like(pos)
            elif vtx_col.dim() == 3:
                vtx_col = vtx_col[0]
            stacked_col.append(vtx_col)
            accumulated_vertex_numb += pos.shape[0]

        stacked_pos = torch.vstack(stacked_pos)
        stacked_pos_idx = torch.vstack(stacked_pos_idx)
        stacked_col = torch.vstack(stacked_col)

        pos_clip = transform_pos(mtx, stacked_pos)
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, stacked_pos_idx, resolution=(self.height, self.width))
        color, _ = dr.interpolate(stacked_col, rast_out, stacked_pos_idx)
        color = dr.antialias(color, rast_out, pos_clip, stacked_pos_idx)

        ones = torch.ones_like(stacked_pos[:, :1], device=stacked_pos.device)[None]
        mask_soft, _ = dr.interpolate(ones, rast_out, stacked_pos_idx)
        mask_soft = dr.antialias(mask_soft, rast_out, pos_clip, stacked_pos_idx)

        color = torch.flip(color, dims=[1])
        mask_soft = torch.flip(mask_soft, dims=[1])

        color_dict = {s: color[i] for i, s in enumerate(self.serial_list)}
        mask_dict = {s: mask_soft[i] for i, s in enumerate(self.serial_list)}
        return color_dict, mask_dict


def load_target_masks(ep_dir: str, cam_ids, undistort_maps, render_sizes) -> Dict[str, np.ndarray]:
    masks = {}
    mask_dir = os.path.join(ep_dir, "raw", "masks")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")
    for cam_id in cam_ids:
        mask_path = os.path.join(mask_dir, f"{cam_id}.png")
        if not os.path.exists(mask_path):
            continue
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            continue
        if mask_raw.ndim == 3 and mask_raw.shape[2] == 4:
            mask = mask_raw[:, :, 3]
        elif mask_raw.ndim == 3:
            mask = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
        else:
            mask = mask_raw
        mapx, mapy = undistort_maps[cam_id]
        mask = apply_undistort_map(mask, mapx, mapy)
        render_w, render_h = render_sizes[cam_id]
        if mask.shape[1] != render_w or mask.shape[0] != render_h:
            mask = cv2.resize(mask, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
        mask = (mask.astype(np.float32) / 255.0)
        masks[cam_id] = mask
    return masks


def load_images(ep_dir: str, cam_ids, undistort_maps, render_sizes) -> Dict[str, np.ndarray]:
    images = {}
    img_dir = os.path.join(ep_dir, "raw", "images")
    for cam_id in cam_ids:
        img_path = os.path.join(img_dir, f"{cam_id}.png")
        if not os.path.exists(img_path):
            continue
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        mapx, mapy = undistort_maps[cam_id]
        image_bgr = apply_undistort_map(image_bgr, mapx, mapy)
        render_w, render_h = render_sizes[cam_id]
        if image_bgr.shape[1] != render_w or image_bgr.shape[0] != render_h:
            image_bgr = cv2.resize(image_bgr, (render_w, render_h), interpolation=cv2.INTER_AREA)
        images[cam_id] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return images


def load_link_mesh_from_urdf(urdf, link_name: str, urdf_dir: str) -> trimesh.Trimesh:
    link = urdf.link_map[link_name]
    meshes = []
    for visual in link.visuals:
        geom = visual.geometry
        mesh = None
        if hasattr(geom, "mesh") and geom.mesh is not None:
            filename = geom.mesh.filename
            mesh_path = os.path.join(urdf_dir, filename)
            mesh = trimesh.load(mesh_path, force="mesh")
            if geom.mesh.scale is not None:
                mesh.apply_scale(geom.mesh.scale)
        elif hasattr(geom, "box") and geom.box is not None:
            size = geom.box.size
            mesh = trimesh.creation.box(extents=size)
        elif hasattr(geom, "cylinder") and geom.cylinder is not None:
            mesh = trimesh.creation.cylinder(
                radius=geom.cylinder.radius, height=geom.cylinder.length
            )
        elif hasattr(geom, "sphere") and geom.sphere is not None:
            mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
        if mesh is None:
            continue
        if visual.origin is not None:
            mesh.apply_transform(visual.origin)
        meshes.append(mesh)
    if not meshes:
        raise RuntimeError(f"No visuals found for link: {link_name}")
    return trimesh.util.concatenate(meshes)

class TorchUrdfKinematics:
    def __init__(self, urdf_path: str, device: str = "cuda"):
        import yourdfpy

        self.device = device
        self.urdf = yourdfpy.URDF.load(urdf_path, build_scene_graph=False)
        self.urdf_dir = os.path.dirname(urdf_path)
        with open(urdf_path, "r") as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(dtype=torch.float32, device=device)
        self.joint_names = self.chain.get_joint_parameter_names()

        self.link_meshes = {}
        for link_name in self.urdf.link_map.keys():
            try:
                mesh = load_link_mesh_from_urdf(self.urdf, link_name, self.urdf_dir)
            except Exception:
                continue
            verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
            self.link_meshes[link_name] = (verts, faces)

    def forward(self, qpos: torch.Tensor) -> list:
        if qpos.dim() == 1:
            qpos = qpos[None, :]
        fk = self.chain.forward_kinematics(qpos)
        meshes = []
        for link_name, (verts, faces) in self.link_meshes.items():
            tf = fk.get(link_name, None)
            if tf is None:
                continue
            T = tf.get_matrix()[0]
            v = torch.cat([verts, torch.ones((verts.shape[0], 1), device=self.device)], dim=1)
            v_w = (v @ T.T)[:, :3]
            meshes.append({"verts": v_w, "faces": faces, "name": link_name})
        return meshes


def filter_hand_meshes(meshes):
    hand_prefixes = (
        "left_thumb_",
        "left_index_",
        "left_middle_",
        "left_ring_",
        "left_little_",
    )
    filtered = []
    for m in meshes:
        name = m.get("name", "")
        if name == "base_link" or name.startswith(hand_prefixes):
            filtered.append(m)
    return filtered


def compute_loss_torch(mask_pred: Dict[str, torch.Tensor], mask_tgt: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for cam_id, tgt_mask in mask_tgt.items():
        pred = mask_pred.get(cam_id)
        if pred is None:
            continue
        pred_mask = pred[..., 0] if pred.ndim == 3 else pred
        if pred_mask.shape != tgt_mask.shape:
            raise ValueError(f"Mask shape mismatch for {cam_id}: {pred_mask.shape} vs {tgt_mask.shape}")
        losses.append(torch.mean((pred_mask - tgt_mask) ** 2))
    if not losses:
        return torch.tensor(float("inf"), device=next(iter(mask_tgt.values())).device)
    return torch.mean(torch.stack(losses))


def mask_stats(mask_pred: Dict[str, torch.Tensor], cam_ids) -> Dict[str, Tuple[float, float, float]]:
    stats = {}
    for cam_id in cam_ids:
        pred = mask_pred.get(cam_id)
        if pred is None:
            continue
        pred_mask = pred[..., 0] if pred.ndim == 3 else pred
        stats[cam_id] = (
            float(pred_mask.min().item()),
            float(pred_mask.max().item()),
            float(pred_mask.sum().item()),
        )
    return stats


def save_debug_grid(
    out_dir: str,
    images: Dict[str, np.ndarray],
    mask_pred: Dict[str, np.ndarray],
    tag: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    overlays = []
    for cam_id in sorted(images.keys()):
        pred_mask = mask_pred[cam_id][..., 0] if mask_pred[cam_id].ndim == 3 else mask_pred[cam_id]
        overlay = overlay_mask(images[cam_id], pred_mask, color=(0, 255, 0), alpha=0.5)
        out_path = os.path.join(out_dir, f"{tag}_{cam_id}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlays.append(overlay)
    if overlays:
        grid = make_image_grid(overlays)
        grid_path = os.path.join(out_dir, f"{tag}_grid.png")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def save_debug_compare_grid(
    out_dir: str,
    mask_pred: Dict[str, np.ndarray],
    mask_gt: Dict[str, np.ndarray],
    tag: str,
    images: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    overlays = []
    for cam_id in sorted(mask_gt.keys()):
        pred_mask = mask_pred[cam_id][..., 0] if mask_pred[cam_id].ndim == 3 else mask_pred[cam_id]
        gt_mask = mask_gt[cam_id]
        if images is not None and cam_id in images:
            canvas = images[cam_id]
        else:
            h, w = gt_mask.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        overlay = overlay_mask(canvas, gt_mask, color=(255, 0, 0), alpha=0.6)
        overlay = overlay_mask(overlay, pred_mask, color=(0, 255, 0), alpha=0.4)
        out_path = os.path.join(out_dir, f"{tag}_{cam_id}.png")
        # cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlays.append(overlay)
    if overlays:
        grid = make_image_grid(overlays)
        grid_path = os.path.join(out_dir, f"{tag}_grid.png")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=os.path.join(shared_dir, "inspire_pinky_calibration"))
    parser.add_argument("--ep", type=str, default="0")
    parser.add_argument("--gd-steps", type=int, default=200)
    parser.add_argument("--gd-lr-float", type=float, default=5e-3)
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--debug-every", type=int, default=100)
    parser.add_argument("--debug-dir", type=str, default="raw/debug")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--urdf-path", type=str, default=os.path.join(rsc_path, "robot", "xarm_inspire_left_new_floating.urdf"))
    parser.add_argument(
        "--cam-ids",
        nargs="*",
        default=[
            "22641005",
            "22641023",
            "22645021",
            "22684210",
            "22684737",
            "22684755",
            "23022632",
            "23180202",
            "23280594",
        ],
        help="Camera IDs to use for optimization.",
    )
    args = parser.parse_args()

    ep_dir = os.path.join(args.base_dir, args.ep)

    intrinsics, extrinsics = load_camparam(ep_dir)
    cam_ids = [cid for cid in args.cam_ids if cid in intrinsics]
    if not cam_ids:
        cam_ids = sorted(intrinsics.keys())
    c2r = np.load(os.path.join(ep_dir, "C2R.npy"))

    hand_state = np.load(os.path.join(ep_dir, "raw", "hand", "state.npy"))
    arm_state = np.load(os.path.join(ep_dir, "raw", "arm", "state.npy"))

    hand_qpos = np.zeros(18, dtype=float)
    full_qpos = np.concatenate([arm_state.reshape(1, -1), hand_qpos.reshape(1, -1)], axis=1)[0]

    urdf_path = args.urdf_path
    robot = RobotModule(urdf_path)

    robot_dof = robot.get_num_joints()
    if full_qpos.shape[0] != robot_dof:
        raise ValueError(f"qpos length mismatch: {full_qpos.shape[0]} vs {robot_dof}")

    # Camera transforms and undistort maps.
    render_extrinsics = {}
    undistort_maps = {}
    render_sizes = {}
    for cam_id, intr in intrinsics.items():
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsics[cam_id]
        cam_from_robot = cam_from_world @ c2r
        render_extrinsics[cam_id] = cam_from_robot[:3, :]
        _, mapx, mapy = precomute_undistort_map(intr)
        undistort_maps[cam_id] = (mapx, mapy)
        scale = float(args.render_scale)
        if scale != 1.0:
            intr["intrinsics_undistort"][:2, :] *= scale
            intr["width"] = max(1, int(round(intr["width"] * scale)))
            intr["height"] = max(1, int(round(intr["height"] * scale)))
        render_sizes[cam_id] = (intr["width"], intr["height"])

    intrinsics_used = {cid: intrinsics[cid] for cid in cam_ids if cid in intrinsics}
    render_extrinsics_used = {cid: render_extrinsics[cid] for cid in cam_ids if cid in render_extrinsics}
    renderer = TorchBatchRenderer(intrinsics_used, render_extrinsics_used)
    target_masks = load_target_masks(ep_dir, cam_ids, undistort_maps, render_sizes)
    if not target_masks:
        raise RuntimeError("No masks found; check raw/masks/*.png")
    available_cam_ids = sorted(target_masks.keys())
    images = load_images(ep_dir, available_cam_ids, undistort_maps, render_sizes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_masks_torch = {
        cid: torch.tensor(target_masks[cid], dtype=torch.float32, device=device)
        for cid in available_cam_ids
    }
    urdf_kin = TorchUrdfKinematics(urdf_path, device=device)

    joint_names = robot.get_joint_names()
    base_qpos = full_qpos.copy()

    print(f"Using {len(available_cam_ids)} view(s): {available_cam_ids}")
    print("Optimizing arm_to_hand floating joints only.")

    # Build qpos tensor in chain order.
    joint_to_val = {name: base_qpos[i] for i, name in enumerate(joint_names)}
    chain_joint_names = urdf_kin.joint_names
    qpos_base = torch.tensor(
        [joint_to_val.get(n, 0.0) for n in chain_joint_names],
        dtype=torch.float32,
        device=device,
    )
    # Build correction transforms so pk FK aligns with RobotModule FK at base_qpos.
    # robot.update_cfg(base_qpos)
    # robot_tf = {}
    # for link_name in urdf_kin.link_meshes.keys():
    #     try:
    #         T_np = robot.get_transform(link_name, robot.urdf.base_link, collision_geometry=False)
    #     except Exception:
    #         continue
    #     robot_tf[link_name] = torch.tensor(T_np, dtype=torch.float32, device=device)
    # fk_base = urdf_kin.chain.forward_kinematics(qpos_base)
    # corr_dict = {}
    # for link_name, T_robot in robot_tf.items():
    #     tf_pk = fk_base.get(link_name, None)
    #     if tf_pk is None:
    #         continue
    #     T_pk = tf_pk.get_matrix()[0]
    #     try:
    #         T_pk_inv = torch.linalg.inv(T_pk)
    #     except RuntimeError:
    #         continue
    #     corr_dict[link_name] = T_robot @ T_pk_inv
    # urdf_kin.set_corrections(corr_dict)

    float_joints = [
        "arm_to_hand_x",
        "arm_to_hand_y",
        "arm_to_hand_z",
        "arm_to_hand_roll",
        "arm_to_hand_pitch",
        "arm_to_hand_yaw",
    ]
    idx_float = [chain_joint_names.index(jn) for jn in float_joints if jn in chain_joint_names]
    init_float = qpos_base[idx_float].clone()

    # Debug before optimization.
    with torch.no_grad():
        qpos_dbg = qpos_base.clone()
        before_meshes = urdf_kin.forward(qpos_dbg)
        _, before_mask = renderer.render_multi(filter_hand_meshes(before_meshes))
        before_mask_np = {k: v.detach().cpu().numpy() for k, v in before_mask.items()}
        init_loss = compute_loss_torch(before_mask, target_masks_torch).item()
        print(f"Initial mask stats: {mask_stats(before_mask, available_cam_ids)}")
    print(f"Initial loss: {init_loss:.6f}")
    debug_root = os.path.join(ep_dir, args.debug_dir)
    if images:
        save_debug_compare_grid(debug_root, before_mask_np, target_masks, "before", images=images)

    # Optimize arm_to_hand floating joints (6-DOF).
    theta_float = init_float.clone().requires_grad_(True)
    opt_float = torch.optim.Adam([theta_float], lr=args.gd_lr_float)
    best_theta = theta_float.detach().clone()
    best_loss = float("inf")

    for i in range(1, args.gd_steps + 1):
        opt_float.zero_grad()
        qpos_step = qpos_base.clone()
        qpos_step[idx_float] = theta_float
        meshes = urdf_kin.forward(qpos_step)
        _, mask_pred = renderer.render_multi(filter_hand_meshes(meshes))
        loss = compute_loss_torch(mask_pred, target_masks_torch)
        loss.backward()
        grad_norm = theta_float.grad.norm().item() if theta_float.grad is not None else 0.0
        opt_float.step()

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_theta = theta_float.detach().clone()
        if i == 1 or i == args.gd_steps or i % max(1, args.gd_steps // 5) == 0:
            print(f"float gd {i}/{args.gd_steps}: loss={loss.item():.6f}, grad_norm={grad_norm:.6e}")
        if images and args.debug_every > 0 and i % args.debug_every == 0:
            with torch.no_grad():
                dbg_meshes = urdf_kin.forward(qpos_step.detach())
                _, dbg_mask = renderer.render_multi(filter_hand_meshes(dbg_meshes))
                dbg_mask_np = {k: v.detach().cpu().numpy() for k, v in dbg_mask.items()}
            save_debug_compare_grid(debug_root, dbg_mask_np, target_masks, f"float_step{i:04d}", images=images)

    # Debug after optimization.
    with torch.no_grad():
        qpos_after = qpos_base.clone()
        qpos_after[idx_float] = best_theta
        after_meshes = urdf_kin.forward(qpos_after)
        _, after_mask = renderer.render_multi(filter_hand_meshes(after_meshes))
        after_mask_np = {k: v.detach().cpu().numpy() for k, v in after_mask.items()}
    if images:
        save_debug_compare_grid(debug_root, after_mask_np, target_masks, "after", images=images)

    out_json = args.out_json or os.path.join(ep_dir, "lookup_arm_to_hand.json")
    state_key = os.path.basename(ep_dir)
    data = {
        state_key: {
            "arm_to_hand": [float(x) for x in best_theta.tolist()],
            "loss": best_loss,
            "init_arm_to_hand": [float(x) for x in init_float.tolist()],
        }
    }
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved lookupto {out_json}")


if __name__ == "__main__":
    main()
 
