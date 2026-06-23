"""
Silhouette-based single-frame robot (arm-only) state optimizer.

Two modes (select with positional subcommand):

  gen-masks   Generate per-camera binary robot masks for the chosen frame using SAM3.
              Run inside the `sam3` conda env.
              Output: <capture_root>/robot_mask/frame_<fid>/<cam_id>.png

              python -m src.util.robot.optimize_state_via_silhouette gen-masks \
                  --object apple --ep 0 [--frame-id 42] --text-prompt "robot arm"

  optimize    Optimize chosen-frame arm qpos using nvdiffrast differentiable rendering
              and pytorch_kinematics FK. Run inside the `paradex` conda env.
              Output: <capture_root>/silhouette_refine/frame_<fid>/qpos.{npy,json}
                      before.png / after.png  (silhouette-on-image grids)
                      debug/before_grid.png, stepXXXX_grid.png, after_grid.png

              python -m src.util.robot.optimize_state_via_silhouette optimize \
                  --object apple --ep 0 [--frame-id 42]

`--frame-id` selects a master-timeline frame id from raw/timestamps/frame_id.npy.
If omitted, the very first frame is used. Pass the same --frame-id to both modes.

URDF: defaults based on --hand (e.g. inspire_f1 -> xarm_inspire_f1_right.urdf).
Override with --urdf-path. Only the arm joints (joint1..joint6) are optimized;
hand joints are held fixed at their loaded values.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from paradex.utils.path import rsc_path
from paradex.utils.load_data import load_series, resample_to
from paradex.calibration.utils import load_camparam
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map

logging.getLogger("yourdfpy.urdf").setLevel(logging.ERROR)

MASK_ROOT = "robot_mask"
REFINED_ROOT = "silhouette_refine"


# ---------------------------------------------------------------------------
# Shared data-loading (mirrors visualize_all.py for the subset we need)
# ---------------------------------------------------------------------------


def resolve_capture_root(args: argparse.Namespace) -> str:
    hand_dir = getattr(args, "hand_dir", None) or args.hand
    if args.capture_root is None:
        return os.path.join(
            "/home/temp_id/shared_data/capture/eccv2026",
            hand_dir,
            args.object,
            str(args.ep),
        )
    return os.path.join(args.capture_root, args.object, str(args.ep))


def resolve_urdf_path(args: argparse.Namespace) -> str:
    if getattr(args, "urdf_path", None):
        return args.urdf_path
    if args.hand == "inspire_f1":
        return os.path.join(rsc_path, "robot", "xarm_inspire_f1_right.urdf")
    if args.hand == "inspire":
        return os.path.join(rsc_path, "robot", "xarm_inspire_DFTP.urdf")
    if args.hand == "allegro":
        return os.path.join(rsc_path, "robot", "xarm_allegro.urdf")
    if args.hand == "kistar":
        return os.path.join(rsc_path, "robot", "xarm_kistar.urdf")
    return os.path.join(rsc_path, "robot", "xarm.urdf")


def _resolve_frame_idx(video_frame_ids: np.ndarray, requested) -> int:
    if requested is None or requested < 0:
        return 0
    matches = np.where(video_frame_ids == int(requested))[0]
    if matches.size == 0:
        raise ValueError(
            f"--frame-id {requested} not found in frame_id.npy. "
            f"Available range: [{video_frame_ids.min()}, {video_frame_ids.max()}]."
        )
    return int(matches[0])


def load_frame_id(args: argparse.Namespace) -> Dict:
    """Lightweight: resolve capture_root + frame_id without loading hand state.

    Used by gen-masks which runs in the sam3 env (where paradex.robot hand
    utilities — which pull in pinocchio — are not available).
    """
    capture_root = resolve_capture_root(args)
    data_root = os.path.join(capture_root, "raw")
    arm_dir = os.path.join(data_root, "arm")

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    arm_time = arm_time + args.arm_time_offset

    ts_path = os.path.join(data_root, "timestamps", "timestamp.npy")
    fid_path = os.path.join(data_root, "timestamps", "frame_id.npy")
    if os.path.exists(ts_path) and os.path.exists(fid_path):
        video_frame_ids = np.load(fid_path).astype(int)
    else:
        video_frame_ids = np.arange(1, arm_qpos.shape[0] + 1, dtype=int)

    idx = _resolve_frame_idx(video_frame_ids, getattr(args, "frame_id", None))
    return {
        "capture_root": capture_root,
        "data_root": data_root,
        "frame_id": int(video_frame_ids[idx]),
    }


def load_frame_state(args: argparse.Namespace) -> Dict:
    """Return dict with per-frame full (arm+hand) qpos, frame_id, capture_root, urdf_path.

    Matches the layout used by src/util/robot/visualize/visualize_all.py:
    full_qpos = concat([arm_qpos, hand_qpos], axis=1), aligned on the master
    video timeline. Only arm joints are optimized — hand joints are held fixed.
    Frame selection: if `--frame-id` is given, that specific master-timeline frame is
    used. Otherwise the first frame is used.
    """
    from paradex.robot.inspire import (
        inspire_action_to_qpos,
        inspire_f1_action_to_qpos_dof6,
    )

    capture_root = resolve_capture_root(args)
    data_root = os.path.join(capture_root, "raw")
    arm_dir = os.path.join(data_root, "arm")
    hand_dir = os.path.join(data_root, "hand")

    arm_qpos, arm_time_raw = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    arm_time = arm_time_raw + args.arm_time_offset

    hand_time_has_file = False
    if args.hand == "inspire":
        hand_action, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))
        hand_time_has_file = os.path.exists(os.path.join(hand_dir, "time.npy"))
    elif args.hand in ("inspire_f1", "_inspire_f1"):
        hand_action, hand_time = load_series(hand_dir, ("right_joint_states.npy",))
        hand_time_file = os.path.join(hand_dir, "right_joint_states_time.npy")
        if os.path.exists(hand_time_file):
            hand_time = np.load(hand_time_file, allow_pickle=True).astype(float)
            hand_time_has_file = True
        else:
            hand_time_has_file = os.path.exists(os.path.join(hand_dir, "time.npy"))
    elif args.hand == "allegro":
        hand_action, hand_time = load_series(hand_dir, ("position.npy",))
        hand_time_has_file = os.path.exists(os.path.join(hand_dir, "time.npy"))
    else:
        raise ValueError(f"Invalid hand name: {args.hand}")

    # hand is intentionally NOT shifted by arm_time_offset: align it to the raw
    # arm timeline (or its own timestamps) so the offset only affects the arm.
    if not hand_time_has_file:
        if len(arm_time_raw) > 1:
            hand_time = np.linspace(
                arm_time_raw[0], arm_time_raw[-1], hand_action.shape[0], dtype=float
            )
        else:
            hand_time = np.arange(hand_action.shape[0], dtype=float)

    if args.hand == "inspire":
        hand_qpos_src = inspire_action_to_qpos(hand_action)
    elif args.hand in ("inspire_f1", "_inspire_f1"):
        hand_qpos_src = inspire_f1_action_to_qpos_dof6(hand_action)
    else:
        hand_qpos_src = hand_action

    ts_path = os.path.join(data_root, "timestamps", "timestamp.npy")
    fid_path = os.path.join(data_root, "timestamps", "frame_id.npy")
    if os.path.exists(ts_path) and os.path.exists(fid_path):
        video_times = np.load(ts_path)
        video_frame_ids = np.load(fid_path).astype(int)
        arm_video = resample_to(arm_time, arm_qpos, video_times)
        hand_video = resample_to(hand_time, hand_qpos_src, video_times)
    else:
        video_times = arm_time
        video_frame_ids = np.arange(1, arm_qpos.shape[0] + 1, dtype=int)
        arm_video = arm_qpos
        hand_video = resample_to(hand_time, hand_qpos_src, arm_time)
    qpos_video = np.concatenate([arm_video, hand_video], axis=1)

    idx = _resolve_frame_idx(video_frame_ids, getattr(args, "frame_id", None))

    urdf_path = resolve_urdf_path(args)
    return {
        "capture_root": capture_root,
        "data_root": data_root,
        "urdf_path": urdf_path,
        "qpos": qpos_video[idx].astype(np.float64),
        "frame_id": int(video_frame_ids[idx]),
        "video_times": video_times,
        # Full-series fields for multi-frame ("all frames") runs:
        "qpos_series": qpos_video.astype(np.float64),
        "video_frame_ids": video_frame_ids,
        "arm_dof": int(arm_qpos.shape[1]),
    }


def first_frame_image_path(capture_root: str, cam_id: str, frame_id: int) -> str:
    return os.path.join(capture_root, "video_extracted", cam_id, f"{frame_id:05d}.jpg")


def _resolve_frame_id_list(
    args: argparse.Namespace,
    capture_root: str,
    state: Dict,
) -> List[int]:
    """Which frame ids should be processed.

    - --all-frames: every id in raw/timestamps/frame_id.npy (or the arm series fallback).
    - --frame-id <n>: just that one.
    - otherwise: the first frame (state['frame_id']).
    """
    if getattr(args, "all_frames", False):
        data_root = state.get("data_root", os.path.join(capture_root, "raw"))
        fid_path = os.path.join(data_root, "timestamps", "frame_id.npy")
        if os.path.exists(fid_path):
            return [int(f) for f in np.load(fid_path).astype(int).tolist()]
        if "video_frame_ids" in state:
            return [int(f) for f in state["video_frame_ids"].tolist()]
        return [state["frame_id"]]
    return [state["frame_id"]]


def select_cam_ids(args: argparse.Namespace, available: List[str]) -> List[str]:
    if args.cam_ids:
        return [c for c in args.cam_ids if c in available]
    return sorted(available)


# ---------------------------------------------------------------------------
# Mask generation (SAM3)
# ---------------------------------------------------------------------------


def cmd_gen_masks(args: argparse.Namespace) -> None:
    # Coarse skip: if mask root exists and not forced, bail before loading SAM3.
    capture_root_early = resolve_capture_root(args)
    mask_root = os.path.join(capture_root_early, MASK_ROOT)
    if os.path.isdir(mask_root) and not getattr(args, "force_regen_masks", False):
        print(f"[gen-masks] mask root exists, skipping generation entirely: {mask_root}")
        return

    # Lazy imports so this file can still be imported in the paradex env.
    import sam3  # noqa: F401
    import torch
    from PIL import Image

    from sam3 import build_sam3_image_model
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )
    from sam3.eval.postprocessors import PostProcessImage

    state = load_frame_id(args)
    capture_root = state["capture_root"]

    intrinsics, _ = load_camparam(capture_root)
    cam_ids = select_cam_ids(args, list(intrinsics.keys()))
    if not cam_ids:
        raise RuntimeError("No camera ids available from cam_param/.")

    frame_ids = _resolve_frame_id_list(args, capture_root, state)
    print(f"[gen-masks] processing {len(frame_ids)} frame(s)")

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    autocast_ctx.__enter__()
    infer_ctx = torch.inference_mode()
    infer_ctx.__enter__()

    model = build_sam3_image_model(bpe_path=bpe_path)
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=args.detection_threshold,
        to_cpu=True,
    )

    # Precompute undistort maps per cam (invariant across frames).
    undistort_maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cam_id in cam_ids:
        _, mapx, mapy = precomute_undistort_map(intrinsics[cam_id])
        undistort_maps[cam_id] = (mapx, mapy)

    total_saved = 0
    total_expected = 0
    for frame_id in frame_ids:
        out_dir = os.path.join(capture_root, MASK_ROOT, f"frame_{frame_id:06d}")
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        for cam_id in cam_ids:
            img_path = first_frame_image_path(capture_root, cam_id, frame_id)
            if not os.path.exists(img_path):
                print(f"[WARN] missing image for {cam_id} frame {frame_id}: {img_path}")
                continue

            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] failed to read {img_path}")
                continue
            mapx, mapy = undistort_maps[cam_id]
            bgr = apply_undistort_map(bgr, mapx, mapy)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            datapoint = Datapoint(find_queries=[], images=[])
            w, h = pil.size
            datapoint.images = [SAMImage(data=pil, objects=[], size=[h, w])]
            datapoint.find_queries.append(
                FindQueryLoaded(
                    query_text=args.text_prompt,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=1,
                        original_image_id=1,
                        original_category_id=1,
                        original_size=[w, h],
                        object_id=0,
                        frame_index=0,
                    ),
                )
            )
            datapoint = transform(datapoint)
            batch = collate([datapoint], dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            output = model(batch)
            results = postprocessor.process_results(output, batch.find_metadatas)

            det_masks = []
            if results:
                first_key = next(iter(results.keys()))
                det_masks = results[first_key].get("masks", [])
            if len(det_masks) == 0:
                print(f"[WARN] SAM3 no detections for {cam_id} frame {frame_id}")
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                def _to_bool_2d(m):
                    t = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
                    t = np.asarray(t).astype(bool)
                    while t.ndim > 2:
                        t = t[0]
                    return t
                stacked = np.stack([_to_bool_2d(m) for m in det_masks], axis=0)
                union = np.any(stacked, axis=0)
                mask = (union.astype(np.uint8)) * 255

            out_path = os.path.join(out_dir, f"{cam_id}.png")
            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = 255
            rgba[..., 3] = mask
            Image.fromarray(rgba).save(out_path)
            saved += 1

        total_saved += saved
        total_expected += len(cam_ids)
        print(f"[gen-masks] frame {frame_id}: saved {saved}/{len(cam_ids)} -> {out_dir}")

    infer_ctx.__exit__(None, None, None)
    autocast_ctx.__exit__(None, None, None)
    print(f"[gen-masks] DONE: total saved {total_saved}/{total_expected}")


# ---------------------------------------------------------------------------
# Differentiable rendering + optimization
# ---------------------------------------------------------------------------


def _intr_opencv_to_opengl_proj(K, width, height, near=0.01, far=2.0):
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


def _load_link_mesh_from_urdf(urdf, link_name: str, urdf_dir: str):
    import trimesh

    link = urdf.link_map[link_name]
    meshes = []
    for visual in link.visuals:
        geom = visual.geometry
        mesh = None
        if hasattr(geom, "mesh") and geom.mesh is not None:
            mesh_path = os.path.join(urdf_dir, geom.mesh.filename)
            mesh = trimesh.load(mesh_path, force="mesh")
            if geom.mesh.scale is not None:
                mesh.apply_scale(geom.mesh.scale)
        elif hasattr(geom, "box") and geom.box is not None:
            mesh = trimesh.creation.box(extents=geom.box.size)
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
        return None
    return trimesh.util.concatenate(meshes)


class _TorchUrdfKinematics:
    def __init__(self, urdf_path: str, device: str = "cuda"):
        import pytorch_kinematics as pk
        import torch
        import yourdfpy

        self.device = device
        self.urdf = yourdfpy.URDF.load(urdf_path, build_scene_graph=False)
        self.urdf_dir = os.path.dirname(urdf_path)
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(
                dtype=torch.float32, device=device
            )
        self.joint_names = self.chain.get_joint_parameter_names()

        self.link_meshes: Dict[str, Tuple] = {}
        for link_name in self.urdf.link_map.keys():
            mesh = _load_link_mesh_from_urdf(self.urdf, link_name, self.urdf_dir)
            if mesh is None:
                continue
            verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
            self.link_meshes[link_name] = (verts, faces)

    def forward(self, qpos):
        import torch

        if qpos.dim() == 1:
            qpos = qpos[None, :]
        fk = self.chain.forward_kinematics(qpos)
        out = []
        for link_name, (verts, faces) in self.link_meshes.items():
            tf = fk.get(link_name, None)
            if tf is None:
                continue
            T = tf.get_matrix()[0]
            ones = torch.ones((verts.shape[0], 1), device=self.device)
            v_w = (torch.cat([verts, ones], dim=1) @ T.T)[:, :3]
            out.append({"verts": v_w, "faces": faces, "name": link_name})
        return out


class _TorchBatchRenderer:
    def __init__(self, intrinsics, extrinsics, near=0.01, far=2.0):
        import torch
        import nvdiffrast.torch as dr

        self.glctx = dr.RasterizeCudaContext()
        serial_list = sorted(intrinsics.keys())
        self.serial_list = serial_list
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cam_intrinsics = [intrinsics[s]["intrinsics_undistort"] for s in serial_list]
        cam_extrinsics = [extrinsics[s] for s in serial_list]
        width = intrinsics[serial_list[0]]["width"]
        height = intrinsics[serial_list[0]]["height"]
        self.width, self.height = width, height

        cam_extrs = []
        for ex in cam_extrinsics:
            T = np.eye(4)
            T[:3, :] = ex
            cam_extrs.append(torch.tensor(T, device=self.device).float())
        self.cam_extrs_t = torch.stack(cam_extrs)
        self.intr_opengl = torch.stack(
            [
                torch.tensor(
                    _intr_opencv_to_opengl_proj(K, width, height, near=near, far=far)
                ).to(self.device)
                for K in cam_intrinsics
            ]
        )
        self.flip_z = torch.tensor(
            np.diag([1, -1, -1, 1]).astype(np.float32)
        ).to(self.device)

    def render_masks(self, mesh_list):
        import torch
        import nvdiffrast.torch as dr

        mtx = self.intr_opengl @ self.flip_z @ self.cam_extrs_t

        stacked_pos = []
        stacked_idx = []
        acc = 0
        for m in mesh_list:
            verts = m["verts"]
            pos = verts[0] if verts.dim() == 3 else verts
            stacked_pos.append(pos)
            idx = m["faces"].clone() + acc
            stacked_idx.append(idx)
            acc += pos.shape[0]
        pos = torch.vstack(stacked_pos)
        idx = torch.vstack(stacked_idx)

        posw = torch.cat(
            [pos, torch.ones([pos.shape[0], 1], device=pos.device)], dim=1
        )
        pos_clip = torch.einsum("nj,bij->bni", posw, mtx).contiguous()
        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, idx, resolution=(self.height, self.width)
        )
        ones = torch.ones_like(pos[:, :1])[None]
        mask_soft, _ = dr.interpolate(ones, rast_out, idx)
        mask_soft = dr.antialias(mask_soft, rast_out, pos_clip, idx)
        mask_soft = torch.flip(mask_soft, dims=[1])
        return {s: mask_soft[i, ..., 0] for i, s in enumerate(self.serial_list)}


def _load_target_masks(
    mask_dir: str,
    cam_ids: List[str],
    undistort_maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
    render_sizes: Dict[str, Tuple[int, int]],
    masks_are_undistorted: bool,
) -> Dict[str, np.ndarray]:
    masks = {}
    for cam_id in cam_ids:
        mask_path = os.path.join(mask_dir, f"{cam_id}.png")
        if not os.path.exists(mask_path):
            continue
        raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        if raw.ndim == 3 and raw.shape[2] == 4:
            mask = raw[:, :, 3]
        elif raw.ndim == 3:
            mask = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            mask = raw
        if not masks_are_undistorted:
            mapx, mapy = undistort_maps[cam_id]
            mask = apply_undistort_map(mask, mapx, mapy)
        w, h = render_sizes[cam_id]
        if mask.shape[1] != w or mask.shape[0] != h:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masks[cam_id] = (mask.astype(np.float32) / 255.0)
    return masks


def _load_view_images(
    capture_root: str,
    frame_id: int,
    cam_ids: List[str],
    undistort_maps,
    render_sizes,
) -> Dict[str, np.ndarray]:
    images = {}
    for cam_id in cam_ids:
        p = first_frame_image_path(capture_root, cam_id, frame_id)
        if not os.path.exists(p):
            continue
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        mapx, mapy = undistort_maps[cam_id]
        bgr = apply_undistort_map(bgr, mapx, mapy)
        w, h = render_sizes[cam_id]
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        images[cam_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return images


def _save_compare_grid(
    out_dir: str,
    tag: str,
    images: Dict[str, np.ndarray],
    mask_pred: Dict[str, np.ndarray],
    mask_gt: Dict[str, np.ndarray],
) -> None:
    """Debug grid: GT mask (red) + predicted mask (green) on original image."""
    from paradex.image.overlay import overlay_mask
    from paradex.image.grid import make_image_grid

    os.makedirs(out_dir, exist_ok=True)
    overlays = []
    for cam_id in sorted(mask_pred.keys()):
        pred = mask_pred[cam_id]
        canvas = images.get(cam_id)
        if canvas is None:
            h, w = pred.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        gt = mask_gt.get(cam_id)
        if gt is not None:
            canvas = overlay_mask(canvas, gt, color=(255, 0, 0), alpha=0.6)
        ov = overlay_mask(canvas, pred, color=(0, 255, 0), alpha=0.4)
        overlays.append(ov)
    if overlays:
        grid = make_image_grid(overlays)
        cv2.imwrite(
            os.path.join(out_dir, f"{tag}_grid.png"),
            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
        )


def _save_silhouette_grid(
    out_dir: str,
    tag: str,
    images: Dict[str, np.ndarray],
    mask_pred: Dict[str, np.ndarray],
    silhouette_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> None:
    """Single-layer grid: rendered silhouette projected onto original image."""
    from paradex.image.overlay import overlay_mask
    from paradex.image.grid import make_image_grid

    os.makedirs(out_dir, exist_ok=True)
    overlays = []
    for cam_id in sorted(mask_pred.keys()):
        pred = mask_pred[cam_id]
        canvas = images.get(cam_id)
        if canvas is None:
            h, w = pred.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        ov = overlay_mask(canvas, pred, color=silhouette_color, alpha=alpha)
        overlays.append(ov)
    if not overlays:
        return
    grid = make_image_grid(overlays)
    out_path = os.path.join(out_dir, f"{tag}.png")
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[silhouette] saved {tag} -> {out_path}")


def _build_cam_context(
    args: argparse.Namespace,
    capture_root: str,
    intrinsics,
    extrinsics,
) -> Dict:
    """Build per-cam rendering structures that do not depend on frame_id."""
    c2r = np.load(os.path.join(capture_root, "C2R.npy"))

    cam_ids = select_cam_ids(args, list(intrinsics.keys()))
    if not cam_ids:
        raise RuntimeError("No cameras found with intrinsics.")

    render_extrinsics: Dict[str, np.ndarray] = {}
    undistort_maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    render_sizes: Dict[str, Tuple[int, int]] = {}
    intrinsics_used: Dict[str, Dict] = {}
    scale = float(args.render_scale)
    for cam_id in cam_ids:
        intr = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in intrinsics[cam_id].items()}
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsics[cam_id]
        cam_from_robot = cam_from_world @ c2r
        render_extrinsics[cam_id] = cam_from_robot[:3, :]
        _, mapx, mapy = precomute_undistort_map(intr)
        undistort_maps[cam_id] = (mapx, mapy)
        if scale != 1.0:
            intr["intrinsics_undistort"] = intr["intrinsics_undistort"].copy()
            intr["intrinsics_undistort"][:2, :] *= scale
            intr["width"] = max(1, int(round(intr["width"] * scale)))
            intr["height"] = max(1, int(round(intr["height"] * scale)))
        render_sizes[cam_id] = (intr["width"], intr["height"])
        intrinsics_used[cam_id] = intr

    return {
        "cam_ids": cam_ids,
        "render_extrinsics": render_extrinsics,
        "undistort_maps": undistort_maps,
        "render_sizes": render_sizes,
        "intrinsics_used": intrinsics_used,
    }


def _optimize_single_frame(
    args: argparse.Namespace,
    capture_root: str,
    urdf_path: str,
    frame_id: int,
    base_qpos_np: np.ndarray,
    cam_ctx: Dict,
    renderer,
    urdf_kin,
    actuated_names: List[str],
    chain_names: List[str],
    opt_idx: List[int],
    device: str,
    save_debug: bool,
    prev_theta: Optional["torch.Tensor"] = None,
):
    """Optimize arm qpos for a single frame. Returns refined actuated qpos (np)."""
    import torch

    cam_ids = cam_ctx["cam_ids"]
    undistort_maps = cam_ctx["undistort_maps"]
    render_sizes = cam_ctx["render_sizes"]

    mask_dir = args.mask_dir or os.path.join(capture_root, MASK_ROOT, f"frame_{frame_id:06d}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"Mask directory not found: {mask_dir}. Run `gen-masks` first."
        )
    target_masks = _load_target_masks(
        mask_dir, cam_ids, undistort_maps, render_sizes, masks_are_undistorted=True
    )
    loss_cam_ids = [c for c in cam_ids if c in target_masks and target_masks[c].sum() > 0]
    if not loss_cam_ids:
        print(f"[optimize] frame {frame_id}: no non-empty masks, skipping optimization; "
              "returning base qpos unchanged.")
        return base_qpos_np.copy(), None, None, prev_theta

    print(f"[optimize] frame {frame_id}: loss cams {len(loss_cam_ids)}/{len(cam_ids)}")
    images = _load_view_images(capture_root, frame_id, cam_ids, undistort_maps, render_sizes)

    target_masks_t = {
        c: torch.tensor(target_masks[c], dtype=torch.float32, device=device)
        for c in loss_cam_ids
    }

    name_to_val = {n: float(v) for n, v in zip(actuated_names, base_qpos_np)}
    qpos_base = torch.tensor(
        [name_to_val.get(n, 0.0) for n in chain_names],
        dtype=torch.float32,
        device=device,
    )

    temporal_w = float(getattr(args, "temporal_weight", 0.0) or 0.0)
    use_prev = prev_theta is not None and temporal_w > 0.0
    if use_prev and getattr(args, "warm_start", True):
        theta_init = prev_theta.detach().to(device)
    else:
        theta_init = qpos_base[opt_idx].detach().clone()
    theta = theta_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=args.lr)

    prev_theta_t = prev_theta.detach().to(device) if use_prev else None

    def forward_loss(qpos_full, theta_cur=None):
        meshes = urdf_kin.forward(qpos_full)
        pred = renderer.render_masks(meshes)
        parts = []
        for c, tgt in target_masks_t.items():
            p = pred.get(c)
            if p is None:
                continue
            parts.append(torch.mean((p - tgt) ** 2))
        sil_loss = torch.mean(torch.stack(parts))
        if use_prev and theta_cur is not None:
            temp_loss = temporal_w * torch.mean((theta_cur - prev_theta_t) ** 2)
            return sil_loss + temp_loss, pred, sil_loss.detach(), temp_loss.detach()
        zero = torch.zeros((), device=device)
        return sil_loss, pred, sil_loss.detach(), zero

    out_root = os.path.join(capture_root, REFINED_ROOT, f"frame_{frame_id:06d}")
    debug_dir = os.path.join(out_root, "debug")
    os.makedirs(out_root, exist_ok=True)

    with torch.no_grad():
        init_loss, pred_init, init_sil, _ = forward_loss(qpos_base, theta_cur=theta_init)
    pred_init_np = {c: v.detach().cpu().numpy() for c, v in pred_init.items()}
    _save_silhouette_grid(out_root, "before", images, pred_init_np)
    if save_debug:
        _save_compare_grid(debug_dir, "before", images, pred_init_np, target_masks)

    best_loss = float("inf")
    best_sil = float("inf")
    best_theta = theta.detach().clone()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        qpos_full = qpos_base.clone()
        qpos_full[opt_idx] = theta
        loss, _, sil_val, temp_val = forward_loss(qpos_full, theta_cur=theta)
        loss.backward()
        optimizer.step()

        # Track best by silhouette loss only so the temporal term doesn't bias selection.
        if sil_val.item() < best_sil:
            best_sil = float(sil_val.item())
            best_loss = float(loss.item())
            best_theta = theta.detach().clone()

        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            gnorm = theta.grad.norm().item() if theta.grad is not None else 0.0
            print(f"  step {step}/{args.steps}: loss={loss.item():.6f} "
                  f"sil={sil_val.item():.6f} temp={temp_val.item():.6f} "
                  f"grad_norm={gnorm:.4e}")
        if save_debug and args.debug_every > 0 and step % args.debug_every == 0:
            with torch.no_grad():
                qpos_full = qpos_base.clone()
                qpos_full[opt_idx] = theta
                _, pred_dbg, _, _ = forward_loss(qpos_full, theta_cur=theta)
            _save_compare_grid(
                debug_dir,
                f"step{step:04d}",
                images,
                {c: v.detach().cpu().numpy() for c, v in pred_dbg.items()},
                target_masks,
            )

    with torch.no_grad():
        qpos_final = qpos_base.clone()
        qpos_final[opt_idx] = best_theta
        _, pred_after, _, _ = forward_loss(qpos_final, theta_cur=best_theta)
    pred_after_np = {c: v.detach().cpu().numpy() for c, v in pred_after.items()}
    print(f"[optimize] frame {frame_id}: init_sil={init_sil.item():.6f} "
          f"best_sil={best_sil:.6f} (best_total={best_loss:.6f})")
    _save_silhouette_grid(out_root, "after", images, pred_after_np)
    if save_debug:
        _save_compare_grid(debug_dir, "after", images, pred_after_np, target_masks)

    # Mirror after.png into a flat folder for quick flipping across frames.
    after_src = os.path.join(out_root, "after.png")
    if os.path.exists(after_src):
        flat_dir = os.path.join(capture_root, REFINED_ROOT, "after_all")
        os.makedirs(flat_dir, exist_ok=True)
        import shutil as _shutil
        _shutil.copyfile(after_src, os.path.join(flat_dir, f"frame_{frame_id:06d}.png"))

    chain_vals = qpos_final.detach().cpu().numpy().tolist()
    chain_map = {n: v for n, v in zip(chain_names, chain_vals)}
    refined_actuated = np.array(
        [chain_map.get(n, name_to_val[n]) for n in actuated_names], dtype=np.float64
    )
    np.save(os.path.join(out_root, "qpos.npy"), refined_actuated)
    with open(os.path.join(out_root, "qpos.json"), "w") as f:
        json.dump(
            {
                "frame_id": int(frame_id),
                "init_loss": float(init_loss.item()),
                "init_sil": float(init_sil.item()),
                "best_sil": float(best_sil),
                "best_loss": float(best_loss),
                "temporal_weight": float(temporal_w),
                "used_prev_theta": bool(use_prev),
                "cam_ids": cam_ids,
                "actuated_joint_names": actuated_names,
                "qpos_init": [float(x) for x in base_qpos_np.tolist()],
                "qpos_refined": [float(x) for x in refined_actuated.tolist()],
                "optimized_chain_joints": [chain_names[i] for i in opt_idx],
            },
            f,
            indent=2,
        )
    return refined_actuated, float(init_loss.item()), float(best_loss), best_theta.detach()


def cmd_optimize(args: argparse.Namespace) -> None:
    import torch
    import yourdfpy

    state = load_frame_state(args)
    capture_root = state["capture_root"]
    urdf_path = state["urdf_path"]
    arm_dof = int(state["arm_dof"])

    intrinsics, extrinsics = load_camparam(capture_root)
    cam_ctx = _build_cam_context(args, capture_root, intrinsics, extrinsics)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = _TorchBatchRenderer(
        {c: cam_ctx["intrinsics_used"][c] for c in cam_ctx["cam_ids"]},
        {c: cam_ctx["render_extrinsics"][c] for c in cam_ctx["cam_ids"]},
    )
    urdf_kin = _TorchUrdfKinematics(urdf_path, device=device)
    yurdf = yourdfpy.URDF.load(urdf_path, build_scene_graph=False)
    actuated_names = yurdf.actuated_joint_names
    chain_names = urdf_kin.joint_names
    opt_idx = _resolve_optimize_indices(chain_names, args)

    # Resolve frame list + per-frame qpos.
    frame_ids = _resolve_frame_id_list(args, capture_root, state)
    if args.all_frames:
        video_frame_ids = state["video_frame_ids"]
        qpos_series = state["qpos_series"]
        fid_to_idx = {int(f): i for i, f in enumerate(video_frame_ids.tolist())}
    else:
        fid_to_idx = {int(state["frame_id"]): 0}
        qpos_series = state["qpos"][None, :]

    if len(qpos_series.shape) == 1 or qpos_series.shape[1] != len(actuated_names):
        pass  # passthrough; per-frame builds chain anyway
    print(f"[optimize] frames to process: {len(frame_ids)}")

    save_debug = (not args.all_frames) or args.force_debug
    refined_all: List[np.ndarray] = []
    processed_fids: List[int] = []
    prev_theta = None
    for fid in frame_ids:
        if fid not in fid_to_idx:
            print(f"[optimize] frame {fid}: no base qpos available, skipping.")
            continue
        base_qpos = qpos_series[fid_to_idx[fid]].astype(np.float64)
        refined, _init, _best, prev_theta = _optimize_single_frame(
            args, capture_root, urdf_path, int(fid), base_qpos,
            cam_ctx, renderer, urdf_kin, actuated_names, chain_names, opt_idx,
            device, save_debug=save_debug, prev_theta=prev_theta,
        )
        refined_all.append(refined)
        processed_fids.append(int(fid))

    if not refined_all:
        print("[optimize] no frames produced results.")
        return

    if args.all_frames:
        arr = np.stack(refined_all, axis=0)  # [N, D_actuated]
        arm_arr = arr[:, :arm_dof]
        proc_dir = os.path.join(capture_root, "processed", "arm")
        os.makedirs(proc_dir, exist_ok=True)
        fid_arr = np.asarray(processed_fids, dtype=int)
        np.save(os.path.join(proc_dir, "position.npy"), arm_arr)
        np.save(os.path.join(proc_dir, "frame_id.npy"), fid_arr)
        vt = state.get("video_times")
        if vt is not None:
            t_arr = np.asarray(
                [vt[fid_to_idx[int(f)]] for f in processed_fids], dtype=float
            )
            np.save(os.path.join(proc_dir, "time.npy"), t_arr)
        print(f"[optimize] aggregate arm qpos saved -> {proc_dir}/position.npy "
              f"shape={arm_arr.shape}")


ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def _resolve_optimize_indices(chain_names: List[str], args: argparse.Namespace) -> List[int]:
    """Select which pk-chain joints to optimize."""
    if args.optimize == "arm":
        idx = [i for i, n in enumerate(chain_names) if n in ARM_JOINT_NAMES]
        if not idx:
            raise ValueError(
                f"No arm joints {ARM_JOINT_NAMES} found in chain: {chain_names}"
            )
        return idx
    if args.optimize == "all":
        return list(range(len(chain_names)))
    if args.optimize == "joints":
        names = set(args.joints or [])
        idx = [i for i, n in enumerate(chain_names) if n in names]
        if not idx:
            raise ValueError(
                f"--optimize joints given but no match among chain joints: {args.joints}"
            )
        return idx
    raise ValueError(f"Unknown --optimize value: {args.optimize}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--arm", type=str, default="xarm")
    p.add_argument(
        "--hand",
        type=str,
        default="inspire_f1",
        help="Hand type. Controls qpos conversion + default URDF selection.",
    )
    p.add_argument(
        "--hand-dir",
        type=str,
        default=None,
        help="Override the hand-folder name under the capture root. "
        "Defaults to --hand. Useful when the capture was logged under a "
        "different folder name than the hand type (e.g. hand=inspire_f1 "
        "but folder=inspire_f1_right).",
    )
    p.add_argument("--object", type=str, required=True)
    p.add_argument("--ep", type=int, required=True)
    p.add_argument("--capture-root", default=None)
    p.add_argument("--arm_time_offset", type=float, default=0.28)
    p.add_argument(
        "--frame-id",
        type=int,
        default=None,
        help="Master-timeline frame id (as stored in raw/timestamps/frame_id.npy). "
        "Default: first frame.",
    )
    p.add_argument(
        "--all-frames",
        action="store_true",
        help="Run over every frame in frame_id.npy. Also aggregates optimized arm "
        "qpos into <capture_root>/processed/arm/{position,time,frame_id}.npy.",
    )
    p.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="URDF to use for FK. Default: combined arm+hand URDF selected via --hand "
        "(e.g. inspire_f1 -> xarm_inspire_f1_right.urdf).",
    )
    p.add_argument(
        "--cam-ids",
        nargs="*",
        default=None,
        help="Camera IDs to use. Default: all cameras with both cam_param and mask.",
    )


def _add_gen_mask_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--text-prompt", type=str, default="robot arm")
    p.add_argument("--detection-threshold", type=float, default=0.3)
    p.add_argument(
        "--force-regen-masks",
        action="store_true",
        help="Regenerate masks even if PNGs already exist for the frame.",
    )


def _add_optimize_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help=f"Default: <capture_root>/{MASK_ROOT}/frame_<frame_id:06d>",
    )
    p.add_argument("--render-scale", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--debug-every", type=int, default=50)
    p.add_argument(
        "--force-debug",
        action="store_true",
        help="Keep per-frame debug grids even with --all-frames.",
    )
    p.add_argument(
        "--temporal-weight",
        type=float,
        default=0.0,
        help="Weight λ for temporal smoothness term λ‖θ_t − θ_{t-1}‖² (optimized joints only).",
    )
    p.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        help="Disable warm-start (θ_t init from θ_{t-1}) when temporal-weight>0.",
    )
    p.set_defaults(warm_start=True)
    p.add_argument(
        "--optimize",
        type=str,
        default="arm",
        choices=["arm", "all", "joints"],
        help="Which joints to optimize. 'arm' (default): joint1..joint6 only. "
        "'all': every actuated joint. 'joints': pass names via --joints.",
    )
    p.add_argument(
        "--joints",
        nargs="*",
        default=None,
        help="With --optimize joints: explicit pk-chain joint names to optimize.",
    )


def _build_gen_masks_subproc_args(args: argparse.Namespace) -> List[str]:
    """Rebuild CLI for gen-masks subprocess from the `run` namespace."""
    out: List[str] = [
        "gen-masks",
        "--arm", args.arm,
        "--hand", args.hand,
        "--object", args.object,
        "--ep", str(args.ep),
        "--arm_time_offset", str(args.arm_time_offset),
        "--text-prompt", args.text_prompt,
        "--detection-threshold", str(args.detection_threshold),
    ]
    if args.capture_root is not None:
        out += ["--capture-root", args.capture_root]
    if getattr(args, "hand_dir", None):
        out += ["--hand-dir", args.hand_dir]
    if args.frame_id is not None:
        out += ["--frame-id", str(args.frame_id)]
    if getattr(args, "all_frames", False):
        out += ["--all-frames"]
    if args.urdf_path is not None:
        out += ["--urdf-path", args.urdf_path]
    if args.cam_ids:
        out += ["--cam-ids", *args.cam_ids]
    if getattr(args, "force_regen_masks", False):
        out += ["--force-regen-masks"]
    return out


def cmd_run(args: argparse.Namespace) -> None:
    """Generate masks (SAM3 in sam3 env) then optimize (nvdiffrast in current env).

    The gen-masks subprocess skips per-frame dirs that already have a full set
    of mask PNGs unless --force-regen-masks is passed.
    """
    capture_root_early = resolve_capture_root(args)
    mask_root = os.path.join(capture_root_early, MASK_ROOT)
    if os.path.isdir(mask_root) and not getattr(args, "force_regen_masks", False):
        print(f"[run] mask root exists, skipping gen-masks subprocess: {mask_root}")
    else:
        this_script = os.path.abspath(__file__)
        sub_args = _build_gen_masks_subproc_args(args)
        cmd = [
            "conda", "run", "-n", args.sam3_env, "--no-capture-output",
            "python", this_script, *sub_args,
        ]
        print(f"[run] Launching mask generation in conda env '{args.sam3_env}':")
        print("      " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    cmd_optimize(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen-masks", help="Generate robot masks with SAM3 (sam3 env).")
    _add_common_args(p_gen)
    _add_gen_mask_args(p_gen)
    p_gen.set_defaults(func=cmd_gen_masks)

    p_opt = sub.add_parser("optimize", help="Silhouette-based qpos optimization (paradex env).")
    _add_common_args(p_opt)
    _add_optimize_args(p_opt)
    p_opt.set_defaults(func=cmd_optimize)

    p_run = sub.add_parser(
        "run",
        help="Run gen-masks (subprocess in sam3 env) + optimize end-to-end from the paradex env.",
    )
    _add_common_args(p_run)
    _add_gen_mask_args(p_run)
    _add_optimize_args(p_run)
    p_run.add_argument(
        "--sam3-env",
        type=str,
        default="sam3",
        help="conda env name for SAM3 mask generation subprocess.",
    )
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
