from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))

import argparse
import datetime
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import trimesh
import zmq

from paradex.calibration.utils import load_camparam, save_current_camparam
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.path import shared_dir


def _send_rpc_once(addr: str, req: dict, timeout_ms: int = 300000) -> dict:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(addr)
        sock.send_json(req)
        return sock.recv_json()
    finally:
        sock.close()
        ctx.term()


def _to_shared_data_path(abs_path: str) -> str:
    norm = os.path.normpath(abs_path)
    marker = f"{os.sep}shared_data{os.sep}"
    if marker in norm:
        tail = norm.split(marker, 1)[1]
        return os.path.join("shared_data", tail).replace(os.sep, "/")
    if norm.endswith(f"{os.sep}shared_data"):
        return "shared_data"
    raise ValueError(f"path is not under shared_data: {abs_path}")


_MANO_CACHE: Dict[bool, Any] = {}


def _get_mano(model_dir: str, is_rhand: bool):
    if is_rhand in _MANO_CACHE:
        return _MANO_CACHE[is_rhand]
    import smplx
    m = smplx.MANOLayer(model_path=model_dir, is_rhand=is_rhand,
                        use_pca=False, flat_hand_mean=True, batch_size=1)
    m.eval()
    _MANO_CACHE[is_rhand] = m
    return m


def _mano_forward_to_target(hd: Dict[str, Any], model_dir: str, is_rhand: bool):
    """Forward MANO with saved (target-frame) params -> (verts_target_cam, faces)."""
    import torch
    try:
        m = _get_mano(model_dir, is_rhand)
    except Exception as e:
        print(f"[proj] MANO load failed (is_rhand={is_rhand}): {e}")
        return None, None
    go = torch.tensor(hd["global_orient"], dtype=torch.float32).view(1, 1, 3, 3)
    hp = torch.tensor(hd["hand_pose"], dtype=torch.float32).view(1, 15, 3, 3)
    betas = torch.tensor(hd["betas"], dtype=torch.float32).view(1, -1)
    transl = torch.tensor(hd["transl"], dtype=torch.float32).view(1, 3)
    with torch.no_grad():
        out = m.forward(global_orient=go, hand_pose=hp, betas=betas, transl=transl,
                        pose2rot=False)
    verts = out.vertices[0].cpu().numpy().astype(np.float64)
    faces = np.asarray(m.faces, dtype=np.int32)
    return verts, faces


def _to_4x4(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = mat
        return out
    raise ValueError(f"matrix must be 4x4 or 3x4, got {mat.shape}")


def _normalize_pose_z_yaw(pose: np.ndarray, target_yaw_rad: float = 0.0) -> np.ndarray:
    """Rotate pose around world Z axis so its x-axis yaw matches target. Position fixed."""
    pose = _to_4x4(pose)
    R = pose[:3, :3]
    t = pose[:3, 3].copy()
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    delta = float(target_yaw_rad - yaw)
    cz, sz = float(np.cos(delta)), float(np.sin(delta))
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rz @ R
    out[:3, 3] = t
    return out


def _render_mesh_overlay(
    img: np.ndarray,
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    color: tuple = (80, 220, 80),
    alpha: float = 0.55,
) -> np.ndarray:
    """Depth-sorted filled-triangle overlay (cv2). verts_cam already in camera frame."""
    h, w = img.shape[:2]
    z = verts_cam[:, 2]
    valid = z > 1e-3
    proj = (K @ verts_cam.T).T
    z_safe = np.where(np.abs(proj[:, 2:3]) < 1e-9, 1e-9, proj[:, 2:3])
    uv = proj[:, :2] / z_safe

    f = np.asarray(faces, dtype=np.int64)
    fv = valid[f[:, 0]] & valid[f[:, 1]] & valid[f[:, 2]]
    f = f[fv]
    if len(f) == 0:
        return img.copy()

    v0 = verts_cam[f[:, 0]]; v1 = verts_cam[f[:, 1]]; v2 = verts_cam[f[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_norm = np.maximum(n_norm, 1e-8)
    n = n / n_norm
    # simple Lambertian shading from camera direction
    shade = np.clip(0.35 + 0.65 * np.abs(n[:, 2]), 0.0, 1.0)
    base = np.array(color, dtype=np.float32)
    face_colors = (shade[:, None] * base[None, :]).astype(np.uint8)

    z_mean = (z[f[:, 0]] + z[f[:, 1]] + z[f[:, 2]]) / 3.0
    order = np.argsort(-z_mean)  # back to front

    overlay = img.copy()
    uv_int = np.round(uv).astype(np.int32)
    for idx in order:
        tri = uv_int[f[idx]]
        if (tri[:, 0].max() < 0 or tri[:, 0].min() >= w or
                tri[:, 1].max() < 0 or tri[:, 1].min() >= h):
            continue
        cv2.fillConvexPoly(overlay, tri, color=tuple(int(c) for c in face_colors[idx]))
    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)


def _extract_pose_response(res: Dict[str, Any]) -> Dict[str, Any]:
    pose = res.get("object_6d", res)
    out = {
        "pose_world": pose.get("pose_world"),
        "pose_left_cam": pose.get("pose_left_cam"),
        "R_world": pose.get("R_world"),
        "t_world": pose.get("t_world"),
    }
    if out["pose_world"] is None:
        raise RuntimeError(f"invalid response, pose_world missing: {res}")
    return out

TARGET_SERIAL = "23029839"


def main():
    parser = argparse.ArgumentParser(description="Capture a single image from all cameras and save.")
    parser.add_argument("--save_path", default="nips2026/test_image", help="relative path under shared_data")
    parser.add_argument("--mesh_root", default="/home/temp_id/shared_data/mesh_blender",
                        help="root where {mesh_name}/{mesh_name}.obj live")
    parser.add_argument("--rcc_entry", default="image_main.py")
    parser.add_argument("--serial", default=TARGET_SERIAL, help="Camera serial to extract.")
    parser.add_argument("--rpc_addr", default="tcp://192.168.0.14:5570")
    parser.add_argument("--rpc_timeout_ms", type=int, default=300000)
    parser.add_argument("--mesh_name", "--name", dest="mesh_name", default=None,
                        help="mesh name for object 6d inference; if omitted, RPC step is skipped")
    parser.add_argument("--normalize_z", action="store_true",
                        help="rotate object pose around world Z so its x-axis yaw = normalize_z_deg")
    parser.add_argument("--normalize_z_deg", type=float, default=0.0)
    parser.add_argument("--run_hamer", action="store_true",
                        help="After capture, run HaMeR + MANO fit (with betas) on the captured frame.")
    parser.add_argument("--hand", choices=["right", "left", "both"], default="both")
    parser.add_argument("--hamer_python", default="/home/temp_id/anaconda3/envs/hamer-mp/bin/python")
    parser.add_argument("--hamer_repo", default="/home/temp_id/hamer-mediapipe")
    parser.add_argument("--mano_model_dir", default="/home/temp_id/hamer-mediapipe/_DATA/data/mano",
                        help="dir holding MANO_RIGHT.pkl and MANO_LEFT.pkl")
    parser.add_argument("--hamer_frame_name", default="00001.png",
                        help="Frame name used inside the staged scene structure.")
    parser.add_argument("--resume_dir", default=None,
                        help="Skip capture; use existing absolute capture dir (e.g. shared_data/.../20260506_161016).")
    args = parser.parse_args()

    if args.resume_dir is not None:
        abs_save_path = os.path.abspath(args.resume_dir)
        if not os.path.isdir(abs_save_path):
            raise FileNotFoundError(f"--resume_dir not found: {abs_save_path}")
        print(f"[resume] reusing {abs_save_path} (skipping capture)")
    else:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_save_path = os.path.join(args.save_path, date_str)
        abs_save_path = os.path.join(shared_dir, rel_save_path)

        rcc = remote_camera_controller(args.rcc_entry)
        save_current_camparam(abs_save_path)

        print("Press 'c' + Enter to start 10s countdown capture...")
        while True:
            key = input().strip().lower()
            if key == "c":
                break
        for i in range(10, 0, -1):
            print(f"  Capturing in {i}...")
            time.sleep(1)

        print(f"Capturing image to {abs_save_path}")
        try:
            rcc.start("image", False, f"shared_data/{rel_save_path}/raw")
            rcc.stop()
        finally:
            rcc.end()

        # Wait for remote images to be saved via NFS/shared storage
        img_path = os.path.join(abs_save_path, "raw", "images", f"{args.serial}.png")
        print("Waiting for image to arrive...")
        for _ in range(30):
            if os.path.exists(img_path):
                break
            time.sleep(1)
        else:
            print(f"[WARN] Timed out waiting for {img_path}")

    # Extract target camera image + intrinsic into a single folder
    serial = args.serial
    extract_dir = os.path.join(abs_save_path, serial)
    os.makedirs(extract_dir, exist_ok=True)

    img_path = os.path.join(abs_save_path, "raw", "images", f"{serial}.png")
    if os.path.exists(img_path):
        shutil.copy2(img_path, os.path.join(extract_dir, f"{serial}.png"))
    else:
        print(f"[WARN] Image not found: {img_path}")

    intr_path = os.path.join(abs_save_path, "cam_param", "intrinsics.json")
    if os.path.exists(intr_path):
        with open(intr_path, "r") as f:
            all_intr = json.load(f)
        if serial in all_intr:
            with open(os.path.join(extract_dir, "intrinsics.json"), "w") as f:
                json.dump({serial: all_intr[serial]}, f, indent=2)
        else:
            print(f"[WARN] Serial {serial} not found in intrinsics.json")

    print(f"Done. Extracted to {extract_dir}")

    object_pose_target_4x4: Optional[np.ndarray] = None
    if args.mesh_name is not None:
        req = {
            "command": "infer",
            "image_path": _to_shared_data_path(abs_save_path),
            "mesh_name": args.mesh_name,
        }
        print(f"Requesting object 6d to {args.rpc_addr} for mesh {args.mesh_name}")
        res = _send_rpc_once(args.rpc_addr, req, timeout_ms=args.rpc_timeout_ms)
        pose_out = _extract_pose_response(res)

        object_json_path = os.path.join(abs_save_path, "object_6d.json")
        with open(object_json_path, "w", encoding="utf-8") as f:
            json.dump(pose_out, f, ensure_ascii=False, indent=2)
        print(f"saved: {object_json_path}")
        print(json.dumps(pose_out, ensure_ascii=False, indent=2))

        _, extrinsic = load_camparam(abs_save_path)
        if args.serial not in extrinsic:
            raise KeyError(f"serial {args.serial} not found in extrinsics.json")
        E_target = np.eye(4, dtype=np.float64)
        E_target[:3, :] = np.asarray(extrinsic[args.serial], dtype=np.float64)

        pose_world_raw = _to_4x4(np.asarray(pose_out["pose_world"], dtype=np.float64))
        if args.normalize_z:
            pose_world_4x4 = _normalize_pose_z_yaw(
                pose_world_raw, target_yaw_rad=np.deg2rad(float(args.normalize_z_deg))
            )
            print(f"[normalize_z] yaw -> {args.normalize_z_deg} deg around world Z")
        else:
            pose_world_4x4 = pose_world_raw
        pose_target = E_target @ pose_world_4x4
        object_pose_target_4x4 = pose_target

        pose_target_out = {
            "serial": args.serial,
            "mesh_name": args.mesh_name,
            "pose_world_raw": pose_world_raw.tolist(),
            "pose_world_normalized": pose_world_4x4.tolist(),
            "pose_target_cam": pose_target.tolist(),
            "R_target_cam": pose_target[:3, :3].tolist(),
            "t_target_cam": pose_target[:3, 3].tolist(),
            "extrinsic_target_cam_from_world": E_target.tolist(),
            "normalize_z": bool(args.normalize_z),
            "normalize_z_deg": float(args.normalize_z_deg),
        }
        target_json_path = os.path.join(extract_dir, "object_6d.json")
        with open(target_json_path, "w", encoding="utf-8") as f:
            json.dump(pose_target_out, f, ensure_ascii=False, indent=2)
        print(f"saved: {target_json_path}")

        _copy_object_mesh(extract_dir, args.mesh_root, args.mesh_name)

    if args.run_hamer:
        run_hamer_and_save(args, abs_save_path, extract_dir)

    if args.mesh_name is not None or args.run_hamer:
        _render_final_projection(extract_dir, args, args.mesh_name, object_pose_target_4x4)

    print(f"Final outputs at: {extract_dir}")


def run_hamer_and_save(args, abs_save_path: str, extract_dir: str) -> None:
    raw_images_dir = os.path.join(abs_save_path, "raw", "images")
    if not os.path.isdir(raw_images_dir):
        raise FileNotFoundError(f"raw images dir missing: {raw_images_dir}")

    frames_root = os.path.join(abs_save_path, "frames")
    os.makedirs(frames_root, exist_ok=True)
    staged = 0
    for fn in sorted(os.listdir(raw_images_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        cam_dir = os.path.join(frames_root, stem)
        os.makedirs(cam_dir, exist_ok=True)
        dst = os.path.join(cam_dir, args.hamer_frame_name)
        if not os.path.lexists(dst):
            try:
                os.symlink(os.path.join(raw_images_dir, fn), dst)
            except OSError:
                shutil.copy2(os.path.join(raw_images_dir, fn), dst)
        staged += 1
    if staged == 0:
        raise RuntimeError(f"no images staged for hamer in {frames_root}")
    print(f"[hamer] staged {staged} cams under {frames_root}")

    fit_json_path = os.path.join(abs_save_path, "hand", "mano_fit_betas.json")
    helper = os.path.abspath(os.path.join(os.path.dirname(__file__), "_hamer_single_frame_fit.py"))
    cmd = [
        args.hamer_python, helper,
        "--scene_dir", abs_save_path,
        "--frame_name", args.hamer_frame_name,
        "--hand", args.hand,
        "--out_path", fit_json_path,
    ]
    env = os.environ.copy()
    env["HAMER_REPO"] = args.hamer_repo
    env["PYTHONPATH"] = args.hamer_repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print("[hamer] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=args.hamer_repo, env=env)

    with open(fit_json_path, "r", encoding="utf-8") as f:
        fit_all = json.load(f)

    _, extrinsic = load_camparam(abs_save_path)
    if args.serial not in extrinsic:
        raise KeyError(f"serial {args.serial} not in extrinsics")
    E_target = np.eye(4, dtype=np.float64)
    E_target[:3, :] = np.asarray(extrinsic[args.serial], dtype=np.float64)
    R_E = E_target[:3, :3]
    t_E = E_target[:3, 3]

    out_target = {
        "serial": args.serial,
        "frame_name": fit_all.get("frame_name"),
        "extrinsic_target_cam_from_world": E_target.tolist(),
    }
    for hand_name, fit in fit_all.get("hands", {}).items():
        if not fit.get("detected"):
            out_target[hand_name] = {"detected": False,
                                     "n_views_detected": fit.get("n_views_detected")}
            continue
        go = np.asarray(fit["global_orient"], dtype=np.float64)
        transl = np.asarray(fit["transl"], dtype=np.float64)
        J0 = np.asarray(fit["J0_betas"], dtype=np.float64)

        new_go = R_E @ go.reshape(3, 3)
        new_transl = R_E @ (transl.reshape(3) + J0) - J0 + t_E

        joints_world = np.asarray(fit["joints_world"], dtype=np.float64)
        verts_world = np.asarray(fit["verts_world"], dtype=np.float64)
        joints_target = joints_world @ R_E.T + t_E
        verts_target = verts_world @ R_E.T + t_E

        out_target[hand_name] = {
            "global_orient": new_go.reshape(go.shape).tolist(),
            "hand_pose": fit["hand_pose"],
            "betas": fit["betas"],
            "transl": new_transl.reshape(transl.shape).tolist(),
        }
    target_json = os.path.join(extract_dir, "mano_params.json")
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(out_target, f)
    print(f"[hamer] saved (target frame): {target_json}")
    print(f"[hamer] world-frame fit:     {fit_json_path}")


def _render_final_projection(
    extract_dir: str,
    args,
    mesh_name: Optional[str],
    object_pose_target_4x4: Optional[np.ndarray],
) -> None:
    img_path = os.path.join(extract_dir, f"{args.serial}.png")
    if not os.path.exists(img_path):
        print(f"[proj] no target image at {img_path}, skipping projection")
        return
    intr_path = os.path.join(extract_dir, "intrinsics.json")
    if not os.path.exists(intr_path):
        print(f"[proj] no intrinsics at {intr_path}, skipping projection")
        return

    with open(intr_path, "r", encoding="utf-8") as f:
        intr_all = json.load(f)
    intr = intr_all[args.serial]
    K = np.asarray(intr["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[proj] failed to read {img_path}")
        return
    out = img.copy()

    # Object mesh overlay
    if mesh_name is not None and object_pose_target_4x4 is not None:
        mesh_path = os.path.join(args.mesh_root, mesh_name, f"{mesh_name}.obj")
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path, process=False, force="mesh")
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            verts_h = np.hstack([verts, np.ones((len(verts), 1))])
            verts_cam = (object_pose_target_4x4 @ verts_h.T).T[:, :3]
            out = _render_mesh_overlay(out, verts_cam, faces, K,
                                       color=(80, 200, 255), alpha=0.55)
        else:
            print(f"[proj] object mesh not found: {mesh_path}")

    # Hand mesh overlay (verts already in target frame)
    mano_json = os.path.join(extract_dir, "mano_params.json")
    if os.path.exists(mano_json):
        with open(mano_json, "r", encoding="utf-8") as f:
            mp_data = json.load(f)
        hand_colors = {"right": (90, 230, 90), "left": (230, 140, 90)}
        for hname, is_rhand in (("right", True), ("left", False)):
            hd = mp_data.get(hname)
            if not isinstance(hd, dict) or "global_orient" not in hd:
                continue
            verts_cam, faces_h = _mano_forward_to_target(hd, args.mano_model_dir, is_rhand)
            if verts_cam is None:
                continue
            out = _render_mesh_overlay(out, verts_cam, faces_h, K,
                                       color=hand_colors.get(hname, (200, 200, 200)),
                                       alpha=0.6)

    proj_path = os.path.join(extract_dir, "projection.png")
    cv2.imwrite(proj_path, out)
    print(f"[proj] saved: {proj_path}")


def _copy_object_mesh(extract_dir: str, mesh_root: str, mesh_name: str) -> None:
    src = os.path.join(mesh_root, mesh_name, f"{mesh_name}.obj")
    if not os.path.exists(src):
        print(f"[mesh] not found: {src}")
        return
    dst = os.path.join(extract_dir, f"{mesh_name}.obj")
    shutil.copy2(src, dst)
    print(f"[mesh] copied: {dst}")


if __name__ == "__main__":
    main()
