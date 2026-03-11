import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import trimesh

PARADEX_ROOT = os.path.join(os.path.expanduser("~"), "paradex")
if PARADEX_ROOT not in sys.path:
    sys.path.insert(0, PARADEX_ROOT)

from paradex.robot.inspire import inspire_action_to_qpos, inspire_f1_action_to_qpos_dof6
from paradex.utils.load_data import load_series, resample_to
from paradex.utils.path import rsc_path, shared_dir
from paradex.visualization.robot import RobotModule


ARM_LINK_NAMES = {
'base.obj', 'link1.obj', 'link2.obj', 'link3.obj', 'link4.obj', 'link5.obj', 'link6.obj'
}

TACTILE_VERTEX_MAP = {
    "little_tip":    ("left_pinky_3",  [10051, 17056, 23370, 19478]),
    "little_nail":   ("left_pinky_3",  [12084, 8347, 11491, 12122]),
    "little_pad":    ("left_pinky_2",  [10744, 20595, 27203,  8159]),

    "ring_tip":      ("left_ring_3",   [11090, 17235, 23668, 19815]),
    "ring_nail":     ("left_ring_3",   [13153,  9108, 12550, 13184]),
    "ring_pad":      ("left_ring_2",   [11835, 21270, 28049,  9152]),

    "middle_tip":    ("left_middle_3", [17628, 23920, 18195, 25321]),
    "middle_nail":   ("left_middle_3", [19999, 16826, 19402, 20026]),
    "middle_pad":    ("left_middle_2", [31488, 17368, 17385, 32330]),

    "index_tip":     ("left_index_2",  [16097, 22352, 16853, 23820]),
    "index_nail":    ("left_index_2",  [20197, 12013, 4598, 13379]),
    "index_pad":     ("left_index_1",  [36688, 18750, 19219, 37862]),

    "thumb_tip":     ("left_thumb_4",  [9676, 23934, 26403, 28719]),
    "thumb_nail":    ("left_thumb_4",  [18621, 16421, 35555, 37778]),
    "thumb_middle":  ("left_thumb_2",  [15649, 22156, 14346, 5837]),
    "thumb_pad":     ("left_thumb_2",  [19300, 18427, 10008, 8949]),

    "palm":          ("base_link",     [68864, 61448, 68207, 67649]),
}

INSPIRE_TACTILE_LINK_CANDIDATES = {
    "little_tip": ["right_little_2", "left_little_2"],
    "little_nail": ["right_little_2", "left_little_2"],
    "little_pad": ["right_little_1", "left_little_1"],
    "ring_tip": ["right_ring_2", "left_ring_2"],
    "ring_nail": ["right_ring_2", "left_ring_2"],
    "ring_pad": ["right_ring_1", "left_ring_1"],
    "middle_tip": ["right_middle_2", "left_middle_2"],
    "middle_nail": ["right_middle_2", "left_middle_2"],
    "middle_pad": ["right_middle_1", "left_middle_1"],
    "index_tip": ["right_index_2", "left_index_2"],
    "index_nail": ["right_index_2", "left_index_2"],
    "index_pad": ["right_index_1", "left_index_1"],
    "thumb_tip": ["right_thumb_4", "left_thumb_4", "right_thumb_3", "left_thumb_3"],
    "thumb_nail": ["right_thumb_4", "left_thumb_4", "right_thumb_3", "left_thumb_3"],
    "thumb_middle": ["right_thumb_2", "left_thumb_2"],
    "thumb_pad": ["right_thumb_2", "left_thumb_2", "right_thumb_1", "left_thumb_1"],
    "palm": ["base_link"],
}

TACTILE_LAYOUT = {
    "little_tip":    (3000, 3, 3),
    "little_nail":   (3018, 12, 8),
    "little_pad":    (3210, 10, 8),
    "ring_tip":      (3370, 3, 3),
    "ring_nail":     (3388, 12, 8),
    "ring_pad":      (3580, 10, 8),
    "middle_tip":    (3740, 3, 3),
    "middle_nail":   (3758, 12, 8),
    "middle_pad":    (3950, 10, 8),
    "index_tip":     (4110, 3, 3),
    "index_nail":    (4128, 12, 8),
    "index_pad":     (4320, 10, 8),
    "thumb_tip":     (4480, 3, 3),
    "thumb_nail":    (4498, 12, 8),
    "thumb_middle":  (4690, 3, 3),
    "thumb_pad":     (4708, 12, 8),
    "palm":          (4900, 8, 14),
}

ZONE_TO_LINK = {
    "little": "little_force_sensor",
    "ring": "ring_force_sensor",
    "middle": "middle_force_sensor",
    "index": "index_force_sensor",
    "thumb": "thumb_force_sensor",
    "palm_right": "plam_force_sensor",
    "palm_middle": "plam_force_sensor",
    "palm_left": "plam_force_sensor",
}
PALM_ZONES = ("palm_right", "palm_middle", "palm_left")
TACTILE_ARROW_RGBA = np.array([255, 0, 0, 255], dtype=np.uint8)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "1", "y", "on"):
        return True
    if v in ("no", "false", "f", "0", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{v}'")


def parse_camera_ids(raw: Optional[str]) -> Optional[Set[str]]:
    if raw is None:
        return None
    vals = [x.strip() for x in raw.split(",")]
    vals = [x for x in vals if x]
    return set(vals)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_cam_from_world(extrinsic_value: Any) -> np.ndarray:
    arr = np.asarray(extrinsic_value, dtype=float)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=float)
        out[:3, :] = arr
        return out
    if arr.shape == (4, 4):
        return arr
    if arr.size == 12:
        out = np.eye(4, dtype=float)
        out[:3, :] = arr.reshape(3, 4)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {arr.shape} (expected 3x4 or 4x4)")


def build_intrinsic_for_viewer(serial: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if "intrinsics_undistort" in payload:
        k = np.asarray(payload["intrinsics_undistort"], dtype=float)
    elif "original_intrinsics" in payload:
        k = np.asarray(payload["original_intrinsics"], dtype=float)
    else:
        raise ValueError(f"{serial}: missing 'intrinsics_undistort'/'original_intrinsics'")
    if k.shape != (3, 3):
        raise ValueError(f"{serial}: intrinsic matrix shape {k.shape} is not 3x3")

    width = payload.get("width")
    height = payload.get("height")
    if width is None or height is None:
        cx = float(k[0, 2])
        cy = float(k[1, 2])
        width = int(round(cx * 2.0))
        height = int(round(cy * 2.0))
        print(f"[WARN] {serial}: missing width/height, fallback to width={width}, height={height}")

    return {
        "intrinsics_undistort": k.tolist(),
        "width": int(width),
        "height": int(height),
    }


def validate_rotation_matrix(r: np.ndarray, det_tol: float = 1e-2, orth_tol: float = 1e-2) -> Tuple[bool, str]:
    det = np.linalg.det(r)
    orth_err = np.linalg.norm(r.T @ r - np.eye(3), ord="fro")
    if abs(det - 1.0) > det_tol:
        return False, f"det(R)={det:.6f}"
    if orth_err > orth_tol:
        return False, f"orth_error={orth_err:.6e}"
    return True, "ok"


def add_cameras_to_scene(
    vis,
    capture_root: str,
    c2r: np.ndarray,
    frustum_size: float,
    show_axes: bool,
    show_labels: bool,
    image_root: Optional[str] = None,
    initial_frame_id: Optional[int] = None,
    view_scale: float = 1.0,
    fov_scale: float = 1.0,
    selected_ids: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    uniform_camera_color = (80, 80, 80)
    camparam_dir = os.path.join(capture_root, "cam_param")
    intr_path = os.path.join(camparam_dir, "intrinsics.json")
    extr_path = os.path.join(camparam_dir, "extrinsics.json")
    if not (os.path.exists(intr_path) and os.path.exists(extr_path)):
        print(f"[WARN] cam_param not found under {camparam_dir}; skipping camera visualization.")
        return 0, 0

    intrinsics = load_json(intr_path)
    extrinsics = load_json(extr_path)
    k_intr = set(intrinsics.keys())
    k_extr = set(extrinsics.keys())
    serials = sorted(k_intr & k_extr)
    if (k_extr - k_intr):
        print(f"[WARN] Missing intrinsics for cameras: {sorted(k_extr - k_intr)}")
    if (k_intr - k_extr):
        print(f"[WARN] Missing extrinsics for cameras: {sorted(k_intr - k_extr)}")
    if selected_ids is not None:
        unknown = sorted(selected_ids - (k_intr | k_extr))
        if unknown:
            print(f"[WARN] Requested camera IDs not found: {unknown}")
        serials = [s for s in serials if s in selected_ids]

    added = 0
    skipped = 0
    use_camera_view = image_root is not None and os.path.isdir(image_root)
    if image_root is not None and not os.path.isdir(image_root):
        print(f"[WARN] camera image root not found: {image_root}. Frustum-only mode.")
    for serial in serials:
        try:
            intrinsic_view = build_intrinsic_for_viewer(serial, intrinsics[serial])
            cam_from_world = parse_cam_from_world(extrinsics[serial])
            cam_from_robot = cam_from_world @ c2r
            valid, reason = validate_rotation_matrix(cam_from_robot[:3, :3])
            if not valid:
                print(f"[WARN] camera {serial}: invalid rotation ({reason}), skipped.")
                skipped += 1
                continue
            world_from_cam = np.linalg.inv(cam_from_robot)

            image = None
            if use_camera_view and initial_frame_id is not None:
                img_path = os.path.join(image_root, serial, f"{int(initial_frame_id):05d}.jpg")
                if os.path.exists(img_path):
                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        if view_scale != 1.0:
                            h, w = image.shape[:2]
                            nw = max(1, int(round(w * view_scale)))
                            nh = max(1, int(round(h * view_scale)))
                            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

            vis.add_camera(
                name=serial,
                extrinsic=world_from_cam,
                intrinsic=intrinsic_view,
                color=uniform_camera_color,
                size=frustum_size,
                show_axes=show_axes,
                image=image,
                fov_scale=fov_scale,
            )
            if show_labels:
                vis.server.scene.add_label(f"/cameras/{serial}_frame/label", serial)
            added += 1
        except Exception as e:
            print(f"[WARN] camera {serial}: skipped due to error: {e}")
            skipped += 1
    return added, skipped


def try_load_transform_from_npy(object_tracking_dir: str, object_frame_idx: int) -> Optional[np.ndarray]:
    npy_path = os.path.join(object_tracking_dir, f"frame_{object_frame_idx}.npy")
    if not os.path.exists(npy_path):
        return None
    arr = np.load(npy_path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.shape != (4, 4):
        return None
    return arr.astype(np.float64)


def build_npz_transform_lookup(npz_path: str) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if not os.path.exists(npz_path):
        return out

    data = np.load(npz_path, allow_pickle=True)
    for key in data.files:
        match = re.fullmatch(r"frame_(\d+)", key)
        if match:
            idx = int(match.group(1))
            arr = np.asarray(data[key])
            if arr.shape == (4, 4):
                out[idx] = arr.astype(np.float64)

    for key in ("obj_T_frames", "arr_0"):
        if key not in data.files:
            continue
        arr = np.asarray(data[key])
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            for idx in range(arr.shape[0]):
                out.setdefault(idx, arr[idx].astype(np.float64))
    return out


def get_object_transform(
    object_tracking_dir: str,
    npz_lookup: Dict[int, np.ndarray],
    object_frame_idx: int,
) -> Optional[np.ndarray]:
    transform = try_load_transform_from_npy(object_tracking_dir, object_frame_idx)
    if transform is not None:
        return transform
    return npz_lookup.get(object_frame_idx)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points.astype(np.float64), ones], axis=1)
    transformed = (transform @ points_h.T).T
    return transformed[:, :3].astype(np.float32)


def compute_min_vertex_distances(
    src_vertices: np.ndarray,
    dst_vertices: np.ndarray,
    src_chunk: int = 2048,
    dst_chunk: int = 8192,
) -> np.ndarray:
    if src_vertices.ndim != 2 or src_vertices.shape[1] != 3:
        raise ValueError(f"Expected src vertices shape (N,3), got {src_vertices.shape}")
    if dst_vertices.ndim != 2 or dst_vertices.shape[1] != 3:
        raise ValueError(f"Expected dst vertices shape (M,3), got {dst_vertices.shape}")

    out = np.full((src_vertices.shape[0],), np.inf, dtype=np.float32)
    for src_start in range(0, src_vertices.shape[0], src_chunk):
        src_stop = min(src_start + src_chunk, src_vertices.shape[0])
        src_chunk_vertices = src_vertices[src_start:src_stop].astype(np.float32, copy=False)
        best_sq = np.full((src_chunk_vertices.shape[0],), np.inf, dtype=np.float32)
        for dst_start in range(0, dst_vertices.shape[0], dst_chunk):
            dst_stop = min(dst_start + dst_chunk, dst_vertices.shape[0])
            dst_chunk_vertices = dst_vertices[dst_start:dst_stop].astype(np.float32, copy=False)
            diff = src_chunk_vertices[:, None, :] - dst_chunk_vertices[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            best_sq = np.minimum(best_sq, np.min(dist_sq, axis=1))
        out[src_start:src_stop] = np.sqrt(best_sq)
    return out


def unpack_tactile_frame(raw, index):
    out = {}
    for k, (off, r, c) in index.items():
        block = raw[off:off + r * c]
        out[k] = block.reshape(r, c)
    return out


def build_tactile_index_from_layout(layout):
    idx, off = {}, 0
    for k, (_, r, c) in layout.items():
        idx[k] = (off, r, c)
        off += r * c
    return idx


def _flatten_tactile_frame(frame: Dict[str, Any], layout: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    parts = []
    for name, (_, r, c) in layout.items():
        if name not in frame:
            parts.append(np.zeros(r * c, dtype=float))
            continue
        arr = np.asarray(frame[name])
        if arr.ndim == 2:
            if arr.shape != (r, c):
                if arr.size != r * c:
                    raise ValueError(f"Tactile '{name}' has shape {arr.shape}, expected {(r, c)}")
                arr = arr.reshape(r, c)
        elif arr.ndim == 1:
            if arr.size != r * c:
                raise ValueError(f"Tactile '{name}' has size {arr.size}, expected {r * c}")
            arr = arr.reshape(r, c)
        else:
            raise ValueError(f"Tactile '{name}' has shape {arr.shape}, expected 1D/2D")
        parts.append(arr.reshape(-1))
    return np.concatenate(parts, axis=0).astype(float)


def _unwrap_tactile_item(item: Any) -> Any:
    if isinstance(item, np.ndarray) and item.dtype == object:
        if item.shape == ():
            return item.item()
        if item.size == 1:
            return item.reshape(()).item()
    return item


def normalize_tactile_sequence(payload: Any, layout: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    if isinstance(payload, np.ndarray) and payload.dtype != object and payload.ndim == 2:
        return payload.astype(float)

    if isinstance(payload, np.ndarray) and payload.dtype == object:
        if payload.shape == ():
            payload = payload.item()
        else:
            payload = payload.tolist()

    if isinstance(payload, dict):
        t_len = None
        for v in payload.values():
            arr = np.asarray(v)
            if arr.ndim == 3:
                t_len = arr.shape[0]
                break
        if t_len is None:
            return _flatten_tactile_frame(payload, layout)[None, :]

        frames = []
        for t in range(t_len):
            frame = {}
            for name, val in payload.items():
                arr = np.asarray(val)
                frame[name] = arr[t] if arr.ndim == 3 else arr
            frames.append(_flatten_tactile_frame(frame, layout))
        return np.stack(frames, axis=0)

    if isinstance(payload, (list, tuple)):
        first = _unwrap_tactile_item(payload[0]) if payload else None
        if payload and isinstance(first, dict):
            frames = [_flatten_tactile_frame(_unwrap_tactile_item(frame), layout) for frame in payload]
            return np.stack(frames, axis=0)
        arr = np.asarray(payload)
        if arr.ndim == 2:
            return arr.astype(float)

    raise ValueError(f"Unsupported tactile payload type: {type(payload)}")


def normalize_force_dict_sequence(payload: Any) -> List[Dict[str, float]]:
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.tolist()
    if isinstance(payload, dict):
        return [{k: float(v) for k, v in payload.items()}]
    if isinstance(payload, (list, tuple)):
        out = []
        for item in payload:
            if isinstance(item, np.ndarray) and item.shape == () and item.dtype == object:
                item = item.item()
            if not isinstance(item, dict):
                continue
            out.append({k: float(v) for k, v in item.items()})
        return out
    raise ValueError(f"Unsupported force payload type: {type(payload)}")


def resample_force_dict_sequence(seq: List[Dict[str, float]], target_len: int) -> List[Dict[str, float]]:
    if target_len <= 0:
        return []
    if len(seq) == 0:
        return [{} for _ in range(target_len)]
    if len(seq) == target_len:
        return seq
    src = np.linspace(0.0, len(seq) - 1, target_len)
    idx = np.clip(np.round(src).astype(int), 0, len(seq) - 1)
    return [seq[i] for i in idx]


def _extract_zone_force(tactile: Dict[str, float], zone: str) -> Tuple[float, float, float]:
    normal = float(tactile.get(f"{zone}_normal_force", 0.0))
    tangent = float(tactile.get(f"{zone}_tangential_force", 0.0))
    direction_deg = float(tactile.get(f"{zone}_tangential_direction", -1.0))
    return normal, tangent, direction_deg


@dataclass
class SensorFrame:
    link_name: str
    anchor_local: np.ndarray
    normal_local: np.ndarray
    tangent_x_local: np.ndarray
    tangent_y_local: np.ndarray


def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return fallback.copy()
    return v / n


def _estimate_surface_frame(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    centered = vertices - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    normal = eigvecs[:, order[0]]
    tangent_x = eigvecs[:, order[-1]]
    normal = _safe_normalize(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    tangent_x = tangent_x - np.dot(tangent_x, normal) * normal
    tangent_x = _safe_normalize(tangent_x, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    proj = vertices @ normal
    q_high = np.quantile(proj, 0.9)
    q_low = np.quantile(proj, 0.1)
    if abs(q_low) > abs(q_high):
        normal = -normal
        surf_mask = proj <= q_low
    else:
        surf_mask = proj >= q_high
    if np.count_nonzero(surf_mask) < 4:
        surf_mask = np.ones(len(vertices), dtype=bool)
    anchor = vertices[surf_mask].mean(axis=0)

    if np.dot(tangent_x, np.array([1.0, 0.0, 0.0], dtype=np.float64)) < 0:
        tangent_x = -tangent_x
    tangent_y = _safe_normalize(np.cross(normal, tangent_x), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    tangent_x = _safe_normalize(np.cross(tangent_y, normal), np.array([1.0, 0.0, 0.0], dtype=np.float64))
    return anchor, normal, tangent_x, tangent_y


def _build_sensor_frames_from_local_mesh(
    local_meshes_by_link: Dict[str, trimesh.Trimesh], zone_to_link: Dict[str, str]
) -> Dict[str, SensorFrame]:
    frames = {}
    for zone in ("little", "ring", "middle", "index", "thumb"):
        link = zone_to_link[zone]
        tm = local_meshes_by_link.get(link)
        if tm is None or len(tm.vertices) < 4:
            continue
        anchor, normal, tx, ty = _estimate_surface_frame(np.asarray(tm.vertices))
        frames[zone] = SensorFrame(link, anchor, normal, tx, ty)

    palm_link = zone_to_link["palm_middle"]
    palm_tm = local_meshes_by_link.get(palm_link)
    if palm_tm is not None and len(palm_tm.vertices) > 8:
        verts = np.asarray(palm_tm.vertices)
        _, normal, tx, ty = _estimate_surface_frame(verts)
        split_axis = verts @ tx
        q1, q2 = np.quantile(split_axis, [1.0 / 3.0, 2.0 / 3.0])
        masks = {
            "palm_right": split_axis <= q1,
            "palm_middle": (split_axis > q1) & (split_axis <= q2),
            "palm_left": split_axis > q2,
        }
        for zone in PALM_ZONES:
            mask = masks[zone]
            if np.count_nonzero(mask) < 8:
                mask = np.ones(len(verts), dtype=bool)
            seg = verts[mask]
            seg_proj = seg @ normal
            t = np.quantile(seg_proj, 0.85)
            seg_mask = seg_proj >= t
            if np.count_nonzero(seg_mask) < 4:
                seg_mask = np.ones(len(seg), dtype=bool)
            anchor = seg[seg_mask].mean(axis=0)
            frames[zone] = SensorFrame(palm_link, anchor, normal, tx, ty)
    return frames


def _world_sensor_frame(sensor: SensorFrame, link_pose: np.ndarray):
    r = link_pose[:3, :3]
    p = link_pose[:3, 3]
    anchor = r @ sensor.anchor_local + p
    normal = _safe_normalize(r @ sensor.normal_local, np.array([0.0, 0.0, 1.0]))
    tx = _safe_normalize(r @ sensor.tangent_x_local, np.array([1.0, 0.0, 0.0]))
    ty = _safe_normalize(r @ sensor.tangent_y_local, np.array([0.0, 1.0, 0.0]))
    return anchor, normal, tx, ty


def _parse_xyz(text: str, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    vals = str(text).split() if text is not None else []
    if len(vals) != 3:
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in vals], dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    rr, rp, ry = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(rr), np.sin(rr)
    cp, sp = np.cos(rp), np.sin(rp)
    cy, sy = np.cos(ry), np.sin(ry)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry_m = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry_m @ rx


def _parse_rgba(text: str) -> Optional[np.ndarray]:
    if not text:
        return None
    vals = [float(x) for x in str(text).split()]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        return None
    rgba = np.clip(np.array(vals, dtype=np.float64), 0.0, 1.0)
    return (rgba * 255.0).astype(np.uint8)


def load_local_link_meshes_from_urdf(
    urdf_path: str, required_links: Optional[Set[str]] = None
) -> Dict[str, trimesh.Trimesh]:
    root = ET.parse(urdf_path).getroot()
    base_dir = os.path.dirname(urdf_path)
    material_rgba: Dict[str, np.ndarray] = {}
    for material in root.findall("material"):
        name = material.attrib.get("name", "")
        color_node = material.find("color")
        if not name or color_node is None:
            continue
        rgba = _parse_rgba(color_node.attrib.get("rgba", ""))
        if rgba is not None:
            material_rgba[name] = rgba

    out: Dict[str, trimesh.Trimesh] = {}
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        if required_links is not None and link_name not in required_links:
            continue
        parts = []
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_node = geom.find("mesh")
            if mesh_node is None:
                continue
            mesh_rel = mesh_node.attrib.get("filename", "")
            if not mesh_rel:
                continue
            mesh_path = os.path.join(base_dir, mesh_rel)
            if not os.path.exists(mesh_path):
                continue
            try:
                loaded = trimesh.load(mesh_path, force="mesh", process=False)
            except Exception:
                continue
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not geoms:
                    continue
                tm = trimesh.util.concatenate(geoms)
            else:
                tm = loaded
            tm = tm.copy()
            scale = _parse_xyz(mesh_node.attrib.get("scale", "1 1 1"), (1.0, 1.0, 1.0))
            tm.vertices = tm.vertices * scale[None, :]
            origin_node = visual.find("origin")
            if origin_node is not None:
                xyz = _parse_xyz(origin_node.attrib.get("xyz", "0 0 0"))
                rpy = _parse_xyz(origin_node.attrib.get("rpy", "0 0 0"))
                t = np.eye(4, dtype=np.float64)
                t[:3, :3] = _rpy_to_matrix(rpy)
                t[:3, 3] = xyz
                tm.apply_transform(t)
            material_node = visual.find("material")
            if material_node is not None and len(tm.vertices) > 0:
                visual_rgba = None
                color_node = material_node.find("color")
                if color_node is not None:
                    visual_rgba = _parse_rgba(color_node.attrib.get("rgba", ""))
                else:
                    material_name = material_node.attrib.get("name", "")
                    if material_name in material_rgba:
                        visual_rgba = material_rgba[material_name]
                if visual_rgba is not None:
                    tm.visual.vertex_colors = np.tile(visual_rgba[None, :], (len(tm.vertices), 1))
            parts.append(tm)
        if parts:
            out[link_name] = trimesh.util.concatenate(parts)
    return out


def compute_contact_arrow(tm, vids):
    v = tm.vertices[vids]
    n = tm.vertex_normals[vids]
    c = v.mean(axis=0)
    n = n.mean(axis=0)
    n /= np.linalg.norm(n) + 1e-8
    return c, n


def compute_contact_arrow_safe(tm: trimesh.Trimesh, vids: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(vids) == 0 or len(tm.vertices) == 0:
        return None
    max_vid = max(int(v) for v in vids)
    if max_vid >= len(tm.vertices):
        return None
    return compute_contact_arrow(tm, vids)


def compute_contact_arrow_fallback(tm: trimesh.Trimesh) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(tm.vertices) == 0:
        return None
    c = np.asarray(tm.vertices).mean(axis=0)
    vn = getattr(tm, "vertex_normals", None)
    if vn is not None and len(vn) > 0:
        n = np.asarray(vn).mean(axis=0)
    else:
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    n = n / (np.linalg.norm(n) + 1e-8)
    return c, n


def make_arrow_mesh(start, direction, length, color_rgba):
    if length < 1e-6:
        return None
    shaft_radius = 0.005
    head_radius = 0.006
    shaft_height = length * 0.7
    head_height = length * 0.35
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_height)
    head = trimesh.creation.cone(radius=head_radius, height=head_height)
    shaft.apply_translation([0, 0, shaft_height * 0.5])
    head.apply_translation([0, 0, shaft_height + head_height * 0.3])
    arrow = trimesh.util.concatenate([shaft, head])
    arrow.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))
    arrow.apply_translation(start)
    arrow.visual.vertex_colors = np.tile(color_rgba, (arrow.vertices.shape[0], 1))
    return arrow


def get_mesh(robot_module: RobotModule, state):
    robot_module.update_cfg(state)
    scene = robot_module.scene
    out = {}
    for link_name, mesh in scene.geometry.items():
        transform = scene.graph.get(link_name)[0]
        link_mesh = mesh.copy()
        link_mesh.apply_transform(transform)
        out[link_name] = link_mesh
    return out


def _norm_link_token(name: str) -> str:
    token = str(name).strip().lower()
    token = token.split("/")[-1]
    token = token.replace(".stl", "").replace(".obj", "")
    return token


def resolve_mesh_key(meshes: Dict[str, trimesh.Trimesh], expected_link: str) -> Optional[str]:
    if expected_link in meshes:
        return expected_link
    norm_to_key = {_norm_link_token(k): k for k in meshes.keys()}
    candidates = [expected_link]
    if "left_" in expected_link:
        candidates.append(expected_link.replace("left_", "right_"))
    if "right_" in expected_link:
        candidates.append(expected_link.replace("right_", "left_"))
    for c in candidates:
        n = _norm_link_token(c)
        if n in norm_to_key:
            return norm_to_key[n]
    return None


def build_contact_colors(distances: np.ndarray, clip_distance: float) -> np.ndarray:
    clip_distance = max(float(clip_distance), 1e-6)
    normalized = np.clip(distances / clip_distance, 0.0, 1.0)

    x = 1.0 - normalized
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=1) * 255.0

    alpha = np.full((distances.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb.astype(np.uint8), alpha], axis=1)


def resolve_urdf_path(arm: str, hand: str) -> str:
    if hand == "inspire_f1":
        urdf_path = os.path.join(rsc_path, "robot", f"{arm}_{hand}_right.urdf")
    elif hand == "allegro":
        urdf_path = os.path.join(rsc_path, "robot", f"{arm}_{hand}.urdf")
    elif hand == "inspire":
        urdf_path = os.path.join(rsc_path, "robot", f"{arm}_{hand}_DFTP.urdf")
    else:
        raise ValueError(f"Unsupported hand type: {hand}")

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    return urdf_path


def convert_hand_action_to_qpos(hand: str, hand_action: np.ndarray) -> np.ndarray:
    if hand == "inspire":
        return inspire_action_to_qpos(hand_action)
    if hand == "inspire_f1":
        return inspire_f1_action_to_qpos_dof6(hand_action)
    if hand == "allegro":
        return hand_action
    raise ValueError(f"Unsupported hand type: {hand}")


def load_hand_action(base_path: str, hand: str) -> tuple[np.ndarray, np.ndarray]:
    hand_dir = os.path.join(base_path, "raw", "hand")
    if hand == "inspire":
        hand_action, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))
    elif hand == "inspire_f1":
        hand_action, hand_time = load_series(hand_dir, ("right_joint_states.npy",))
    elif hand == "allegro":
        hand_action, hand_time = load_series(hand_dir, ("position.npy",))
    else:
        raise ValueError(f"Unsupported hand type: {hand}")

    return np.asarray(hand_action), np.asarray(hand_time, dtype=float)


def load_aligned_robot_qpos(
    base_path: str, hand: str, arm_time_offset: float, frame_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_root = os.path.join(base_path, "raw")
    arm_dir = os.path.join(raw_root, "arm")
    timestamp_path = os.path.join(raw_root, "timestamps", "timestamp.npy")
    frame_id_path = os.path.join(raw_root, "timestamps", "frame_id.npy")

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    arm_qpos = np.asarray(arm_qpos, dtype=float)
    arm_time = np.asarray(arm_time, dtype=float) + arm_time_offset

    hand_action, hand_time = load_hand_action(base_path, hand)
    hand_time_path = os.path.join(base_path, "raw", "hand", "time.npy")
    if not os.path.exists(hand_time_path):
        if len(arm_time) > 1:
            hand_time = np.linspace(arm_time[0], arm_time[-1], hand_action.shape[0], dtype=float)
        else:
            hand_time = np.arange(hand_action.shape[0], dtype=float)
    hand_qpos = convert_hand_action_to_qpos(hand, hand_action)
    hand_qpos = np.asarray(resample_to(hand_time, hand_qpos, arm_time), dtype=float)

    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

    if os.path.exists(timestamp_path) and os.path.exists(frame_id_path):
        video_times = np.asarray(np.load(timestamp_path), dtype=float)
        video_frame_ids = np.asarray(np.load(frame_id_path), dtype=int)
        qpos_video = np.asarray(resample_to(arm_time, full_qpos, video_times), dtype=float)
    else:
        video_times = arm_time
        video_frame_ids = np.arange(1, full_qpos.shape[0] + 1, dtype=int)
        qpos_video = full_qpos

    if frame_offset != 0:
        shifted = np.zeros_like(qpos_video)
        max_idx = len(qpos_video) - 1
        for i in range(len(qpos_video)):
            src = min(max(i + frame_offset, 0), max_idx)
            shifted[i] = qpos_video[src]
        qpos_video = shifted

    return qpos_video, video_frame_ids, video_times


def get_contact_qpos(qpos_video: np.ndarray, video_frame_ids: np.ndarray, contact_frame: int) -> tuple[np.ndarray, int]:
    matches = np.flatnonzero(video_frame_ids == contact_frame)
    if matches.size > 0:
        video_idx = int(matches[0])
    else:
        video_idx = int(np.clip(contact_frame - 1, 0, len(qpos_video) - 1))
    return qpos_video[video_idx], video_idx


def get_video_index(video_frame_ids: np.ndarray, frame_id: int, seq_len: int) -> int:
    matches = np.flatnonzero(video_frame_ids == frame_id)
    if matches.size > 0:
        return int(matches[0])
    return int(np.clip(frame_id - 1, 0, seq_len - 1))


def get_all_robot_link_names(robot_module: RobotModule) -> list[str]:
    return [link_name for link_name in robot_module.scene.geometry.keys() if link_name != "world"]


def get_non_arm_link_names(robot_module: RobotModule) -> list[str]:
    return [link_name for link_name in get_all_robot_link_names(robot_module) if link_name not in ARM_LINK_NAMES]


def build_robot_hand_mesh(robot_module: RobotModule, state: np.ndarray, hand_link_names: list[str]) -> trimesh.Trimesh:
    robot_module.update_cfg(state)
    scene = robot_module.scene
    meshes = []
    for link_name in hand_link_names:
        
        mesh = scene.geometry.get(link_name)
        if mesh is None:
            continue
        transform = scene.graph.get(link_name)[0]
        link_mesh = mesh.copy()
        link_mesh.apply_transform(transform)
        meshes.append(link_mesh)

    if not meshes:
        raise ValueError("No robot hand meshes found in URDF scene")
    return trimesh.util.concatenate(meshes)


def load_object_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh.copy()
    if isinstance(mesh, list):
        return trimesh.util.concatenate(mesh)
    return trimesh.util.concatenate(tuple(mesh.geometry.values()))


def load_object_world_trajectory_npz(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Object trajectory file not found: {npz_path}")

    payload = np.load(npz_path, allow_pickle=True)
    arr = None
    if isinstance(payload, np.lib.npyio.NpzFile):
        if payload.files:
            per_frame = []
            for key in payload.files:
                value = np.asarray(payload[key])
                if value.shape == (4, 4):
                    match = re.search(r"frame_(\d+)", key)
                    frame_idx = int(match.group(1)) if match else None
                    per_frame.append((frame_idx, key, value))
                else:
                    per_frame = []
                    break

            if per_frame:
                if all(idx is not None for idx, _, _ in per_frame):
                    per_frame.sort(key=lambda x: x[0])
                else:
                    per_frame.sort(key=lambda x: x[1])
                arr = np.stack([value for _, _, value in per_frame], axis=0)

        for key in ("obj_T_frames", "T", "poses", "trajectory", "traj", "arr_0"):
            if arr is None and key in payload.files:
                arr = payload[key]
                break
    else:
        arr = payload

    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape == (4, 4):
        arr = arr[None, ...]
    elif arr.ndim == 2 and arr.shape[1] == 16:
        arr = arr.reshape(arr.shape[0], 4, 4)
    elif arr.ndim == 3 and arr.shape[1:] == (3, 4):
        padded = np.tile(np.eye(4, dtype=float), (arr.shape[0], 1, 1))
        padded[:, :3, :] = arr
        arr = padded

    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"Unsupported object trajectory shape from {npz_path}: {arr.shape}")
    return arr.astype(float)


def set_mesh_alpha(mesh: trimesh.Trimesh, alpha: float) -> None:
    alpha_u8 = int(round(float(np.clip(alpha, 0.0, 1.0)) * 255.0))
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is not None:
        arr = np.asarray(colors).copy()
        if arr.ndim == 2 and arr.shape[1] >= 4:
            arr[:, 3] = alpha_u8
            mesh.visual.vertex_colors = arr


def load_object_pose_txt(txt_path: str) -> np.ndarray:
    arr = np.loadtxt(txt_path, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"Unsupported object pose shape from {txt_path}: {arr.shape}")
    return arr


def resolve_capture_root(capture_root: Optional[str], hand: str, object_name: str, ep: int) -> str:
    if capture_root is None:
        return os.path.join(os.path.expanduser("~"), "shared_data", "capture", "eccv2026", hand, object_name, str(ep))
    return os.path.join(capture_root, object_name, str(ep))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robot-hand/object contact viewer with visualize_all.py-style inputs"
    )
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument(
        "--hand",
        type=str,
        default="inspire_f1",
        choices=("inspire", "inspire_f1", "allegro"),
    )
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--capture-root", default=None)
    parser.add_argument("--object-mesh", type=str, default=None, help="Mesh file for the object.")
    parser.add_argument("--visualize-object", action="store_true")
    parser.add_argument("--fix-object-position", action="store_true")
    parser.add_argument("--object-pos-path", default=None)
    parser.add_argument("--transparent-robot", action="store_true")
    parser.add_argument("--object-alpha", type=float, default=1.0, help="Object mesh opacity [0,1].")
    parser.add_argument("--contact-frame", type=int, required=True, help="Camera frame index to analyze")
    parser.add_argument(
        "--tactile-frame",
        type=int,
        default=None,
        help="Camera frame index used only for tactile visualization (defaults to --contact-frame).",
    )
    parser.add_argument(
        "--hand_to_object_offset",
        type=int,
        default=1,
        help="object_frame_idx = contact_frame - offset",
    )
    parser.add_argument(
        "--distance_clip",
        type=float,
        default=0.02,
        help="Distance in meters mapped to the coldest color",
    )
    parser.add_argument(
        "--compute-contact",
        action="store_true",
        help="Compute hand/object contact distances and apply heatmap coloring",
    )
    parser.add_argument(
        "--arm_time_offset",
        type=float,
        default=0.09,
        help="Offset added to arm timestamps before frame alignment",
    )
    parser.add_argument(
        "--frame_offset",
        type=int,
        default=0,
        help="Positive => robot leads object; negative => robot lags object.",
    )
    parser.add_argument(
        "--show_hand",
        action="store_true",
        help="Show the contact-colored hand mesh",
    )
    parser.add_argument(
        "--show_object",
        action="store_true",
        help="Show the contact-colored object mesh",
    )
    parser.add_argument(
        "--show_actual_pose",
        action="store_true",
        help="Render the color-coded hand/object at their actual frame poses instead of canonical space",
    )
    parser.add_argument("--visualize-tactile", action="store_true", help="Visualize tactile force as arrows.")
    parser.add_argument("--max-normal-force", type=float, default=50.0)
    parser.add_argument("--max-tangential-force", type=float, default=50.0)
    parser.add_argument("--max-arrow-len", type=float, default=0.2)
    parser.add_argument("--show-cameras", type=str2bool, default=False, help="Show camera frustums in the same scene.")
    parser.add_argument("--camera-ids", type=str, default=None, help="Comma-separated camera IDs to visualize.")
    parser.add_argument("--camera-frustum-size", type=float, default=0.08, help="Camera frustum depth/size.")
    parser.add_argument("--show-camera-axes", type=str2bool, default=True, help="Show camera axes.")
    parser.add_argument("--show-camera-labels", type=str2bool, default=False, help="Show camera serial labels.")
    parser.add_argument(
        "--camera-image-root",
        type=str,
        default=None,
        help="Root directory with per-camera frames, e.g., <capture_root>/video_extracted/<cam_id>/00001.jpg",
    )
    parser.add_argument("--camera-view-scale", type=float, default=1.0, help="Scale factor for camera-view images.")
    parser.add_argument("--camera-fov-scale", type=float, default=1.0, help="Scale factor applied to camera FOV.")
    parser.add_argument(
        "--viewer-view-path",
        type=str,
        default=None,
        help="Path to a saved viser view JSON. If set, load this on client connect.",
    )
    parser.add_argument(
        "--use-attached-container-view",
        action="store_true",
        help="Load .viser_views/attached_container_view.json in this workspace.",
    )
    args = parser.parse_args()

    from paradex.visualization.visualizer.viser import ViserViewer

    if not args.show_hand and not args.show_object:
        args.show_hand = True
        args.show_object = True

    object_name = args.object
    capture_root = resolve_capture_root(args.capture_root, args.hand, object_name, args.ep)
    c2r_path = os.path.join(capture_root, "C2R.npy")
    object_pos_path = os.path.join(capture_root, "single_frame_refine_output", "refined_pose_world.txt")
    if args.object_mesh is None:
        object_mesh_path = os.path.join(shared_dir, "mesh", object_name, f"{object_name}.obj")
    else:
        object_mesh_path = args.object_mesh

    if not os.path.exists(object_mesh_path):
        raise FileNotFoundError(f"object mesh not found: {object_mesh_path}")
    # if not os.path.isdir(object_tracking_dir):
    #     raise FileNotFoundError(f"object tracking folder not found: {object_tracking_dir}")
    if not os.path.exists(c2r_path):
        raise FileNotFoundError(f"C2R file not found: {c2r_path}")

    urdf_path = resolve_urdf_path(args.arm, args.hand)
    robot_module = RobotModule(urdf_path)
    all_robot_link_names = get_all_robot_link_names(robot_module)
    non_arm_link_names = get_non_arm_link_names(robot_module)
    print("non arm link names", non_arm_link_names)

    qpos_video, video_frame_ids, video_times = load_aligned_robot_qpos(
        capture_root,
        args.hand,
        args.arm_time_offset,
        args.frame_offset,
    )
    contact_qpos, video_idx = get_contact_qpos(qpos_video, video_frame_ids, args.contact_frame)
    if len(contact_qpos) < robot_module.get_num_joints():
        raise ValueError(
            f"Aligned qpos length {len(contact_qpos)} is smaller than URDF joint count {robot_module.get_num_joints()}"
        )

    c2r = np.asarray(np.load(c2r_path), dtype=np.float64)
    if c2r.shape != (4, 4):
        raise ValueError(f"C2R must be 4x4, got {c2r.shape}")
    r2c = np.linalg.inv(c2r)

    fixed_object_pose = r2c @ load_object_pose_txt(object_pos_path)
    obj_traj = np.tile(fixed_object_pose[None, :, :], (len(video_times), 1, 1))
    object_frame_idx = -1 if args.fix_object_position else args.contact_frame - args.hand_to_object_offset

    object_transform_cam = obj_traj[video_idx]
    object_mesh = load_object_mesh(object_mesh_path)

    contact_state = contact_qpos[: robot_module.get_num_joints()].copy()
    robot_mesh_for_contact = build_robot_hand_mesh(robot_module, contact_state, non_arm_link_names)
    tactile_force_frame = None
    tactile_legacy_frame = None
    tactile_frame = args.contact_frame if args.tactile_frame is None else int(args.tactile_frame)
    tactile_video_idx = get_video_index(video_frame_ids, tactile_frame, len(video_times))
    tactile_display_state_for_check = (
        np.zeros((robot_module.get_num_joints(),), dtype=float) if args.show_hand else contact_state
    )
    if args.visualize_tactile:
        hand_dir = os.path.join(capture_root, "raw", "hand")
        tactile_candidates = [
            os.path.join(hand_dir, "right_tactile.npy"),
            os.path.join(hand_dir, "tactile.npy"),
        ]
        tactile_path = next((p for p in tactile_candidates if os.path.exists(p)), None)
        if tactile_path is None:
            raise FileNotFoundError(f"Tactile file not found. Tried: {tactile_candidates}")
        tactile_payload = np.load(tactile_path, allow_pickle=True)

        if args.hand == "inspire_f1":
            tactile_force_seq = normalize_force_dict_sequence(tactile_payload)
            tactile_force_i = resample_force_dict_sequence(tactile_force_seq, len(video_times))
            tactile_force_frame = tactile_force_i[tactile_video_idx]
            print("[INFO] tactile raw dict values at contact frame:")
            for key in sorted(tactile_force_frame.keys()):
                print(f"  - {key}: {float(tactile_force_frame[key]):.6f}")
            tactile_lengths = {}
            for zone in ZONE_TO_LINK.keys():
                normal_force, tangential_force, _ = _extract_zone_force(tactile_force_frame, zone)
                normal_len = np.clip(normal_force / args.max_normal_force, 0.0, 1.0) * args.max_arrow_len
                tangential_len = np.clip(tangential_force / args.max_tangential_force, 0.0, 1.0) * args.max_arrow_len
                tactile_lengths[zone] = float(np.sqrt(normal_len ** 2 + tangential_len ** 2))
            print("[INFO] tactile arrow length by region (m) [precompute]:")
            for name in sorted(tactile_lengths.keys()):
                print(f"  - {name}: {tactile_lengths[name]:.6f}")
            required_links = set(ZONE_TO_LINK.values())
            local_meshes = load_local_link_meshes_from_urdf(urdf_path, required_links=required_links)
            sensor_frames = _build_sensor_frames_from_local_mesh(local_meshes, ZONE_TO_LINK)
            predicted = sum(1 for z in sensor_frames.keys() if tactile_lengths.get(z, 0.0) > 1e-6)
            if predicted > 0:
                print(f"[INFO] tactile precheck: matching ok, arrows likely visible ({predicted} zones).")
            else:
                print("[WARN] tactile precheck: matching/force weak, arrows may not be visible.")
        elif args.hand == "inspire":
            hand_action, hand_time = load_hand_action(capture_root, args.hand)
            hand_time_path = os.path.join(hand_dir, "time.npy")
            if not os.path.exists(hand_time_path):
                arm_dir = os.path.join(capture_root, "raw", "arm")
                _, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
                arm_time = np.asarray(arm_time, dtype=float) + args.arm_time_offset
                if len(arm_time) > 1:
                    hand_time = np.linspace(arm_time[0], arm_time[-1], hand_action.shape[0], dtype=float)
                else:
                    hand_time = np.arange(hand_action.shape[0], dtype=float)
            tactile_seq = normalize_tactile_sequence(tactile_payload, TACTILE_LAYOUT)
            n_tactile = min(len(hand_time), tactile_seq.shape[0])
            tactile_i = resample_to(
                np.asarray(hand_time[:n_tactile], dtype=float),
                np.asarray(tactile_seq[:n_tactile], dtype=float),
                video_times,
            )
            tactile_legacy_frame = unpack_tactile_frame(
                tactile_i[tactile_video_idx], build_tactile_index_from_layout(TACTILE_LAYOUT)
            )
            tactile_lengths = {}
            for name in TACTILE_VERTEX_MAP.keys():
                if name not in tactile_legacy_frame:
                    continue
                p = tactile_legacy_frame[name].mean()
                tactile_lengths[name] = float(np.clip(p / 1000.0, 0, 1) * 0.2)
            print("[INFO] tactile arrow length by region (m) [precompute]:")
            for name in sorted(tactile_lengths.keys()):
                print(f"  - {name}: {tactile_lengths[name]:.6f}")
            tactile_robot_check = RobotModule(urdf_path)
            meshes_check = get_mesh(tactile_robot_check, tactile_display_state_for_check)
            matched_nonzero = 0
            unmatched_links = 0
            bad_vids = 0
            for name, (link, vids) in TACTILE_VERTEX_MAP.items():
                if tactile_lengths.get(name, 0.0) <= 1e-6:
                    continue
                candidates = INSPIRE_TACTILE_LINK_CANDIDATES.get(name, [link])
                mesh_key = None
                for cand in candidates:
                    mesh_key = resolve_mesh_key(meshes_check, cand)
                    if mesh_key is not None:
                        break
                if mesh_key is None:
                    unmatched_links += 1
                    continue
                if compute_contact_arrow_safe(meshes_check[mesh_key], vids) is None:
                    if compute_contact_arrow_fallback(meshes_check[mesh_key]) is None:
                        bad_vids += 1
                        continue
                matched_nonzero += 1
            if matched_nonzero > 0:
                print(
                    f"[INFO] tactile precheck: matching ok, arrows likely visible "
                    f"({matched_nonzero} regions; unmatched={unmatched_links}, bad_vids={bad_vids})."
                )
            else:
                print(
                    f"[WARN] tactile precheck: arrows unlikely (matched_nonzero=0, "
                    f"unmatched={unmatched_links}, bad_vids={bad_vids})."
                )
        else:
            print(f"[WARN] tactile visualization is not supported for hand={args.hand}")


    if args.show_hand:
        # Keep hand mesh and tactile arrows in the same canonical pose.
        display_state = np.zeros((robot_module.get_num_joints(),), dtype=float)
        tactile_display_state = display_state
    else:
        display_state = contact_state
        tactile_display_state = contact_state
    hand_mesh_to_show = build_robot_hand_mesh(robot_module, display_state, non_arm_link_names)
    object_mesh_to_show = object_mesh.copy()
    if args.compute_contact:
        posed_object_vertices = transform_points(np.asarray(object_mesh.vertices), object_transform_cam)
        posed_robot_vertices = np.asarray(robot_mesh_for_contact.vertices, dtype=np.float32)

        robot_distances = compute_min_vertex_distances(posed_robot_vertices, posed_object_vertices)
        object_distances = compute_min_vertex_distances(posed_object_vertices, posed_robot_vertices)

        hand_mesh_to_show.visual = trimesh.visual.ColorVisuals(
            mesh=hand_mesh_to_show,
            vertex_colors=build_contact_colors(robot_distances, args.distance_clip),
        )
        object_mesh_to_show.visual = trimesh.visual.ColorVisuals(
            mesh=object_mesh_to_show,
            vertex_colors=build_contact_colors(object_distances, args.distance_clip),
        )
    set_mesh_alpha(object_mesh_to_show, args.object_alpha)
    object_display_transform = object_transform_cam if args.show_actual_pose else np.eye(4, dtype=float)

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    if args.use_attached_container_view:
        vis.view_state_path = Path(".viser_views") / "attached_container_view.json"
    elif args.viewer_view_path is not None:
        vis.view_state_path = Path(args.viewer_view_path)
    if not vis.view_state_path.exists():
        print(f"[WARN] saved view not found: {vis.view_state_path}")
    server = vis.server
    server.scene.world_axes.visible = False
    if args.show_hand:
        server.scene.add_mesh_trimesh("/robot_hand", hand_mesh_to_show)
    if args.visualize_tactile:
        tactile_robot = RobotModule(urdf_path)
        tactile_robot.update_cfg(tactile_display_state)
        added_tactile_arrows = 0
        if tactile_force_frame is not None:
            required_links = set(ZONE_TO_LINK.values())
            local_meshes = load_local_link_meshes_from_urdf(urdf_path, required_links=required_links)
            sensor_frames = _build_sensor_frames_from_local_mesh(local_meshes, ZONE_TO_LINK)
            print(f"[INFO] tactile sensor frames detected: {len(sensor_frames)}")
            for zone, sensor in sensor_frames.items():
                try:
                    link_pose = tactile_robot.get_transform(
                        sensor.link_name, tactile_robot.urdf.base_link, collision_geometry=False
                    )
                except Exception:
                    continue
                anchor, normal, tx, ty = _world_sensor_frame(sensor, link_pose)
                vis_normal = -normal
                normal_force, tangential_force, tangential_deg = _extract_zone_force(tactile_force_frame, zone)
                normal_len = np.clip(normal_force / args.max_normal_force, 0.0, 1.0) * args.max_arrow_len
                normal_vec = vis_normal * normal_len
                tangential_vec = np.zeros(3, dtype=np.float64)
                if tangential_deg >= 0.0 and tangential_force > 0.0:
                    theta = np.deg2rad(tangential_deg)
                    tangential_dir = np.cos(theta) * tx + np.sin(theta) * ty
                    tangential_dir = _safe_normalize(tangential_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))
                    tangential_len = np.clip(tangential_force / args.max_tangential_force, 0.0, 1.0) * args.max_arrow_len
                    tangential_vec = tangential_dir * tangential_len

                total_vec = normal_vec + tangential_vec
                length = float(np.linalg.norm(total_vec))
                if length <= 1e-6:
                    continue
                direction = total_vec / (length + 1e-12)
                arrow = make_arrow_mesh(anchor, direction, length, TACTILE_ARROW_RGBA)
                if arrow is not None:
                    server.scene.add_mesh_trimesh(f"/contact/{zone}", arrow)
                    added_tactile_arrows += 1
        elif tactile_legacy_frame is not None:
            meshes = get_mesh(tactile_robot, tactile_display_state)
            for name, (link, vids) in TACTILE_VERTEX_MAP.items():
                if name not in tactile_legacy_frame:
                    continue
                candidates = INSPIRE_TACTILE_LINK_CANDIDATES.get(name, [link])
                mesh_key = None
                for cand in candidates:
                    mesh_key = resolve_mesh_key(meshes, cand)
                    if mesh_key is not None:
                        break
                if mesh_key is None:
                    continue
                p = tactile_legacy_frame[name].mean()
                length = np.clip(p / 1000.0, 0, 1) * 0.2
                arrow_cn = compute_contact_arrow_safe(meshes[mesh_key], vids)
                if arrow_cn is None:
                    arrow_cn = compute_contact_arrow_fallback(meshes[mesh_key])
                if arrow_cn is None:
                    continue
                c, n = arrow_cn
                arrow = make_arrow_mesh(c, n, length, TACTILE_ARROW_RGBA)
                if arrow is not None:
                    server.scene.add_mesh_trimesh(f"/contact/{name}", arrow)
                    added_tactile_arrows += 1
        if added_tactile_arrows == 0:
            print("[WARN] tactile enabled but no contact arrows were added (scene tree '/contact' will be absent).")
        else:
            print(f"[INFO] tactile arrows added: {added_tactile_arrows}")
    if args.show_object or args.visualize_object:
        vis.add_object(object_name, object_mesh_to_show, object_display_transform, opacity=args.object_alpha)
        if object_name in vis.obj_dict:
            vis.obj_dict[object_name]["frame"].show_axes = False
    if args.show_cameras:
        selected_camera_ids = parse_camera_ids(args.camera_ids)
        camera_image_root = (
            args.camera_image_root
            if args.camera_image_root is not None
            else os.path.join(capture_root, "video_extracted")
        )
        n_added, n_skipped = add_cameras_to_scene(
            vis,
            capture_root=capture_root,
            c2r=c2r,
            frustum_size=args.camera_frustum_size,
            show_axes=args.show_camera_axes,
            show_labels=args.show_camera_labels,
            image_root=camera_image_root,
            initial_frame_id=int(args.contact_frame),
            view_scale=float(args.camera_view_scale),
            fov_scale=float(args.camera_fov_scale),
            selected_ids=selected_camera_ids,
        )
        print(f"[INFO] camera visualization added={n_added}, skipped={n_skipped}")

    with server.gui.add_folder("Contact"):
        server.gui.add_text("Object", initial_value=object_name)
        server.gui.add_text("Episode", initial_value=str(args.ep))
        server.gui.add_text("Contact Frame", initial_value=str(args.contact_frame))
        server.gui.add_text("Tactile Frame", initial_value=str(tactile_frame))
        server.gui.add_text("Object Frame", initial_value=str(object_frame_idx))
        server.gui.add_text("Video Index", initial_value=str(video_idx))
        server.gui.add_text("Tactile Video Index", initial_value=str(tactile_video_idx))
        server.gui.add_text("Distance Clip (m)", initial_value=f"{args.distance_clip:.4f}")
        server.gui.add_text(
            "Color Meaning",
            initial_value="Hot = near contact, Cold = farther" if args.compute_contact else "Original mesh colors",
        )
        server.gui.add_text("Compute Contact", initial_value=str(args.compute_contact))
        server.gui.add_text(
            "Display Space",
            initial_value="Actual pose" if args.show_actual_pose else "Canonical (no pose transform)",
        )
        server.gui.add_text(
            "Robot Hand Pose",
            initial_value="Display/Tactile: zero qpos, Contact color: aligned qpos"
            if args.show_hand
            else "Display/Tactile/Contact: aligned qpos",
        )

    print(f"[INFO] capture root: {capture_root}")
    print(f"[INFO] contact frame: {args.contact_frame}")
    print(f"[INFO] tactile frame: {tactile_frame}")
    print(f"[INFO] object frame: {object_frame_idx}")
    print(f"[INFO] video index used: {video_idx}")
    print(f"[INFO] tactile video index used: {tactile_video_idx}")
    print(f"[INFO] URDF: {urdf_path}")
    print(f"[INFO] object mesh: {object_mesh_path}")
    print(f"[INFO] all robot links: {', '.join(all_robot_link_names)}")
    print(f"[INFO] non-arm links (used for contact/render): {', '.join(non_arm_link_names)}")
    print(f"[INFO] using visualize_all.py loading/alignment for robot qpos and object pose")
    print(f"[INFO] compute_contact={args.compute_contact}")
    print(
        f"[INFO] hand qpos mode: {'display+tactile=zero, contact=aligned' if args.show_hand else 'display+tactile+contact=aligned'}"
    )
    print(f"[INFO] object pose uses r2c @ pose_world, hand mesh stays robot-space, camera uses c2r")
    print(f"[INFO] show_actual_pose={args.show_actual_pose}")
    print(f"[INFO] show_hand={args.show_hand} show_object={args.show_object or args.visualize_object}")
    print(f"[INFO] show_cameras={args.show_cameras}")
    print(f"[INFO] visualize_tactile={args.visualize_tactile}")

    vis.start_viewer()


if __name__ == "__main__":
    main()
