import argparse
import copy
import os
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import trimesh
import yourdfpy

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.visualization.robot import RobotModule
from paradex.utils.load_data import load_series, resample_to
from paradex.robot.inspire import inspire_f1_action_to_qpos_dof6, inspire_action_to_qpos
from paradex.robot.utils import get_robot_urdf_path

# yourdfpy can emit per-frame mimic-chain warnings on RH56F1 URDF
# (thumb_4 -> thumb_3 -> thumb_2). The stdout spam itself becomes a bottleneck.
logging.getLogger("yourdfpy.urdf").setLevel(logging.ERROR)

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "1", "y", "on"):
        return True
    if v in ("no", "false", "f", "0", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{v}'")


SOURCE_QPOS_ORDER = [
    "openarm_right_joint1",
    "openarm_right_joint4",
    "openarm_left_joint6",
    "openarm_left_joint5",
    "openarm_left_joint2",
    "openarm_left_joint7",
    "openarm_left_joint4",
    "openarm_right_joint5",
    "openarm_left_joint1",
    "openarm_right_joint6",
    "openarm_right_joint3",
    "openarm_left_joint3",
    "openarm_right_joint7",
    "openarm_right_joint2",
]

TARGET_QPOS_ORDER = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

# /openarm/joint_states(16): [head_pitch, head_yaw, r1, r4, l6, l5, l2, l7, l4, r5, l1, r6, r3, l3, r7, r2]
JOINT_STATES16_TO_SOURCE14 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

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


def _zone_to_link_for_side(side: str) -> Dict[str, str]:
    if side not in ("left", "right"):
        return dict(ZONE_TO_LINK)
    prefix = f"{side}_hand_"
    return {k: prefix + v for k, v in ZONE_TO_LINK.items()}


def expand_openarm_inspire_qpos28_to_urdf40(q28: np.ndarray) -> np.ndarray:
    q28 = np.asarray(q28)
    if q28.ndim != 2 or q28.shape[1] != 28:
        raise ValueError(f"Expected [T,28] qpos, got {q28.shape}")

    T = q28.shape[0]
    q40 = np.zeros((T, 40), dtype=q28.dtype)

    # head + arm
    q40[:, :16] = q28[:, :16]

    # left hand dof6: [thumb1, thumb2, index1, middle1, ring1, little1]
    l = q28[:, 16:22]
    l_thumb3 = 1.2953 * l[:, 1]
    l_thumb4 = 0.8962 * l_thumb3
    l_index2 = 1.1545 * l[:, 2]
    l_middle2 = 1.1545 * l[:, 3]
    l_ring2 = 1.1545 * l[:, 4]
    l_little2 = 1.1545 * l[:, 5]
    q40[:, 16:28] = np.stack(
        [
            l[:, 0],
            l[:, 1],
            l_thumb3,
            l_thumb4,
            l[:, 2],
            l_index2,
            l[:, 3],
            l_middle2,
            l[:, 4],
            l_ring2,
            l[:, 5],
            l_little2,
        ],
        axis=1,
    )

    # right hand dof6
    r = q28[:, 22:28]
    r_thumb3 = 1.2953 * r[:, 1]
    r_thumb4 = 0.8962 * r_thumb3
    r_index2 = 1.1545 * r[:, 2]
    r_middle2 = 1.1545 * r[:, 3]
    r_ring2 = 1.1545 * r[:, 4]
    r_little2 = 1.1545 * r[:, 5]
    q40[:, 28:40] = np.stack(
        [
            r[:, 0],
            r[:, 1],
            r_thumb3,
            r_thumb4,
            r[:, 2],
            r_index2,
            r[:, 3],
            r_middle2,
            r[:, 4],
            r_ring2,
            r[:, 5],
            r_little2,
        ],
        axis=1,
    )
    return q40







def interpolate_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == target_len:
        return seq
    x_src = np.linspace(0, len(seq) - 1, len(seq))
    x_tgt = np.linspace(0, len(seq) - 1, target_len)
    flat = seq.reshape(len(seq), -1)
    out = np.stack(
        [np.interp(x_tgt, x_src, flat[:, i]) for i in range(flat.shape[1])],
        axis=1
    )
    return out.reshape((target_len,) + seq.shape[1:])


def to_numpy_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return np.asarray(x.detach())
    return np.asarray(x)


def _maybe_stack_array(payload: Any) -> Optional[np.ndarray]:
    if isinstance(payload, (list, tuple)):
        return None
    arr = to_numpy_array(payload)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr.astype(float)
    return None


def _build_T_from_frame(frame: Any, idx: int) -> np.ndarray:
    if isinstance(frame, dict):
        for k in ("T", "pose", "world_T_obj", "T_world_obj", "T_object_world"):
            if k in frame:
                T = to_numpy_array(frame[k])
                break
        else:
            if "obj_R" in frame and "obj_t" in frame:
                R = to_numpy_array(frame["obj_R"])
                t = to_numpy_array(frame["obj_t"]).reshape(3)
                T = np.eye(4, dtype=float)
                T[:3, :3] = R
                T[:3, 3] = t
            else:
                raise ValueError(f"Frame {idx} missing T/obj_R/obj_t")
    else:
        T = to_numpy_array(frame)

    T = np.asarray(T)
    if T.shape == (4, 4):
        return T.astype(float)
    if T.ndim == 3 and T.shape[0] == 1 and T.shape[1:] == (4, 4):
        return T[0].astype(float)
    if T.size == 16:
        return T.reshape(4, 4).astype(float)
    if T.shape == (3, 4):  # pad last row
        padded = np.eye(4, dtype=float)
        padded[:3, :] = T
        return padded
    raise ValueError(f"Frame {idx} transform shape {T.shape} cannot be interpreted as 4x4")


def reorder_openarm_qpos_to_target_order(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2:
        raise ValueError(f"Expected 2D qpos array, got shape {qpos.shape}")
    if qpos.shape[1] == 16:
        head_qpos = qpos[:, :2]
        qpos_source = qpos[:, JOINT_STATES16_TO_SOURCE14]
    elif qpos.shape[1] == 14:
        head_qpos = None
        qpos_source = qpos
    else:
        raise ValueError(
            f"Expected 14(SOURCE order) or 16(joint_states order) joints, got {qpos.shape[1]}"
        )
    source_index = {joint_name: idx for idx, joint_name in enumerate(SOURCE_QPOS_ORDER)}
    target_indices = [source_index[joint_name] for joint_name in TARGET_QPOS_ORDER]
    arm_qpos_target = qpos_source[:, target_indices]
    if head_qpos is None:
        return arm_qpos_target
    return np.concatenate([head_qpos, arm_qpos_target], axis=1)


def load_object_trajectory(track_dir: str) -> np.ndarray:
    # Load the first pickle/npz/npy in the tracking folder into an array of shape [T, 4, 4].
    import glob
    import pickle

    candidates = sorted(
        glob.glob(os.path.join(track_dir, "*.pickle"))
        + glob.glob(os.path.join(track_dir, "*.pkl"))
        + glob.glob(os.path.join(track_dir, "*.npy"))
        + glob.glob(os.path.join(track_dir, "*.npz"))
    )
    if not candidates:
        raise FileNotFoundError(f"No object trajectory found in {track_dir}")

    path = candidates[0]
    if path.endswith((".npy", ".npz")):
        payload = np.load(path, allow_pickle=True)
    else:
        with open(path, "rb") as f:
            payload = pickle.load(f)

    stacked = _maybe_stack_array(payload)
    if stacked is not None:
        return stacked

    if isinstance(payload, dict):
        for key in ("T", "poses", "traj", "trajectory"):
            if key in payload:
                stacked = _maybe_stack_array(payload[key])
                if stacked is not None:
                    return stacked
        frames = payload.get("frames", payload)
        if isinstance(frames, dict):
            frames = [frames[k] for k in sorted(frames.keys())]
    else:
        frames = payload

    if isinstance(frames, np.ndarray):
        arr = np.asarray(frames)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(float)

    T_list = []
    for idx, frame in enumerate(frames):
        T_list.append(_build_T_from_frame(frame, idx))

    if not T_list:
        raise ValueError("No transforms loaded from object trajectory")

    return np.stack(T_list, axis=0)


def load_object_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, list):
        return trimesh.util.concatenate(mesh)
    raise ValueError(f"Unexpected mesh type: {type(mesh)}")


TACTILE_VERTEX_MAP = {
    "little_tip":    ("left_little_2", [30136, 32377, 10140, 21072]),
    "little_nail":   ("left_little_2", [19218, 8443, 26413, 15619]),
    "little_pad":    ("left_little_1", [53841, 15868, 16563, 51124]),

    "ring_tip":      ("left_ring_2",   [15360, 29482, 1307, 6517]),
    "ring_nail":     ("left_ring_2",   [20197, 18901, 9956, 11658]),
    "ring_pad":      ("left_ring_1",   [41463, 53892, 19863, 39231]),

    "middle_tip":    ("left_middle_2", [16989, 36909, 28443, 20206]),
    "middle_nail":   ("left_middle_2", [24530, 32740, 34230, 26026]),
    "middle_pad":    ("left_middle_1", [36682, 18750, 19228, 38119]),

    "index_tip":     ("left_index_2",  [29503, 1307, 6517, 15171]),
    "index_nail":    ("left_index_2",  [20197, 12013, 4598, 13379]),
    "index_pad":     ("left_index_1",  [36688, 18750, 19219, 37862]),

    "thumb_tip":     ("left_thumb_4",  [9676, 23934, 26403, 28719]),
    "thumb_nail":    ("left_thumb_4",  [18621, 16421, 35555, 37778]),
    "thumb_middle":  ("left_thumb_2",  [15649, 22156, 14346, 5837]),
    "thumb_pad":     ("left_thumb_2",  [19300, 18427, 10008, 8949]),

    "palm":          ("base_link",     [68864, 61448, 68207, 67649]),
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
        # Handle stacked dict (values are [T, r, c]) or single-frame dict.
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
        if payload:
            first = _unwrap_tactile_item(payload[0])
        else:
            first = None
        if payload and isinstance(first, dict):
            frames = [_flatten_tactile_frame(_unwrap_tactile_item(frame), layout) for frame in payload]
            return np.stack(frames, axis=0)
        arr = np.asarray(payload)
        if arr.ndim == 2:
            return arr.astype(float)

    raise ValueError(f"Unsupported tactile payload type: {type(payload)}")


def normalize_force_dict_sequence(payload: Any) -> List[Dict[str, float]]:
    # Expected format from inspire_f1_state_receiver: object array of per-frame dicts.
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


def _legacy_zone_force(tactile_frame: Dict[str, np.ndarray], zone: str) -> Tuple[float, float, float]:
    group_map = {
        "little": ("little_tip", "little_nail", "little_pad"),
        "ring": ("ring_tip", "ring_nail", "ring_pad"),
        "middle": ("middle_tip", "middle_nail", "middle_pad"),
        "index": ("index_tip", "index_nail", "index_pad"),
        "thumb": ("thumb_tip", "thumb_nail", "thumb_middle", "thumb_pad"),
        "palm_right": ("palm",),
        "palm_middle": ("palm",),
        "palm_left": ("palm",),
    }
    vals = []
    for name in group_map.get(zone, ()):
        if name in tactile_frame:
            vals.append(float(np.asarray(tactile_frame[name]).mean()))
    if not vals:
        return 0.0, 0.0, -1.0
    # Legacy format has no tangential decomposition.
    return float(np.mean(vals)), 0.0, -1.0


def length_to_color(length, max_len=0.08):
    t = np.clip(length / max_len, 0.0, 1.0)
    if t < 0.25:
        r, g, b = 0, int(4*t*255), 255
    elif t < 0.5:
        r, g, b = 0, 255, int((1-4*(t-0.25))*255)
    elif t < 0.75:
        r, g, b = int(4*(t-0.5)*255), 255, 0
    else:
        r, g, b = 255, int((1-4*(t-0.75))*255), 0
    return np.array([r, g, b, 255], dtype=np.uint8)


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
    tangent_y = _safe_normalize(
        np.cross(normal, tangent_x), np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    tangent_x = _safe_normalize(
        np.cross(tangent_y, normal), np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    return anchor, normal, tangent_x, tangent_y


def _build_sensor_frames_from_local_mesh(
    local_meshes_by_link: Dict[str, trimesh.Trimesh],
    zone_to_link: Dict[str, str],
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
    R = link_pose[:3, :3]
    p = link_pose[:3, 3]
    anchor = R @ sensor.anchor_local + p
    normal = _safe_normalize(R @ sensor.normal_local, np.array([0.0, 0.0, 1.0]))
    tx = _safe_normalize(R @ sensor.tangent_x_local, np.array([1.0, 0.0, 0.0]))
    ty = _safe_normalize(R @ sensor.tangent_y_local, np.array([0.0, 1.0, 0.0]))
    return anchor, normal, tx, ty


def set_arm_mesh_visibility(vis: ViserViewer, robot_name: str, visible: bool) -> None:
    robot_handle = vis.robot_dict.get(robot_name)
    if robot_handle is None or not hasattr(robot_handle, "_meshes"):
        return
    for mesh_name, mesh_handle in robot_handle._meshes.items():
        is_hand_mesh = ("left_hand_" in mesh_name) or ("right_hand_" in mesh_name)
        if is_hand_mesh:
            continue
        if "openarm_" in mesh_name:
            mesh_handle.visible = visible


def compute_contact_arrow(tm, vids):
    v = tm.vertices[vids]
    n = tm.vertex_normals[vids]
    c = v.mean(axis=0)
    n = n.mean(axis=0)
    n /= np.linalg.norm(n) + 1e-8
    return c, n


def make_arrow_mesh(start, direction, length, color_rgba):
    if length < 1e-6:
        return None
    shaft_radius = 0.003
    head_radius = 0.004
    shaft_height = length * 0.7
    head_height = length * 0.35
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_height)
    head  = trimesh.creation.cone(radius=head_radius, height=head_height)
    shaft.apply_translation([0, 0, shaft_height * 0.5])
    head.apply_translation([0, 0, shaft_height + head_height * 0.5])
    arrow = trimesh.util.concatenate([shaft, head])
    arrow.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))
    arrow.apply_translation(start)
    arrow.visual.vertex_colors = np.tile(color_rgba, (arrow.vertices.shape[0], 1))
    return arrow


def get_mesh(robot_wrapper, robot_obj, state):
    robot_wrapper.compute_forward_kinematics(state)
    out = {}
    for ln, meshes in robot_obj.mesh_dict.items():
        if not meshes:
            continue
        T = robot_wrapper.get_link_pose(robot_wrapper.get_link_index(ln))
        merged = []
        for m in meshes:
            mm = copy.deepcopy(m)
            mm.transform(T)
            merged.append(
                trimesh.Trimesh(
                    vertices=np.asarray(mm.vertices),
                    faces=np.asarray(mm.triangles),
                    process=False,
                )
            )
        out[ln] = trimesh.util.concatenate(merged)
    return out


def build_link_color_map(urdf_path: str) -> Dict[str, np.ndarray]:
    urdf = yourdfpy.URDF.load(urdf_path)
    link_colors = {}
    for link in urdf.robot.links:
        rgba = None
        for vis in link.visuals:
            if vis.material is None:
                continue
            if vis.material.color is not None:
                rgba = vis.material.color
                break
            if vis.material.name in urdf._material_map:
                mat = urdf._material_map[vis.material.name]
                if mat.color is not None:
                    rgba = mat.color
                    break
        if rgba is not None:
            if hasattr(rgba, "rgba"):
                rgba_vals = rgba.rgba
            elif all(hasattr(rgba, k) for k in ("r", "g", "b")):
                rgba_vals = [rgba.r, rgba.g, rgba.b, getattr(rgba, "a", 1.0)]
            else:
                rgba_vals = rgba
            link_colors[link.name] = np.array(rgba_vals, dtype=float)
    return link_colors


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
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def load_local_link_meshes_from_urdf(
    urdf_path: str, required_links: Optional[set] = None
) -> Dict[str, trimesh.Trimesh]:
    root = ET.parse(urdf_path).getroot()
    base_dir = os.path.dirname(urdf_path)
    out: Dict[str, trimesh.Trimesh] = {}
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        if required_links is not None and link_name not in required_links:
            continue
        parts: List[trimesh.Trimesh] = []
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
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = _rpy_to_matrix(rpy)
                T[:3, 3] = xyz
                tm.apply_transform(T)
            parts.append(tm)
        if parts:
            out[link_name] = trimesh.util.concatenate(parts)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="openarm")
    parser.add_argument("--hand", type=str, default="inspire_f1")
    parser.add_argument("--capture-root", type=str, default="hri_openarm")
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--object-mesh", type=str, default=None, help="Mesh file for the object.")
    parser.add_argument("--visualize-object", action="store_true")
    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--visualize-arm", type=str2bool, default=True, help="Show/hide openarm meshes while keeping hand meshes visible.")
    parser.add_argument("--tactile-side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--max-normal-force", type=float, default=100.0)
    parser.add_argument("--max-tangential-force", type=float, default=100.0)
    parser.add_argument("--max-arrow-len", type=float, default=0.02)
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th frame for playback.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional hard cap on number of frames after stride. 0 disables cap.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (inclusive) after stride.")
    parser.add_argument("--end-frame", type=int, default=-1, help="End frame index (exclusive) after stride. -1 means until the end.")
    parser.add_argument("--frame-offset", type=int, default=0, help="Positive => robot leads object; negative => robot lags object.")
    args = parser.parse_args()
    object_name = args.object

    capture_root = os.path.join(shared_dir, "capture", args.capture_root, args.object, str(args.ep))
    data_root = os.path.join(capture_root, "raw")
    
    arm_dir = os.path.join(data_root, "arm")
    hand_dir = os.path.join(data_root, "hand")

    object_track_dir = os.path.join(capture_root, "object_tracking")

    timestamp_path = os.path.join(data_root, "timestamps", "timestamp.npy")
    frame_id_path = os.path.join(data_root, "timestamps", "frame_id.npy")


    c2r_path = os.path.join(capture_root, "C2R.npy")
    
    c2r = np.load(c2r_path)
    r2c = np.linalg.inv(c2r)

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    arm_qpos = reorder_openarm_qpos_to_target_order(arm_qpos)
    left_hand_action, left_hand_time = load_series(hand_dir, ("left_joint_states.npy", "left_commands.npy"))
    right_hand_action, right_hand_time = load_series(hand_dir, ("right_joint_states.npy", "right_commands.npy"))
    left_hand_action = resample_to(left_hand_time, left_hand_action, arm_time)
    right_hand_action = resample_to(right_hand_time, right_hand_action, arm_time)

    if args.hand == "inspire_f1":
        left_hand_qpos = inspire_f1_action_to_qpos_dof6(left_hand_action)
        right_hand_qpos = inspire_f1_action_to_qpos_dof6(right_hand_action)
    elif args.hand == "inspire":
        # Backward compatibility for older single-hand datasets.
        left_hand_qpos = inspire_action_to_qpos(left_hand_action)
        right_hand_qpos = inspire_action_to_qpos(right_hand_action)
    else:
        left_hand_qpos = left_hand_action
        right_hand_qpos = right_hand_action

    full_qpos = np.concatenate([arm_qpos, left_hand_qpos, right_hand_qpos], axis=1)

    tactile_seq = None
    tactile_force_seq = None
    tactile_index = None
    tactile_force_seq_by_side: Dict[str, List[Dict[str, float]]] = {}
    if args.visualize_tactile:
        left_tactile_path = os.path.join(hand_dir, "left_tactile.npy")
        right_tactile_path = os.path.join(hand_dir, "right_tactile.npy")
        tactile_candidates = [left_tactile_path, right_tactile_path]
        for side, tactile_path in (("left", left_tactile_path), ("right", right_tactile_path)):
            if not os.path.exists(tactile_path):
                continue
            tactile_payload = np.load(tactile_path, allow_pickle=True)
            try:
                tactile_force_seq_by_side[side] = normalize_force_dict_sequence(tactile_payload)
            except Exception:
                # Legacy fallback only supports a single tactile matrix stream.
                if tactile_seq is None:
                    tactile_seq = normalize_tactile_sequence(tactile_payload, TACTILE_LAYOUT)
                    tactile_index = build_tactile_index_from_layout(TACTILE_LAYOUT)
        if not tactile_force_seq_by_side and tactile_seq is None:
            raise FileNotFoundError(f"Tactile file not found. tried={tactile_candidates}")

    obj_traj = None
    if args.visualize_object:
        obj_traj = load_object_trajectory(object_track_dir)
        print(f"Loaded object trajectory with {obj_traj.shape[0]} frames.")

    # Build master timeline from camera timestamps (pc_time) using fill_framedrop logic.
    
    
    if os.path.exists(timestamp_path) and os.path.exists(frame_id_path):
        video_times = np.load(timestamp_path)
        video_frame_ids = np.load(frame_id_path)
        qpos_video = resample_to(arm_time, full_qpos, video_times)
    else:
        video_times = arm_time
        video_frame_ids = np.arange(1, full_qpos.shape[0] + 1, dtype=int)
        qpos_video = full_qpos
    if args.frame_offset != 0:
        shifted = np.zeros_like(qpos_video)
        max_idx = len(qpos_video) - 1
        for i in range(len(qpos_video)):
            src = min(max(i + args.frame_offset, 0), max_idx)
            shifted[i] = qpos_video[src]
        qpos_video = shifted

    stride = max(1, int(args.stride))
    if stride > 1:
        qpos_video = qpos_video[::stride]
        video_times = video_times[::stride]
        video_frame_ids = video_frame_ids[::stride]
    obj_mesh = None
    if obj_traj is not None:
        # Snap object trajectory onto the master timeline (assume uniform spacing across its original length).
        obj_time = np.linspace(video_times[0], video_times[-1], obj_traj.shape[0])
        obj_traj = resample_to(
            obj_time,
            obj_traj.reshape(obj_traj.shape[0], -1),
            video_times,
        ).reshape(len(video_times), 4, 4)
        obj_traj = np.einsum("ij,tjk->tik", r2c, obj_traj)
        object_mesh_path = args.object_mesh
        if object_mesh_path is None:
            object_mesh_path = os.path.join(shared_dir, "mesh", object_name, f"{object_name}.obj")
        obj_mesh = load_object_mesh(object_mesh_path)
        mat = getattr(obj_mesh.visual, "material", None)
        if mat is not None:
            factor = np.array(getattr(mat, "baseColorFactor", [1, 1, 1, 1]), dtype=float)
            factor[3] = 0.3  # set desired transparency
            mat.baseColorFactor = factor
            mat.alphaMode = "BLEND"  # keep texture, just blend alpha
            mat.doubleSided = True
    tactile_i = interpolate_sequence(tactile_seq, len(video_times)) if tactile_seq is not None else None
    tactile_force_i_by_side: Dict[str, List[Dict[str, float]]] = {}
    for side, tactile_force_seq in tactile_force_seq_by_side.items():
        tactile_force_i_by_side[side] = resample_force_dict_sequence(tactile_force_seq, len(video_times))

    total_frames = len(qpos_video)
    start = max(0, int(args.start_frame))
    end = total_frames if int(args.end_frame) < 0 else min(int(args.end_frame), total_frames)
    if end <= start:
        raise ValueError(
            f"Invalid frame range: start={start}, end={end}, total={total_frames}. "
            "Expected end > start."
        )
    qpos_video = qpos_video[start:end]
    video_times = video_times[start:end]
    video_frame_ids = video_frame_ids[start:end]
    if obj_traj is not None:
        obj_traj = obj_traj[start:end]
    if tactile_i is not None:
        tactile_i = tactile_i[start:end]
    for side in list(tactile_force_i_by_side.keys()):
        tactile_force_i_by_side[side] = tactile_force_i_by_side[side][start:end]

    if args.max_frames and args.max_frames > 0:
        keep = int(args.max_frames)
        qpos_video = qpos_video[:keep]
        video_times = video_times[:keep]
        video_frame_ids = video_frame_ids[:keep]
        if obj_traj is not None:
            obj_traj = obj_traj[:keep]
        if tactile_i is not None:
            tactile_i = tactile_i[:keep]
        for side in list(tactile_force_i_by_side.keys()):
            tactile_force_i_by_side[side] = tactile_force_i_by_side[side][:keep]

    urdf_path = get_robot_urdf_path(args.arm, args.hand)
    link_color_map: Dict[str, np.ndarray] = {}
    robot_fk: Optional[RobotModule] = None
    sensor_frames_by_side: Dict[str, Dict[str, SensorFrame]] = {}
    arrow_handles_by_side: Dict[str, Dict[str, Any]] = {}
    
    if args.visualize_tactile:
        link_color_map = build_link_color_map(urdf_path)
        robot_fk = RobotModule(urdf_path)
        sides_to_render = list(tactile_force_i_by_side.keys())
        if not sides_to_render:
            sides_to_render = [args.tactile_side]
        for side in sides_to_render:
            zone_to_link = _zone_to_link_for_side(side)
            arrow_handles_by_side[side] = {k: None for k in zone_to_link}
            local_meshes_by_link = load_local_link_meshes_from_urdf(
                urdf_path, required_links=set(zone_to_link.values())
            )
            sensor_frames_by_side[side] = _build_sensor_frames_from_local_mesh(local_meshes_by_link, zone_to_link)

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot("robot", urdf_path, include_arm_meshes=args.visualize_arm)
    if obj_mesh is not None and obj_traj is not None:
        vis.add_object(object_name, obj_mesh, obj_traj[0])
        if object_name in vis.obj_dict:
            vis.obj_dict[object_name]["frame"].show_axes = False
    vis.add_traj("traj", {"robot": qpos_video}, {object_name: obj_traj} if obj_traj is not None else {})

    # Wrap the viewer's update to inject tactile arrows per frame.
    original_update_scene = vis.update_scene

    def update_scene_with_tactile(timestep):
        try:
            original_update_scene(timestep)
        except Exception as exc:
            print(f"[visualize_openarm] scene update failed at t={timestep}: {exc}")
            return
        if not args.visualize_tactile:
            return
        if robot_fk is None:
            return
        t = max(0, min(len(video_times) - 1, timestep))
        q = qpos_video[t]
        robot_fk.update_cfg(q[: robot_fk.get_num_joints()])
        tactile_frame = (
            unpack_tactile_frame(tactile_i[t], tactile_index)
            if tactile_i is not None and tactile_index is not None
            else None
        )
        if tactile_force_i_by_side:
            tactile_force_by_side = {
                side: tactile_force_i[t]
                for side, tactile_force_i in tactile_force_i_by_side.items()
            }
        else:
            tactile_force_by_side = {}

        with vis.server.atomic():
            for side, sensor_frames in sensor_frames_by_side.items():
                tactile_force = tactile_force_by_side.get(side)
                for zone, sensor in sensor_frames.items():
                    if sensor is None:
                        continue
                    try:
                        link_pose = robot_fk.get_transform(
                            sensor.link_name, robot_fk.urdf.base_link, collision_geometry=False
                        )
                    except Exception:
                        continue
                    anchor, normal, tx, ty = _world_sensor_frame(sensor, link_pose)
                    vis_normal = -normal
                    if tactile_force is not None:
                        normal_force, tangential_force, tangential_deg = _extract_zone_force(
                            tactile_force, zone
                        )
                    elif tactile_frame is not None:
                        normal_force, tangential_force, tangential_deg = _legacy_zone_force(
                            tactile_frame, zone
                        )
                    else:
                        normal_force, tangential_force, tangential_deg = 0.0, 0.0, -1.0

                    normal_len = (
                        np.clip(normal_force / args.max_normal_force, 0.0, 1.0)
                        * args.max_arrow_len
                    )
                    normal_vec = vis_normal * normal_len
                    tangential_vec = np.zeros(3, dtype=np.float64)
                    if tangential_deg >= 0.0 and tangential_force > 0.0:
                        theta = np.deg2rad(tangential_deg)
                        tangential_dir = np.cos(theta) * tx + np.sin(theta) * ty
                        tangential_dir = _safe_normalize(
                            tangential_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64)
                        )
                        tangential_len = (
                            np.clip(tangential_force / args.max_tangential_force, 0.0, 1.0)
                            * args.max_arrow_len
                        )
                        tangential_vec = tangential_dir * tangential_len
                    total_vec = normal_vec + tangential_vec
                    length = float(np.linalg.norm(total_vec))
                    link_rgba = link_color_map.get(sensor.link_name)
                    if link_rgba is None:
                        color = np.array([255, 255, 255, 255], dtype=np.uint8)
                    else:
                        color = (np.clip(link_rgba, 0.0, 1.0) * 255).astype(np.uint8)
                    direction = total_vec / (length + 1e-12)
                    arrow = make_arrow_mesh(anchor, direction, length, color)
                    if arrow is None:
                        if arrow_handles_by_side[side][zone]:
                            arrow_handles_by_side[side][zone].remove()
                            arrow_handles_by_side[side][zone] = None
                    else:
                        if arrow_handles_by_side[side][zone]:
                            arrow_handles_by_side[side][zone].remove()
                        arrow_handles_by_side[side][zone] = vis.server.scene.add_mesh_trimesh(
                            f"/contact/{side}/{zone}", arrow
                        )

    vis.update_scene = update_scene_with_tactile
    vis.start_viewer()


if __name__ == "__main__":
    main()
