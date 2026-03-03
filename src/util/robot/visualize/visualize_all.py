import argparse
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import trimesh
import yourdfpy

from paradex.utils.path import rsc_path
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.visualization.robot import RobotModule
from paradex.utils.load_data import load_series, resample_to
from paradex.robot.inspire import inspire_action_to_qpos, inspire_f1_action_to_qpos_dof6

# Suppress per-frame yourdfpy mimic-chain warnings (thumb_4 -> thumb_3 -> thumb_2).
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


def load_object_world_trajectory_npz(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Object trajectory file not found: {npz_path}")

    payload = np.load(npz_path, allow_pickle=True)
    arr = None
    if isinstance(payload, np.lib.npyio.NpzFile):
        # Case 1: npz contains per-frame 4x4 arrays with keys like frame_35.npy
        if payload.files:
            per_frame = []
            for key in payload.files:
                v = np.asarray(payload[key])
                if v.shape == (4, 4):
                    m = re.search(r"frame_(\d+)", key)
                    frame_idx = int(m.group(1)) if m else None
                    per_frame.append((frame_idx, key, v))
                else:
                    per_frame = []
                    break

            if per_frame:
                if all(idx is not None for idx, _, _ in per_frame):
                    per_frame.sort(key=lambda x: x[0])
                else:
                    per_frame.sort(key=lambda x: x[1])
                arr = np.stack([v for _, _, v in per_frame], axis=0)

        preferred_keys = (
            "obj_T_frames",
            "T",
            "poses",
            "trajectory",
            "traj",
            "arr_0",
        )
        if arr is None:
            for key in preferred_keys:
                if key in payload.files:
                    arr = payload[key]
                    break
        if arr is None:
            if not payload.files:
                raise ValueError(f"No arrays in {npz_path}")
            arr = payload[payload.files[0]]
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
        raise ValueError(
            f"Unsupported object trajectory shape from {npz_path}: {arr.shape}. "
            "Expected (N,4,4) world poses."
        )

    return arr.astype(float)


def load_object_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, list):
        return trimesh.util.concatenate(mesh)
    raise ValueError(f"Unexpected mesh type: {type(mesh)}")


def set_mesh_alpha(mesh: trimesh.Trimesh, alpha: float) -> None:
    a = float(np.clip(alpha, 0.0, 1.0))
    # Preferred path for textured meshes.
    mat = getattr(mesh.visual, "material", None)
    if mat is not None:
        factor = np.array(getattr(mat, "baseColorFactor", [1, 1, 1, 1]), dtype=float)
        factor[3] = a
        mat.baseColorFactor = factor
        mat.alphaMode = "BLEND"
        mat.doubleSided = True

    # Fallback for non-textured meshes.
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is not None:
        vc_arr = np.asarray(vc).copy()
        if vc_arr.ndim == 2 and vc_arr.shape[1] >= 4:
            vc_arr[:, 3] = int(round(a * 255))
            mesh.visual.vertex_colors = vc_arr
    fc = getattr(mesh.visual, "face_colors", None)
    if fc is not None:
        fc_arr = np.asarray(fc).copy()
        if fc_arr.ndim == 2 and fc_arr.shape[1] >= 4:
            fc_arr[:, 3] = int(round(a * 255))
            mesh.visual.face_colors = fc_arr


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
    # Expected format for inspire_f1: object array of per-frame dicts.
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


def _infer_mesh_color(mesh: trimesh.Trimesh) -> np.ndarray:
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is not None:
        vc_arr = np.asarray(vc)
        if vc_arr.ndim == 2 and vc_arr.shape[0] > 0 and vc_arr.shape[1] >= 3:
            rgba = np.array([255, 255, 255, 255], dtype=np.uint8)
            rgba[: min(4, vc_arr.shape[1])] = vc_arr[0, : min(4, vc_arr.shape[1])].astype(np.uint8)
            return rgba
    return np.array([255, 255, 255, 255], dtype=np.uint8)


def _apply_robot_mesh_colors(vis: ViserViewer, robot_name: str, link_rgba: Dict[str, np.ndarray]) -> None:
    robot = vis.robot_dict.get(robot_name)
    if robot is None or not hasattr(robot, "_meshes"):
        return
    link_names = sorted(link_rgba.keys(), key=len, reverse=True)
    for mesh_name, mesh_handle in robot._meshes.items():
        matched = None
        for ln in link_names:
            if mesh_name.endswith(f"/{ln}") or f"/{ln}/" in mesh_name:
                matched = ln
                break
        if matched is None:
            continue
        rgba = link_rgba[matched].astype(np.uint8)
        mesh_handle.color = tuple(int(c) for c in rgba[:3])
        mesh_handle.opacity = float(rgba[3]) / 255.0


def load_local_link_meshes_from_urdf(
    urdf_path: str, required_links: Optional[set] = None
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
        parts: List[trimesh.Trimesh] = []
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_node = geom.find("mesh")
            if mesh_node is None:
                continue
            visual_rgba = None
            material_node = visual.find("material")
            if material_node is not None:
                color_node = material_node.find("color")
                if color_node is not None:
                    visual_rgba = _parse_rgba(color_node.attrib.get("rgba", ""))
                else:
                    material_name = material_node.attrib.get("name", "")
                    if material_name in material_rgba:
                        visual_rgba = material_rgba[material_name]

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
            if visual_rgba is not None and len(tm.vertices) > 0:
                tm.visual.vertex_colors = np.tile(visual_rgba[None, :], (len(tm.vertices), 1))
            parts.append(tm)
        if parts:
            out[link_name] = trimesh.util.concatenate(parts)
    return out


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
    shaft_radius = 0.005
    head_radius = 0.006
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire_f1")
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--object-mesh", type=str, default=None, help="Mesh file for the object.")
    parser.add_argument("--visualize-object", action="store_true")
    parser.add_argument("--object-alpha", type=float, default=0.3, help="Object mesh opacity [0,1]. Lower => more transparent.")
    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--max-normal-force", type=float, default=100.0)
    parser.add_argument("--max-tangential-force", type=float, default=100.0)
    parser.add_argument("--max-arrow-len", type=float, default=0.05)
    parser.add_argument("--frame-offset", type=int, default=0, help="Positive => robot leads object; negative => robot lags object.")
    parser.add_argument("--arm_time_offset", type=float, default=0.09)
    args = parser.parse_args()
    object_name = args.object

    capture_root = os.path.join("/home/temp_id/shared_data/capture/eccv2026/inspire", args.object, str(args.ep))
    data_root = os.path.join(capture_root, "raw")
    
    arm_dir = os.path.join(data_root, "arm")
    hand_dir = os.path.join(data_root, "hand")

    object_track_dir = os.path.join(capture_root, "object_tracking_result")

    timestamp_path = os.path.join(data_root, "timestamps", "timestamp.npy")
    frame_id_path = os.path.join(data_root, "timestamps", "frame_id.npy")


    c2r_path = os.path.join(capture_root, "C2R.npy")
    
    c2r = np.load(c2r_path)
    r2c = np.linalg.inv(c2r)

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    # hand_action, hand_time = load_series(hand_dir, ("action.npy", "position.npy"))
    
    arm_time = arm_time + args.arm_time_offset
    
    if args.hand == "inspire":
        hand_action, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))
    elif args.hand == "inspire_f1":
        hand_action, hand_time = load_series(hand_dir, ("right_joint_states.npy",))

    # Some datasets only store right_joint_states without hand timestamps.
    # In that case, align hand timeline to arm time range before resampling.
    hand_time_path = os.path.join(hand_dir, "time.npy")
    if not os.path.exists(hand_time_path):
        if len(arm_time) > 1:
            hand_time = np.linspace(arm_time[0], arm_time[-1], hand_action.shape[0], dtype=float)
        else:
            hand_time = np.arange(hand_action.shape[0], dtype=float)

    hand_action = resample_to(hand_time, hand_action, arm_time)
    if args.hand == "inspire":
        hand_qpos = inspire_action_to_qpos(hand_action)
    elif args.hand == "inspire_f1":
        hand_qpos = inspire_f1_action_to_qpos_dof6(hand_action)
    else:
        hand_qpos = hand_action

    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

    tactile_seq = None
    tactile_force_seq = None
    tactile_index = None
    if args.visualize_tactile:
        tactile_path = os.path.join(hand_dir, "right_tactile.npy")
        if not os.path.exists(tactile_path):
            raise FileNotFoundError(f"Tactile file not found: {tactile_path}")
        tactile_payload = np.load(tactile_path, allow_pickle=True)
        try:
            tactile_force_seq = normalize_force_dict_sequence(tactile_payload)
        except Exception:
            tactile_seq = normalize_tactile_sequence(tactile_payload, TACTILE_LAYOUT)
            tactile_index = build_tactile_index_from_layout(TACTILE_LAYOUT)

    obj_traj = None
    if args.visualize_object:
        obj_traj_path = os.path.join(object_track_dir, "obj_T_frames.npz")
        obj_traj = load_object_world_trajectory_npz(obj_traj_path)
        print(f"Loaded object trajectory with {obj_traj.shape[0]} frames from {obj_traj_path}")

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
        obj_mesh = load_object_mesh(args.object_mesh)
        set_mesh_alpha(obj_mesh, args.object_alpha)
    if tactile_seq is not None:
        n_tactile = min(len(hand_time), tactile_seq.shape[0])
        tactile_i = resample_to(
            np.asarray(hand_time[:n_tactile], dtype=float),
            np.asarray(tactile_seq[:n_tactile], dtype=float),
            video_times,
        )
    else:
        tactile_i = None
    tactile_force_i = (
        resample_force_dict_sequence(tactile_force_seq, len(video_times))
        if tactile_force_seq is not None
        else None
    )

    urdf_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}_right.urdf")
    link_color_map = build_link_color_map(urdf_path)
    tactile_robot = RobotModule(urdf_path) if args.visualize_tactile else None
    zone_arrow_color: Dict[str, np.ndarray] = {}
    robot_link_rgba: Dict[str, np.ndarray] = {}
    sensor_frames: Dict[str, SensorFrame] = {}
    if args.visualize_tactile:
        all_local_meshes_by_link = load_local_link_meshes_from_urdf(urdf_path)
        robot_link_rgba = {ln: _infer_mesh_color(tm) for ln, tm in all_local_meshes_by_link.items()}

    if args.visualize_tactile and tactile_force_i is not None:
        required_links = set(ZONE_TO_LINK.values())
        local_meshes_by_link = load_local_link_meshes_from_urdf(urdf_path, required_links=required_links)
        sensor_frames = _build_sensor_frames_from_local_mesh(local_meshes_by_link, ZONE_TO_LINK)
        for zone, sensor in sensor_frames.items():
            tm = local_meshes_by_link.get(sensor.link_name)
            if tm is not None:
                zone_arrow_color[zone] = _infer_mesh_color(tm)

    if args.visualize_tactile:
        if tactile_force_i is not None and sensor_frames:
            arrow_handles = {k: None for k in ZONE_TO_LINK}
        elif tactile_i is not None:
            arrow_handles = {k: None for k in TACTILE_VERTEX_MAP}
        else:
            raise ValueError("Unsupported tactile payload: neither force-dict nor legacy tactile matrix could be parsed.")
    else:
        arrow_handles = {}

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot("robot", urdf_path)
    if robot_link_rgba:
        _apply_robot_mesh_colors(vis, "robot", robot_link_rgba)
    if obj_mesh is not None and obj_traj is not None:
        vis.add_object(object_name, obj_mesh, obj_traj[0], opacity=args.object_alpha)
        if object_name in vis.obj_dict:
            vis.obj_dict[object_name]["frame"].show_axes = False
    vis.add_traj("traj", {"robot": qpos_video}, {object_name: obj_traj} if obj_traj is not None else {})

    # Wrap the viewer's update to inject tactile arrows per frame.
    original_update_scene = vis.update_scene

    def update_scene_with_tactile(timestep):
        original_update_scene(timestep)
        if not args.visualize_tactile:
            return
        t = max(0, min(len(video_times) - 1, timestep))
        q = qpos_video[t]
        tactile_robot.update_cfg(q[: tactile_robot.get_num_joints()])

        with vis.server.atomic():
            if tactile_force_i is not None and sensor_frames:
                tactile_force = tactile_force_i[t]
                for zone, sensor in sensor_frames.items():
                    try:
                        link_pose = tactile_robot.get_transform(
                            sensor.link_name, tactile_robot.urdf.base_link, collision_geometry=False
                        )
                    except Exception:
                        continue
                    anchor, normal, tx, ty = _world_sensor_frame(sensor, link_pose)
                    vis_normal = -normal
                    normal_force, tangential_force, tangential_deg = _extract_zone_force(tactile_force, zone)

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
                    color = zone_arrow_color.get(zone)
                    if color is None:
                        link_rgba = link_color_map.get(sensor.link_name)
                        if link_rgba is None:
                            color = np.array([255, 255, 255, 255], dtype=np.uint8)
                        else:
                            color = (np.clip(link_rgba, 0.0, 1.0) * 255).astype(np.uint8)

                    if length <= 1e-6:
                        if arrow_handles[zone]:
                            arrow_handles[zone].remove()
                            arrow_handles[zone] = None
                        continue

                    direction = total_vec / (length + 1e-12)
                    arrow = make_arrow_mesh(anchor, direction, length, color)
                    if arrow is None:
                        if arrow_handles[zone]:
                            arrow_handles[zone].remove()
                            arrow_handles[zone] = None
                    else:
                        if arrow_handles[zone]:
                            arrow_handles[zone].remove()
                        arrow_handles[zone] = vis.server.scene.add_mesh_trimesh(
                            f"/contact/{zone}", arrow
                        )
            elif tactile_i is not None and tactile_index is not None:
                meshes = get_mesh(tactile_robot, q)
                tactile_frame = unpack_tactile_frame(tactile_i[t], tactile_index)
                for name, (link, vids) in TACTILE_VERTEX_MAP.items():
                    if name not in tactile_frame or link not in meshes:
                        continue
                    p = tactile_frame[name].mean()
                    length = np.clip(p / 1000.0, 0, 1) * 0.2
                    link_rgba = link_color_map.get(link)
                    if link_rgba is None:
                        color = np.array([255, 255, 255, 255], dtype=np.uint8)
                    else:
                        color = (np.clip(link_rgba, 0.0, 1.0) * 255).astype(np.uint8)
                    c, n = compute_contact_arrow(meshes[link], vids)
                    arrow = make_arrow_mesh(c, n, length, color)
                    if arrow is None:
                        if arrow_handles[name]:
                            arrow_handles[name].remove()
                            arrow_handles[name] = None
                    else:
                        if arrow_handles[name]:
                            arrow_handles[name].remove()
                        arrow_handles[name] = vis.server.scene.add_mesh_trimesh(
                            f"/contact/{name}", arrow
                        )

    vis.update_scene = update_scene_with_tactile
    vis.start_viewer()


if __name__ == "__main__":
    main()
