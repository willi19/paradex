import argparse
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from paradex.utils.path import rsc_path
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer


logging.getLogger("yourdfpy.urdf").setLevel(logging.ERROR)

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


@dataclass
class SensorFrame:
    link_name: str
    anchor_local: np.ndarray
    normal_local: np.ndarray
    tangent_x_local: np.ndarray
    tangent_y_local: np.ndarray


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
    if not seq:
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


def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return fallback.copy()
    return v / norm


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
) -> Dict[str, SensorFrame]:
    frames: Dict[str, SensorFrame] = {}
    for zone in ("little", "ring", "middle", "index", "thumb"):
        link_name = ZONE_TO_LINK[zone]
        tm = local_meshes_by_link.get(link_name)
        if tm is None or len(tm.vertices) < 4:
            continue
        anchor, normal, tx, ty = _estimate_surface_frame(np.asarray(tm.vertices))
        frames[zone] = SensorFrame(link_name, anchor, normal, tx, ty)

    palm_link = ZONE_TO_LINK["palm_middle"]
    palm_tm = local_meshes_by_link.get(palm_link)
    if palm_tm is None or len(palm_tm.vertices) <= 8:
        return frames

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
        thresh = np.quantile(seg_proj, 0.85)
        seg_mask = seg_proj >= thresh
        if np.count_nonzero(seg_mask) < 4:
            seg_mask = np.ones(len(seg), dtype=bool)
        anchor = seg[seg_mask].mean(axis=0)
        frames[zone] = SensorFrame(palm_link, anchor, normal, tx, ty)

    return frames


def _world_sensor_frame(sensor: SensorFrame, link_pose: np.ndarray):
    rot = link_pose[:3, :3]
    pos = link_pose[:3, 3]
    anchor = rot @ sensor.anchor_local + pos
    normal = _safe_normalize(rot @ sensor.normal_local, np.array([0.0, 0.0, 1.0]))
    tx = _safe_normalize(rot @ sensor.tangent_x_local, np.array([1.0, 0.0, 0.0]))
    ty = _safe_normalize(rot @ sensor.tangent_y_local, np.array([0.0, 1.0, 0.0]))
    return anchor, normal, tx, ty


def _parse_xyz(text: str, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    vals = str(text).split() if text is not None else []
    if len(vals) != 3:
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in vals], dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


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


def _resolve_mesh_path(mesh_uri: str, urdf_path: str, robot_name: str) -> str:
    if mesh_uri.startswith("package://"):
        rest = mesh_uri[len("package://") :]
        pkg_name, rel = rest.split("/", 1)
        package_root = os.path.dirname(os.path.dirname(urdf_path))
        if pkg_name != robot_name:
            return os.path.join(package_root, rel)
        return os.path.join(package_root, rel)
    return os.path.normpath(os.path.join(os.path.dirname(urdf_path), mesh_uri))


def _infer_mesh_color(mesh: trimesh.Trimesh) -> np.ndarray:
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is None:
        return np.array([255, 255, 255, 255], dtype=np.uint8)
    vc_arr = np.asarray(vc)
    if vc_arr.ndim != 2 or vc_arr.shape[0] == 0 or vc_arr.shape[1] < 3:
        return np.array([255, 255, 255, 255], dtype=np.uint8)
    rgba = np.array([255, 255, 255, 255], dtype=np.uint8)
    rgba[: min(4, vc_arr.shape[1])] = vc_arr[0, : min(4, vc_arr.shape[1])].astype(np.uint8)
    return rgba


def _apply_robot_mesh_colors(vis: ViserViewer, robot_name: str, link_rgba: Dict[str, np.ndarray]) -> None:
    robot = vis.robot_dict.get(robot_name)
    if robot is None or not hasattr(robot, "_meshes"):
        return
    link_names = sorted(link_rgba.keys(), key=len, reverse=True)
    for mesh_name, mesh_handle in robot._meshes.items():
        matched = None
        for link_name in link_names:
            if mesh_name.endswith(f"/{link_name}") or f"/{link_name}/" in mesh_name:
                matched = link_name
                break
        if matched is None:
            continue
        rgba = link_rgba[matched].astype(np.uint8)
        mesh_handle.color = tuple(int(c) for c in rgba[:3])
        mesh_handle.opacity = float(rgba[3]) / 255.0


def load_local_link_meshes_from_urdf(
    urdf_path: str,
    required_links: Optional[set] = None,
) -> Dict[str, trimesh.Trimesh]:
    root = ET.parse(urdf_path).getroot()
    robot_name = root.attrib.get("name", "")
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
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh_node = geometry.find("mesh")
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

            mesh_uri = mesh_node.attrib.get("filename", "")
            if not mesh_uri:
                continue
            mesh_path = _resolve_mesh_path(mesh_uri, urdf_path, robot_name)
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
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = _rpy_to_matrix(rpy)
                transform[:3, 3] = xyz
                tm.apply_transform(transform)
            if visual_rgba is not None and len(tm.vertices) > 0:
                tm.visual.vertex_colors = np.tile(visual_rgba[None, :], (len(tm.vertices), 1))
            parts.append(tm)

        if parts:
            out[link_name] = trimesh.util.concatenate(parts)

    return out


def make_arrow_mesh(
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    color_rgba: np.ndarray,
) -> Optional[trimesh.Trimesh]:
    if length < 1e-6:
        return None
    shaft_radius = 0.005
    head_radius = 0.006
    shaft_height = length * 0.7
    head_height = length * 0.35
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_height)
    head = trimesh.creation.cone(radius=head_radius, height=head_height)
    shaft.apply_translation([0.0, 0.0, shaft_height * 0.5])
    head.apply_translation([0.0, 0.0, shaft_height + head_height * 0.5])
    arrow = trimesh.util.concatenate([shaft, head])
    align = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    if align is not None:
        arrow.apply_transform(align)
    arrow.apply_translation(start)
    arrow.visual.vertex_colors = np.tile(color_rgba, (arrow.vertices.shape[0], 1))
    return arrow


def _hand_urdf_path(hand_side: str) -> str:
    if hand_side == "left":
        return os.path.join(
            rsc_path, "robot", "inspire_f1", "URDF-2", "URDF-2", "RH56F1_L", "urdf", "RH56F1_L.urdf"
        )
    return os.path.join(
        rsc_path, "robot", "inspire_f1", "URDF-2", "URDF-2", "RH56F1_R", "urdf", "RH56F1_R.urdf"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument(
        "--capture-root",
        type=str,
        default="/home/temp_id/shared_data/capture/eccv2026/inspire_f1",
        help="Base directory containing per-object capture folders.",
    )
    parser.add_argument("--hand-side", type=str, choices=("left", "right"), default="right")
    parser.add_argument("--max-normal-force", type=float, default=100.0)
    parser.add_argument("--max-tangential-force", type=float, default=100.0)
    parser.add_argument("--max-arrow-len", type=float, default=0.05)
    args = parser.parse_args()

    capture_root = os.path.join(args.capture_root, args.object, str(args.ep))
    hand_dir = os.path.join(capture_root, "raw", "hand")
    tactile_path = os.path.join(hand_dir, f"{args.hand_side}_tactile.npy")
    if not os.path.exists(tactile_path):
        raise FileNotFoundError(f"Tactile file not found: {tactile_path}")

    tactile_payload = np.load(tactile_path, allow_pickle=True)
    tactile_force_seq = normalize_force_dict_sequence(tactile_payload)
    if not tactile_force_seq:
        raise ValueError(f"No tactile frames found in {tactile_path}")

    timestamp_path = os.path.join(capture_root, "raw", "timestamps", "timestamp.npy")
    num_frames = len(np.load(timestamp_path)) if os.path.exists(timestamp_path) else len(tactile_force_seq)
    tactile_force_i = resample_force_dict_sequence(tactile_force_seq, num_frames)

    urdf_path = _hand_urdf_path(args.hand_side)
    tactile_robot = RobotModule(urdf_path)
    open_qpos = np.zeros((num_frames, tactile_robot.get_num_joints()), dtype=float)

    all_local_meshes_by_link = load_local_link_meshes_from_urdf(urdf_path)
    robot_link_rgba = {link_name: _infer_mesh_color(mesh) for link_name, mesh in all_local_meshes_by_link.items()}

    required_links = set(ZONE_TO_LINK.values())
    sensor_meshes = load_local_link_meshes_from_urdf(urdf_path, required_links=required_links)
    sensor_frames = _build_sensor_frames_from_local_mesh(sensor_meshes)
    if not sensor_frames:
        raise ValueError("Failed to build sensor frames from hand URDF.")

    zone_arrow_color: Dict[str, np.ndarray] = {}
    for zone, sensor in sensor_frames.items():
        mesh = sensor_meshes.get(sensor.link_name)
        if mesh is not None:
            zone_arrow_color[zone] = _infer_mesh_color(mesh)

    arrow_handles = {zone: None for zone in sensor_frames}

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot("robot", urdf_path)
    _apply_robot_mesh_colors(vis, "robot", robot_link_rgba)
    vis.add_traj("traj", {"robot": open_qpos}, {})

    original_update_scene = vis.update_scene

    def update_scene_with_tactile(timestep):
        original_update_scene(timestep)
        t = max(0, min(num_frames - 1, timestep))
        tactile_robot.update_cfg(open_qpos[t])
        tactile_force = tactile_force_i[t]

        with vis.server.atomic():
            for zone, sensor in sensor_frames.items():
                try:
                    link_pose = tactile_robot.get_transform(
                        sensor.link_name,
                        tactile_robot.urdf.base_link,
                        collision_geometry=False,
                    )
                except Exception:
                    continue

                anchor, normal, tx, ty = _world_sensor_frame(sensor, link_pose)
                vis_normal = -normal
                normal_force, tangential_force, tangential_deg = _extract_zone_force(tactile_force, zone)

                normal_len = np.clip(normal_force / args.max_normal_force, 0.0, 1.0) * args.max_arrow_len
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
                handle = arrow_handles[zone]
                if length <= 1e-6:
                    if handle is not None:
                        handle.remove()
                        arrow_handles[zone] = None
                    continue

                direction = total_vec / (length + 1e-12)
                color = zone_arrow_color.get(zone, np.array([255, 255, 255, 255], dtype=np.uint8))
                arrow = make_arrow_mesh(anchor, direction, length, color)
                if handle is not None:
                    handle.remove()
                arrow_handles[zone] = vis.server.scene.add_mesh_trimesh(f"/contact/{zone}", arrow)

    vis.update_scene = update_scene_with_tactile
    vis.start_viewer()


if __name__ == "__main__":
    main()
