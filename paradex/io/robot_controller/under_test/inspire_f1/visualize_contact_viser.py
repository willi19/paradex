import argparse
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import trimesh
import viser
from control_msgs.msg import DynamicInterfaceGroupValues
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from paradex.robot.robot_wrapper_deprecated import RobotWrapper


DEFAULT_URDF = (
    "/home/temp_id/paradex/rsc/robot/inspire_f1/URDF-2/URDF-2/"
    "RH56F1_L/urdf/RH56F1_L.urdf"
)
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


class InspireF1TactileReceiver(Node):
    def __init__(self, hand_side: str = "left", namespace: str = ""):
        if hand_side != "left":
            raise ValueError("This visualizer currently supports left hand only.")

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        super().__init__("inspire_f1_tactile_visualizer")

        self._lock = threading.Lock()
        self._latest_joint_vec: Optional[np.ndarray] = None
        self._latest_joint_map: Dict[str, float] = {}
        self._latest_tactile: Dict[str, float] = {}
        self._hand_side = hand_side

        prefix = f"/{namespace.strip('/')}" if namespace else ""
        side_prefix = f"{prefix}/{hand_side}"
        joint_topics = [f"{side_prefix}/joint_states", f"{prefix}/joint_states"]
        tactile_topics = [f"{side_prefix}/tactile_sensor_states", f"{prefix}/tactile_sensor_states"]
        self._joint_subs = []
        self._tactile_subs = []
        for topic in dict.fromkeys(joint_topics):
            self._joint_subs.append(
                self.create_subscription(JointState, topic, self._joint_cb, 50)
            )
        for topic in dict.fromkeys(tactile_topics):
            self._tactile_subs.append(
                self.create_subscription(
                    DynamicInterfaceGroupValues,
                    topic,
                    self._tactile_cb,
                    50,
                )
            )
        self.get_logger().info(f"joint topics: {list(dict.fromkeys(joint_topics))}")
        self.get_logger().info(f"tactile topics: {list(dict.fromkeys(tactile_topics))}")

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self):
        try:
            self._executor.spin()
        except Exception as exc:
            self.get_logger().error(f"ROS2 spin failed: {exc}")

    def _joint_cb(self, msg: JointState):
        with self._lock:
            self._latest_joint_vec = np.array(msg.position, dtype=np.float64)
            self._latest_joint_map = {
                name: float(val) for name, val in zip(msg.name, msg.position)
            }

    def _tactile_cb(self, msg: DynamicInterfaceGroupValues):
        tactile_map: Dict[str, float] = {}
        group_name = f"{self._hand_side}_hand_tactile_sensor"

        # 1) Expected group name match.
        for idx, group in enumerate(msg.interface_groups):
            if group == group_name and idx < len(msg.interface_values):
                iv = msg.interface_values[idx]
                for key, value in zip(iv.interface_names, iv.values):
                    tactile_map[key] = float(value)
                break

        # 2) Fallback: pick any interface block containing tactile keys.
        if not tactile_map:
            for iv in msg.interface_values:
                keys = list(iv.interface_names)
                if any(
                    key.endswith("_normal_force")
                    or key.endswith("_tangential_force")
                    or key.endswith("_tangential_direction")
                    for key in keys
                ):
                    for key, value in zip(iv.interface_names, iv.values):
                        tactile_map[key] = float(value)
                    break

        if tactile_map:
            with self._lock:
                self._latest_tactile = tactile_map

    def get_snapshot(self) -> Tuple[Optional[np.ndarray], Dict[str, float], Dict[str, float]]:
        with self._lock:
            qvec = None if self._latest_joint_vec is None else self._latest_joint_vec.copy()
            qmap = dict(self._latest_joint_map)
            tactile = dict(self._latest_tactile)
        return qvec, qmap, tactile

    def close(self):
        try:
            self._executor.remove_node(self)
        except Exception:
            pass
        self._executor.shutdown()
        self.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()
        self._spin_thread.join(timeout=2.0)


def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return fallback.copy()
    return v / n


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _parse_xyz(xyz_str: str) -> np.ndarray:
    if not xyz_str:
        return np.zeros(3, dtype=np.float64)
    return np.array([float(x) for x in xyz_str.split()], dtype=np.float64)


def _parse_rpy(rpy_str: str) -> np.ndarray:
    if not rpy_str:
        return np.zeros(3, dtype=np.float64)
    return np.array([float(x) for x in rpy_str.split()], dtype=np.float64)


def _parse_rgba(rgba_str: str) -> Optional[np.ndarray]:
    if not rgba_str:
        return None
    vals = [float(x) for x in rgba_str.split()]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        return None
    rgba = np.clip(np.array(vals, dtype=np.float64), 0.0, 1.0)
    return (rgba * 255.0).astype(np.uint8)


def _resolve_mesh_path(mesh_uri: str, urdf_path: Path, robot_name: str) -> Path:
    if mesh_uri.startswith("package://"):
        rest = mesh_uri[len("package://") :]
        pkg, rel = rest.split("/", 1)
        package_root = urdf_path.parent.parent
        if pkg != robot_name:
            return package_root / rel
        return package_root / rel
    return (urdf_path.parent / mesh_uri).resolve()


def _load_urdf_meshes(urdf_path: str) -> Dict[str, List[trimesh.Trimesh]]:
    urdf_file = Path(urdf_path).resolve()
    root = ET.parse(urdf_file).getroot()
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

    link_meshes: Dict[str, List[trimesh.Trimesh]] = {}
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        meshes: List[trimesh.Trimesh] = []
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
            mesh_path = _resolve_mesh_path(mesh_uri, urdf_file, robot_name)
            if not mesh_path.exists():
                print(f"[WARN] mesh not found: {mesh_path}")
                continue

            loaded = trimesh.load_mesh(mesh_path, process=False)
            if isinstance(loaded, trimesh.Scene):
                parts = [
                    g
                    for g in loaded.geometry.values()
                    if isinstance(g, trimesh.Trimesh)
                ]
                if not parts:
                    continue
                tm = trimesh.util.concatenate(parts)
            else:
                tm = loaded

            origin = visual.find("origin")
            if origin is not None:
                xyz = _parse_xyz(origin.attrib.get("xyz", "0 0 0"))
                rpy = _parse_rpy(origin.attrib.get("rpy", "0 0 0"))
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = _rpy_to_matrix(rpy)
                T[:3, 3] = xyz
                tm = tm.copy()
                tm.apply_transform(T)

            if visual_rgba is not None and len(tm.vertices) > 0:
                tm.visual.vertex_colors = np.tile(visual_rgba[None, :], (len(tm.vertices), 1))

            meshes.append(tm)
        if meshes:
            link_meshes[link_name] = meshes
    return link_meshes


def _merge_meshes(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    copied = [m.copy() for m in meshes]
    return trimesh.util.concatenate(copied)


def _estimate_link_surface_frame(
    mesh_local: trimesh.Trimesh,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    verts = np.asarray(mesh_local.vertices)
    center = verts.mean(axis=0)
    centered = verts - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)

    normal = eigvecs[:, order[0]]
    tangent_x = eigvecs[:, order[-1]]
    normal = _safe_normalize(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    tangent_x = tangent_x - np.dot(tangent_x, normal) * normal
    tangent_x = _safe_normalize(tangent_x, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    proj = verts @ normal
    q_high = np.quantile(proj, 0.9)
    q_low = np.quantile(proj, 0.1)
    side_sign = 1
    if abs(q_low) > abs(q_high):
        normal = -normal
        side_sign = -1
        surf_mask = proj <= q_low
    else:
        surf_mask = proj >= q_high

    if np.count_nonzero(surf_mask) < 4:
        surf_mask = np.ones(len(verts), dtype=bool)
    anchor = verts[surf_mask].mean(axis=0)

    if np.dot(tangent_x, np.array([1.0, 0.0, 0.0], dtype=np.float64)) < 0:
        tangent_x = -tangent_x
    tangent_y = _safe_normalize(
        np.cross(normal, tangent_x), np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    tangent_x = _safe_normalize(
        np.cross(tangent_y, normal), np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    return anchor, normal, tangent_x, tangent_y, side_sign


def _build_sensor_frames(
    local_meshes_by_link: Dict[str, List[trimesh.Trimesh]],
) -> Dict[str, SensorFrame]:
    frames: Dict[str, SensorFrame] = {}
    tangential_flip_zones = {"thumb", "middle", "index"}

    finger_zones = ("little", "ring", "middle", "index", "thumb")
    for zone in finger_zones:
        link_name = ZONE_TO_LINK[zone]
        if link_name not in local_meshes_by_link:
            raise KeyError(f"Required link mesh missing: {link_name}")
        merged = _merge_meshes(local_meshes_by_link[link_name])
        anchor, normal, tx, ty, _ = _estimate_link_surface_frame(merged)
        if zone in tangential_flip_zones:
            tx = -tx
            ty = -ty
        frames[zone] = SensorFrame(link_name, anchor, normal, tx, ty)

    palm_link = ZONE_TO_LINK["palm_middle"]
    if palm_link not in local_meshes_by_link:
        raise KeyError(f"Required link mesh missing: {palm_link}")
    palm_mesh = _merge_meshes(local_meshes_by_link[palm_link])
    verts = np.asarray(palm_mesh.vertices)
    _, normal, tx, ty, side_sign = _estimate_link_surface_frame(palm_mesh)

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
        if side_sign >= 0:
            t = np.quantile(seg_proj, 0.85)
            seg_mask = seg_proj >= t
        else:
            t = np.quantile(seg_proj, 0.15)
            seg_mask = seg_proj <= t
        if np.count_nonzero(seg_mask) < 4:
            seg_mask = np.ones(len(seg), dtype=bool)
        anchor = seg[seg_mask].mean(axis=0)
        frames[zone] = SensorFrame(palm_link, anchor, normal, tx, ty)

    return frames


def _fit_qpos(
    robot_wrapper: RobotWrapper,
    joint_vec: Optional[np.ndarray],
    joint_map: Dict[str, float],
) -> np.ndarray:
    q = np.zeros(robot_wrapper.dof, dtype=np.float64)
    filled = np.zeros(robot_wrapper.dof, dtype=bool)

    dof_names = robot_wrapper.dof_joint_names
    for idx, name in enumerate(dof_names):
        if name in joint_map:
            q[idx] = joint_map[name]
            filled[idx] = True

    if joint_vec is not None:
        n = min(len(joint_vec), len(q))
        for i in range(n):
            if not filled[i]:
                q[i] = joint_vec[i]

    return q


def _length_to_color(length: float, max_len: float = 0.025) -> np.ndarray:
    t = np.clip(length / max_len, 0.0, 1.0)
    if t < 0.25:
        r, g, b = 0, int(4 * t * 255), 255
    elif t < 0.5:
        r, g, b = 0, 255, int((1 - 4 * (t - 0.25)) * 255)
    elif t < 0.75:
        r, g, b = int(4 * (t - 0.5) * 255), 255, 0
    else:
        r, g, b = 255, int((1 - 4 * (t - 0.75)) * 255), 0
    return np.array([r, g, b, 255], dtype=np.uint8)


def _make_arrow_mesh(
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    color: np.ndarray,
) -> Optional[trimesh.Trimesh]:
    if length <= 1e-6:
        return None
    direction = _safe_normalize(direction, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    # Match the thicker arrow style used in visualize_all.py.
    shaft_radius = 0.003
    head_radius = 0.004
    shaft_height = length * 0.7
    head_height = length * 0.35
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_height)
    head = trimesh.creation.cone(radius=head_radius, height=head_height)
    shaft.apply_translation([0.0, 0.0, shaft_height * 0.5])
    head.apply_translation([0.0, 0.0, shaft_height + head_height * 0.5])
    arrow = trimesh.util.concatenate([shaft, head])

    T = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    if T is not None:
        arrow.apply_transform(T)
    arrow.apply_translation(start)
    arrow.visual.vertex_colors = np.tile(color[None, :], (len(arrow.vertices), 1))
    return arrow


def _extract_zone_force(tactile: Dict[str, float], zone: str) -> Tuple[float, float, float]:
    normal = float(tactile.get(f"{zone}_normal_force", 0.0))
    tangent = float(tactile.get(f"{zone}_tangential_force", 0.0))
    direction_deg = float(tactile.get(f"{zone}_tangential_direction", -1.0))
    return normal, tangent, direction_deg


def _world_sensor_frame(sensor: SensorFrame, link_pose: np.ndarray):
    R = link_pose[:3, :3]
    p = link_pose[:3, 3]
    anchor = R @ sensor.anchor_local + p
    normal = _safe_normalize(R @ sensor.normal_local, np.array([0.0, 0.0, 1.0]))
    tx = _safe_normalize(R @ sensor.tangent_x_local, np.array([1.0, 0.0, 0.0]))
    ty = _safe_normalize(R @ sensor.tangent_y_local, np.array([0.0, 1.0, 0.0]))
    return anchor, normal, tx, ty


def _infer_link_arrow_color(local_meshes: List[trimesh.Trimesh]) -> np.ndarray:
    for mesh in local_meshes:
        vc = getattr(mesh.visual, "vertex_colors", None)
        if vc is None:
            continue
        vc_arr = np.asarray(vc)
        if vc_arr.ndim == 2 and vc_arr.shape[0] > 0 and vc_arr.shape[1] >= 3:
            rgba = np.array([255, 255, 255, 255], dtype=np.uint8)
            rgba[: min(4, vc_arr.shape[1])] = vc_arr[0, : min(4, vc_arr.shape[1])].astype(
                np.uint8
            )
            return rgba
    return np.array([255, 255, 255, 255], dtype=np.uint8)


def _rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / qn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-path", type=str, default=DEFAULT_URDF)
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument(
        "--use-live-joints",
        action="store_true",
        help="Use incoming /joint_states. If omitted, keeps the hand open (all zeros).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-normal-force", type=float, default=100.0)
    parser.add_argument("--max-tangential-force", type=float, default=100.0)
    parser.add_argument("--max-arrow-len", type=float, default=0.025)
    return parser.parse_args()


def main():
    args = parse_args()

    server = viser.ViserServer()
    robot_wrapper = RobotWrapper(args.urdf_path)
    receiver = InspireF1TactileReceiver(hand_side="left", namespace=args.namespace)

    local_meshes_by_link = _load_urdf_meshes(args.urdf_path)
    sensor_frames = _build_sensor_frames(local_meshes_by_link)
    sensor_arrow_color = {
        zone: _infer_link_arrow_color(local_meshes_by_link[sensor.link_name])
        for zone, sensor in sensor_frames.items()
        if sensor.link_name in local_meshes_by_link
    }

    robot_handles: Dict[str, viser.MeshHandle] = {}
    arrow_handles: Dict[str, viser.MeshHandle] = {}
    open_hand_qpos = np.zeros(robot_wrapper.dof, dtype=np.float64)
    # Add static meshes once in each link's local frame; update pose only per frame.
    for link_name, local_meshes in local_meshes_by_link.items():
        for i, local_tm in enumerate(local_meshes):
            hname = f"/robot/{link_name}/{i}"
            robot_handles[hname] = server.scene.add_mesh_trimesh(
                name=hname,
                mesh=local_tm,
            )

    try:
        while True:
            joint_vec, joint_map, tactile = receiver.get_snapshot()
            if args.use_live_joints:
                qpos = _fit_qpos(robot_wrapper, joint_vec, joint_map)
            else:
                qpos = open_hand_qpos
            robot_wrapper.compute_forward_kinematics(qpos)

            link_pose_map: Dict[str, np.ndarray] = {}
            for link_name in local_meshes_by_link:
                try:
                    link_pose_map[link_name] = robot_wrapper.get_link_pose(
                        robot_wrapper.get_link_index(link_name)
                    )
                except Exception:
                    continue

            for link_name, local_meshes in local_meshes_by_link.items():
                if link_name not in link_pose_map:
                    continue
                link_pose = link_pose_map[link_name]
                for i, local_tm in enumerate(local_meshes):
                    hname = f"/robot/{link_name}/{i}"
                    handle = robot_handles.get(hname)
                    if handle is None:
                        continue
                    handle.position = link_pose[:3, 3]
                    handle.wxyz = _rotmat_to_wxyz(link_pose[:3, :3])

            for zone, sensor in sensor_frames.items():
                link_pose = link_pose_map.get(sensor.link_name)
                if link_pose is None:
                    continue
                anchor, normal, tx, ty = _world_sensor_frame(sensor, link_pose)
                vis_normal = -normal
                normal_force, tangential_force, tangential_deg = _extract_zone_force(
                    tactile, zone
                )

                normal_len = (
                    np.clip(normal_force / args.max_normal_force, 0.0, 1.0)
                    * args.max_arrow_len
                )
                normal_vec = vis_normal * normal_len
                tangential_vec = np.zeros(3, dtype=np.float64)
                if tangential_deg >= 0.0 and tangential_force > 0.0:
                    theta = np.deg2rad(tangential_deg)
                    tangential_dir = (
                        np.cos(theta) * tx + np.sin(theta) * ty
                    )
                    tangential_dir = _safe_normalize(
                        tangential_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    )
                    tangential_len = (
                        np.clip(
                            tangential_force / args.max_tangential_force,
                            0.0,
                            1.0,
                        )
                        * args.max_arrow_len
                    )
                    tangential_vec = tangential_dir * tangential_len

                total_vec = normal_vec + tangential_vec
                total_len = float(np.linalg.norm(total_vec))
                total_name = f"/contact_arrow/{zone}"
                if total_len > 1e-6:
                    total_dir = total_vec / total_len
                    color = sensor_arrow_color.get(
                        zone, np.array([255, 255, 255, 255], dtype=np.uint8)
                    )
                    total_arrow = _make_arrow_mesh(anchor, total_dir, total_len, color)
                    if total_arrow is not None:
                        if total_name in arrow_handles:
                            arrow_handles[total_name].remove()
                        arrow_handles[total_name] = server.scene.add_mesh_trimesh(
                            name=total_name,
                            mesh=total_arrow,
                        )
                elif total_name in arrow_handles:
                    arrow_handles[total_name].remove()
                    del arrow_handles[total_name]

            time.sleep(max(0.0, 1.0 / args.fps))
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
