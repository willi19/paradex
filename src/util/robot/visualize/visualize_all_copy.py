import argparse
import copy
import os
import re
from typing import Tuple, Dict, Any

import numpy as np
import trimesh
import yourdfpy

from paradex.utils.path import rsc_path
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.robot.robot_module import robot_info
from paradex.utils.load_data import load_series, resample_to
from paradex.robot.inspire import inspire_action_to_qpos

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


def load_object_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, list):
        return trimesh.util.concatenate(mesh)
    raise ValueError(f"Unexpected mesh type: {type(mesh)}")


def load_object_world_trajectory_npz(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Object trajectory file not found: {npz_path}")

    payload = np.load(npz_path, allow_pickle=True)
    arr = None
    if isinstance(payload, np.lib.npyio.NpzFile):
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


def get_mesh(robot_wrapper: RobotWrapper, robot_obj, state):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire")
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--object-mesh", type=str, default=None, help="Mesh file for the object.")
    parser.add_argument("--visualize-object", action="store_true")
    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--frame-offset", type=int, default=0, help="Positive => robot leads object; negative => robot lags object.")
    args = parser.parse_args()
    object_name = args.object

    capture_root = os.path.join("/home/temp_id/shared_data/capture/hri_inspire_left", args.object, str(args.ep))
    data_root = os.path.join(capture_root, "raw")
    
    arm_dir = os.path.join(data_root, "arm")
    hand_dir = os.path.join(data_root, "hand")

    object_traj_path = os.path.join(capture_root, "object_tracking_result", "obj_T_frames.npz")

    timestamp_path = os.path.join(data_root, "timestamps", "timestamp.npy")
    frame_id_path = os.path.join(data_root, "timestamps", "frame_id.npy")


    c2r_path = os.path.join(capture_root, "C2R.npy")
    
    c2r = np.load(c2r_path)
    robot_from_world = np.linalg.inv(c2r)

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    # hand_action, hand_time = load_series(hand_dir, ("action.npy", "position.npy"))
    hand_action, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))

    hand_action = resample_to(hand_time, hand_action, arm_time)
    if args.hand == "inspire":
        hand_qpos = inspire_action_to_qpos(hand_action)
    else:
        hand_qpos = hand_action

    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

    tactile_seq = None
    tactile_index = None
    if args.visualize_tactile:
        tactile_path = os.path.join(hand_dir, "tactile.npy")
        if not os.path.exists(tactile_path):
            raise FileNotFoundError(f"Tactile file not found: {tactile_path}")
        tactile_payload = np.load(tactile_path, allow_pickle=True)
        tactile_seq = normalize_tactile_sequence(tactile_payload, TACTILE_LAYOUT)
        tactile_index = build_tactile_index_from_layout(TACTILE_LAYOUT)

    obj_traj = None
    if args.visualize_object:
        obj_traj = load_object_world_trajectory_npz(object_traj_path)
        print(f"Loaded object trajectory with {obj_traj.shape[0]} frames from {object_traj_path}")

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
        total_frames = len(video_times)
        if obj_traj.shape[0] == total_frames:
            obj_traj_video = obj_traj
        else:
            obj_time = np.linspace(video_times[0], video_times[-1], obj_traj.shape[0], dtype=float)
            obj_traj_video = resample_to(
                obj_time,
                obj_traj.reshape(obj_traj.shape[0], -1),
                video_times,
            ).reshape(total_frames, 4, 4)
        obj_traj = np.einsum("ij,tjk->tik", robot_from_world, obj_traj_video)
        obj_mesh = load_object_mesh(args.object_mesh)
        mat = getattr(obj_mesh.visual, "material", None)
        if mat is not None:
            factor = np.array(getattr(mat, "baseColorFactor", [1, 1, 1, 1]), dtype=float)
            factor[3] = 0.3  # set desired transparency
            mat.baseColorFactor = factor
            mat.alphaMode = "BLEND"  # keep texture, just blend alpha
            mat.doubleSided = True
    tactile_i = interpolate_sequence(tactile_seq, len(video_times)) if tactile_seq is not None else None
    # tactile_i = resample_to(arm_time, tactile_seq, video_times) if tactile_seq is not None else None

    urdf_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}_left_new.urdf")
    link_color_map = build_link_color_map(urdf_path)
    robot_wrapper = RobotWrapper(urdf_path)
    robot_obj = robot_info(urdf_path, down_sample=True)
    arrow_handles = {k: None for k in TACTILE_VERTEX_MAP} if args.visualize_tactile else {}

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot("robot", urdf_path)
    if obj_mesh is not None and obj_traj is not None:
        vis.add_object(object_name, obj_mesh, obj_traj[0])
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
        meshes = get_mesh(robot_wrapper, robot_obj, q)
        tactile_frame = unpack_tactile_frame(tactile_i[t], tactile_index)

        with vis.server.atomic():
            for name, (link, vids) in TACTILE_VERTEX_MAP.items():
                if name not in tactile_frame or link not in meshes:
                    continue
                # tm = meshes[link].copy()
                # tm.apply_transform(c2r)
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
