import argparse
import copy
import os
from typing import Tuple, Dict, Any, Optional

import numpy as np
import trimesh

from paradex.utils.path import rsc_path
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.dataset_acqusition.match_sync import fill_framedrop, get_synced_data
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.robot.robot_module import robot_info


def load_series(data_dir: str, candidates: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            data = np.load(path)
            time_path = os.path.join(data_dir, "time.npy")
            if os.path.exists(time_path):
                t = np.load(time_path)
            else:
                t = np.arange(data.shape[0], dtype=float)
            return data, t
    raise FileNotFoundError(f"No data found in {data_dir} for {candidates}")


def resample_to(times_src: np.ndarray, data_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    if data_src.shape[0] == times_dst.shape[0] and np.allclose(times_src, times_dst):
        return data_src
    order = np.argsort(times_src)
    times_src = times_src[order]
    data_src = data_src[order]
    out = np.zeros((times_dst.shape[0], data_src.shape[1]), dtype=float)
    for j in range(data_src.shape[1]):
        out[:, j] = np.interp(times_dst, times_src, data_src[:, j])
    return out


def inspire_action_to_qpos(action: np.ndarray) -> np.ndarray:
    # action order: little, ring, middle, index, thumb_2, thumb_1
    limits = {
        "pinky_proximal_joint": 1.6,
        "ring_proximal_joint": 1.6,
        "middle_proximal_joint": 1.6,
        "index_proximal_joint": 1.6,
        "thumb_proximal_pitch_joint": 0.55,
        "thumb_proximal_yaw_joint": 1.15,
    }
    qpos = np.zeros_like(action, dtype=float)
    qpos[:, 0] = limits["thumb_proximal_yaw_joint"] * (1.0 - action[:, 5] / 1000.0)
    qpos[:, 1] = limits["thumb_proximal_pitch_joint"] * (1.0 - action[:, 4] / 1000.0)
    qpos[:, 2] = limits["index_proximal_joint"] * (1.0 - action[:, 3] / 1000.0)
    qpos[:, 3] = limits["middle_proximal_joint"] * (1.0 - action[:, 2] / 1000.0)
    qpos[:, 4] = limits["ring_proximal_joint"] * (1.0 - action[:, 1] / 1000.0)
    qpos[:, 5] = limits["pinky_proximal_joint"] * (1.0 - action[:, 0] / 1000.0)
    return qpos


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


SENSOR_COLORS = {
    "little_tip":   np.array([ 52, 152, 219, 255], dtype=np.uint8),  # blue
    "little_nail":  np.array([ 41, 128, 185, 255], dtype=np.uint8),
    "little_pad":   np.array([ 93, 173, 226, 255], dtype=np.uint8),
    "ring_tip":     np.array([ 46, 204, 113, 255], dtype=np.uint8),  # green
    "ring_nail":    np.array([ 39, 174,  96, 255], dtype=np.uint8),
    "ring_pad":     np.array([ 88, 214, 141, 255], dtype=np.uint8),
    "middle_tip":   np.array([230, 126,  34, 255], dtype=np.uint8),  # orange
    "middle_nail":  np.array([211,  84,   0, 255], dtype=np.uint8),
    "middle_pad":   np.array([241, 196,  15, 255], dtype=np.uint8),  # yellow
    "index_tip":    np.array([155,  89, 182, 255], dtype=np.uint8),  # purple
    "index_nail":   np.array([142,  68, 173, 255], dtype=np.uint8),
    "index_pad":    np.array([190, 144, 212, 255], dtype=np.uint8),
    "thumb_tip":    np.array([231,  76,  60, 255], dtype=np.uint8),  # red
    "thumb_nail":   np.array([192,  57,  43, 255], dtype=np.uint8),
    "thumb_middle": np.array([236, 112,  99, 255], dtype=np.uint8),
    "thumb_pad":    np.array([205,  97,  85, 255], dtype=np.uint8),
    "palm":         np.array([ 26, 188, 156, 255], dtype=np.uint8),  # teal
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


def length_to_color(length, max_len=0.025):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--hand", type=str, required=True)
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/temp_id/shared_data/capture/hri_inspire_left/smallbowl/1/raw",
        help="Capture root containing arm/ and hand/ directories.",
    )
    parser.add_argument(
        "--object-track-dir",
        type=str,
        default="/home/temp_id/shared_data/capture/hri_inspire_left/smallbowl/1/object_tracking",
        help="Directory containing object pose pickle/npy.",
    )
    parser.add_argument(
        "--object-mesh",
        type=str,
        default="/home/temp_id/shared_data/mesh/servingbowl_small/servingbowl_small.obj",
        help="Mesh file for the object.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="object",
        help="Name for the object in the viewer.",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Positive => robot leads object; negative => robot lags object.",
    )
    parser.add_argument(
        "--timestamp-path",
        type=str,
        default="/home/temp_id/shared_data/mesh/servingbowl_small/1/raw/timestamps/timestamp.npy",
        help="Timestamp npy used as the master timeline for sync.",
    )
    parser.add_argument(
        "--frame-id-path",
        type=str,
        default="/home/temp_id/shared_data/mesh/servingbowl_small/1/raw/timestamps/frame_id.npy",
        help="Optional frame_id npy; if missing, a sequential id is used.",
    )
    args = parser.parse_args()

    arm_dir = os.path.join(args.data_root, "arm")
    hand_dir = os.path.join(args.data_root, "hand")

    c2r_path = os.path.join(os.path.dirname(args.data_root), "C2R.npy")
    c2r = np.load(c2r_path)
    r2c = np.linalg.inv(c2r)

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    # hand_action, hand_time = load_series(hand_dir, ("action.npy", "position.npy"))
    hand_action, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))

    hand_action = resample_to(hand_time, hand_action, arm_time)
    if args.hand == "inspire":
        hand_qpos = inspire_action_to_qpos(hand_action)
    else:
        hand_qpos = hand_action

    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

    tactile_path = os.path.join(hand_dir, "tactile.npy")
    if not os.path.exists(tactile_path):
        raise FileNotFoundError(f"Tactile file not found: {tactile_path}")
    tactile_seq = np.load(tactile_path)
    tactile_index = build_tactile_index_from_layout(TACTILE_LAYOUT)

    obj_traj = load_object_trajectory(args.object_track_dir)
    print(f"Loaded object trajectory with {obj_traj.shape[0]} frames.")

    # Build master timeline from camera timestamps (pc_time) using fill_framedrop logic.
    ts_path = args.timestamp_path
    frame_id_path = args.frame_id_path
    if os.path.exists(ts_path):
        pc_time_raw = np.load(ts_path)
        if os.path.exists(frame_id_path):
            frame_id_raw = np.load(frame_id_path)
        else:
            frame_id_raw = np.arange(1, len(pc_time_raw) + 1, dtype=int)
        video_times, video_frame_ids = fill_framedrop(frame_id_raw, pc_time_raw)
    else:
        video_times = arm_time
        video_frame_ids = np.arange(1, full_qpos.shape[0] + 1, dtype=int)

    # Snap robot states onto the master timeline.
    qpos_video = get_synced_data(video_times, full_qpos, arm_time)
    if args.frame_offset != 0:
        shifted = np.zeros_like(qpos_video)
        max_idx = len(qpos_video) - 1
        for i in range(len(qpos_video)):
            src = min(max(i + args.frame_offset, 0), max_idx)
            shifted[i] = qpos_video[src]
        qpos_video = shifted

    # Snap object trajectory onto the master timeline (assume uniform spacing across its original length).
    obj_time = np.linspace(video_times[0], video_times[-1], obj_traj.shape[0])
    obj_traj = resample_to(
        obj_time,
        obj_traj.reshape(obj_traj.shape[0], -1),
        video_times,
    ).reshape(len(video_times), 4, 4)
    # obj_traj = np.einsum("ij,tjk->tik", c2r, obj_traj)
    
    obj_traj = np.einsum("ij,tjk->tik", r2c, obj_traj)
    obj_mesh = load_object_mesh(args.object_mesh)
    tactile_i = interpolate_sequence(tactile_seq, len(video_times))

    urdf_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}_left_new.urdf")
    robot_wrapper = RobotWrapper(urdf_path)
    robot_obj = robot_info(urdf_path, down_sample=True)
    arrow_handles = {k: None for k in TACTILE_VERTEX_MAP}
    mesh_handles = {k: None for k in TACTILE_VERTEX_MAP}

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot("robot", urdf_path)
    vis.add_object(args.object_name, obj_mesh, obj_traj[0])
    if args.object_name in vis.obj_dict:
        vis.obj_dict[args.object_name]["frame"].show_axes = False
    vis.add_traj("traj", {"robot": qpos_video}, {args.object_name: obj_traj})

    # Wrap the viewer's update to inject tactile arrows per frame.
    original_update_scene = vis.update_scene

    def update_scene_with_tactile(timestep):
        original_update_scene(timestep)
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
                length = np.clip(p / 1000.0, 0, 1) * 0.025
                color = SENSOR_COLORS.get(name, np.array([255, 255, 255, 255], dtype=np.uint8))
                c, n = compute_contact_arrow(meshes[link], vids)
                arrow = make_arrow_mesh(c, n, length, color)
                if arrow is None:
                    if arrow_handles[name]:
                        arrow_handles[name].remove()
                        arrow_handles[name] = None
                    if mesh_handles[name]:
                        mesh_handles[name].remove()
                        mesh_handles[name] = None
                else:
                    if arrow_handles[name]:
                        arrow_handles[name].remove()
                    arrow_handles[name] = vis.server.scene.add_mesh_trimesh(
                        f"/contact/{name}", arrow
                    )
                    # Add a colored mesh overlay for the corresponding link to improve visibility.
                    colored_mesh = meshes[link].copy()
                    colored_mesh.visual.vertex_colors = np.tile(color, (colored_mesh.vertices.shape[0], 1))
                    if mesh_handles[name]:
                        mesh_handles[name].remove()
                    mesh_handles[name] = vis.server.scene.add_mesh_trimesh(
                        f"/contact_mesh/{name}", colored_mesh
                    )

    vis.update_scene = update_scene_with_tactile
    vis.start_viewer()


if __name__ == "__main__":
    main()
