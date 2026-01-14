import argparse
import os
from typing import Tuple, Dict, Any, Optional

import numpy as np
import trimesh

from paradex.utils.path import rsc_path
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.dataset_acqusition.match_sync import fill_framedrop, get_synced_data


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

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    # hand_action, hand_time = load_series(hand_dir, ("action.npy", "position.npy"))
    hand_action, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))

    hand_action = resample_to(hand_time, hand_action, arm_time)
    if args.hand == "inspire":
        hand_qpos = inspire_action_to_qpos(hand_action)
    else:
        hand_qpos = hand_action

    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

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
    obj_mesh = load_object_mesh(args.object_mesh)

    urdf_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}_left_new.urdf")
    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot("robot", urdf_path, c2r)
    vis.add_object(args.object_name, obj_mesh, obj_traj[0])
    vis.add_traj("traj", {"robot": qpos_video}, {args.object_name: obj_traj})
    vis.start_viewer()


if __name__ == "__main__":
    main()
