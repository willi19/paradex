import os
import glob
import pickle
import numpy as np
import trimesh
from typing import Any, Optional

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
    if T.shape == (3, 4):
        padded = np.eye(4, dtype=float)
        padded[:3, :] = T
        return padded
    raise ValueError(f"Frame {idx} transform shape {T.shape} cannot be interpreted as 4x4")

def load_object_trajectory(track_path: str) -> np.ndarray:
    # Load pickle/npz/npy containing a sequence of 4x4 poses.
    if os.path.isdir(track_path):
        candidates = sorted(
            glob.glob(os.path.join(track_path, "*.pickle"))
            + glob.glob(os.path.join(track_path, "*.pkl"))
            + glob.glob(os.path.join(track_path, "*.npy"))
            + glob.glob(os.path.join(track_path, "*.npz"))
        )
        if not candidates:
            raise FileNotFoundError(f"No object trajectory found in {track_path}")
        path = candidates[0]
    else:
        path = track_path

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
    raise ValueError(f"Unexpected mesh type: {type(mesh)} from {mesh_path}")


def apply_transform(verts: np.ndarray, T: np.ndarray) -> np.ndarray:
    # Apply 4x4 transform to vertices (N,3) -> (N,3).
    homo = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=verts.dtype)], axis=1)
    transformed = (T @ homo.T).T
    return transformed[:, :3]