import argparse
import pickle
from typing import Any, Optional

import numpy as np
import trimesh
import viser
import time

from paradex.io.robot_controller.under_test.visualize_saved_tactile_with_object import BASE_PATH

try:
    import torch
except ImportError:  # torch is only needed if the pickle stores tensors
    torch = None

import os

def to_numpy_array(x: Any) -> np.ndarray:
    """Convert numpy/torch inputs to numpy on CPU."""
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "detach"):
        return np.asarray(x.detach())
    return np.asarray(x)


def _maybe_stack_array(payload: Any) -> Optional[np.ndarray]:
    """Handle payloads that are already stacked arrays."""
    if isinstance(payload, (list, tuple)):
        return None
    arr = to_numpy_array(payload)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr.astype(float)
    return None


def _build_T_from_frame(frame: Any, idx: int) -> np.ndarray:
    """Extract or build a 4x4 transform from a single frame entry."""
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


def load_trajectory(pickle_path: str) -> np.ndarray:
    """
    Load a trajectory pickle and return [T, 4, 4].
    Supports:
      - list/tuple of frames, each frame with 'T' or 'obj_R'/'obj_t'
      - dict with 'frames' or already-stacked array under keys like 'T', 'poses', 'traj', 'trajectory'
      - dict of frame_index -> frame (sorted by key)
    """
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)

    # Directly stacked tensor/array
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

    # Torch tensor or numpy array with shape (T,4,4)
    if torch is not None and isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(float)
    if isinstance(frames, np.ndarray):
        arr = np.asarray(frames)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(float)

    T_list = []
    for idx, frame in enumerate(frames):
        T_list.append(_build_T_from_frame(frame, idx))

    if not T_list:
        raise ValueError("No transforms loaded from trajectory pickle")

    return np.stack(T_list, axis=0)


def load_mesh(mesh_path: Optional[str]) -> trimesh.Trimesh:
    """Load mesh if provided; otherwise build a small sphere placeholder."""
    if mesh_path:
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        if isinstance(mesh, list):
            return trimesh.util.concatenate(mesh)
        raise ValueError(f"Unexpected mesh type: {type(mesh)}")

    # Fallback sphere (2 cm radius) with neutral color.
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.02)
    mesh.visual.vertex_colors = np.tile(
        np.array([200, 200, 200, 255], dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
    return mesh


def main():
    
    parser = argparse.ArgumentParser(
        description="Render an object trajectory (poses in a pickle) in Viser."
    )
    parser.add_argument(
        "--trajectory",
        required=True,
        help="Path to trajectory pickle containing per-frame transforms.",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="Optional path to mesh (obj/stl/ply/glb). If omitted, a small sphere is used.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback FPS for auto-play.",
    )
    args = parser.parse_args()

    object_seq = load_trajectory(args.trajectory)
    object_mesh_template = load_mesh(args.mesh)

    # Ensure the mesh is colored; if not, paint it light gray.
    if not hasattr(object_mesh_template.visual, "vertex_colors") or len(
        object_mesh_template.visual.vertex_colors
    ) == 0:
        object_mesh_template.visual.vertex_colors = np.tile(
            np.array([200, 200, 200, 255], dtype=np.uint8),
            (len(object_mesh_template.vertices), 1),
        )

    T = len(object_seq)
    server = viser.ViserServer()
    object_frame = server.scene.add_frame(
        "/object_frame",
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=True,
        axes_length=0.03,
        axes_radius=0.001,
    )
    object_handle = server.scene.add_mesh_trimesh(
        "/object_frame/object", object_mesh_template
    )

    # Add a floor grid for spatial reference.
    server.scene.add_grid(
        name="/floor",
        width=2.0,
        height=2.0,
        cell_size=0.05,
        position=(0.0, 0.0, 0.0),
    )
    # World axes (X=red, Y=green, Z=blue) for orientation reference.
    world_axes = trimesh.creation.axis(origin_size=0.02, axis_length=0.15)
    server.scene.add_mesh_trimesh("/world_axes", world_axes)

    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider("Frame", 0, T - 1, 1, 0)
        gui_play = server.gui.add_checkbox("Play", True)
        gui_fps = server.gui.add_slider("FPS", 1, 120, 1, args.fps)

    c2r = np.load(os.path.join(BASE_PATH, "C2R.npy"))
    r2c = np.linalg.inv(c2r)

    def render_frame(t: int):
        T_obj = c2r @ object_seq[t]
        object_frame.position = T_obj[:3, 3]
        object_frame.wxyz = trimesh.transformations.quaternion_from_matrix(T_obj)

    render_frame(0)

    @gui_frame.on_update
    def _(_):
        render_frame(gui_frame.value)

    while True:
        if gui_play.value:
            gui_frame.value = (gui_frame.value + 1) % T
            render_frame(gui_frame.value)
        time.sleep(1.0 / gui_fps.value)


if __name__ == "__main__":
    main()
