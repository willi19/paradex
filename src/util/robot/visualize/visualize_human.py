import argparse
import os
import re
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import trimesh
import viser

MANO_HAND_COLOR = np.array([180, 180, 180, 255], dtype=np.uint8)


def list_hand_frame_ids(hand_mesh_dir: str) -> List[int]:
    frame_ids: List[int] = []
    for name in os.listdir(hand_mesh_dir):
        if not name.lower().endswith(".obj"):
            continue
        stem = os.path.splitext(name)[0]
        if re.fullmatch(r"\d+", stem):
            frame_ids.append(int(stem))
    return sorted(frame_ids)


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

    # Case 1: key names like frame_0, frame_1, ...
    for key in data.files:
        m = re.fullmatch(r"frame_(\d+)", key)
        if m:
            idx = int(m.group(1))
            arr = np.asarray(data[key])
            if arr.shape == (4, 4):
                out[idx] = arr.astype(np.float64)

    # Case 2: stacked array under common keys.
    for key in ("obj_T_frames", "arr_0"):
        if key not in data.files:
            continue
        arr = np.asarray(data[key])
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            for i in range(arr.shape[0]):
                out.setdefault(i, arr[i].astype(np.float64))

    return out


def get_object_transform(
    object_tracking_dir: str,
    npz_lookup: Dict[int, np.ndarray],
    object_frame_idx: int,
) -> Optional[np.ndarray]:
    t_npy = try_load_transform_from_npy(object_tracking_dir, object_frame_idx)
    if t_npy is not None:
        return t_npy
    return npz_lookup.get(object_frame_idx, None)


def load_c2r(base_path: str) -> np.ndarray:
    c2r_path = os.path.join(base_path, "C2R.npy")
    if not os.path.exists(c2r_path):
        print(f"[WARN] C2R not found at {c2r_path}; using identity.")
        return np.eye(4, dtype=np.float64)

    c2r = np.asarray(np.load(c2r_path, allow_pickle=True), dtype=np.float64)
    if c2r.shape == (3, 4):
        c2r_h = np.eye(4, dtype=np.float64)
        c2r_h[:3, :] = c2r
        c2r = c2r_h
    if c2r.shape != (4, 4):
        raise ValueError(f"Invalid C2R shape at {c2r_path}: {c2r.shape}. Expected (4,4) or (3,4).")
    return c2r


def main():
    parser = argparse.ArgumentParser(description="Viser viewer for hand mesh + object tracking transforms")
    parser.add_argument("--base_path", type=str, required=True, help="Base path containing hand_mesh_single and object_tracking_result")
    parser.add_argument(
        "--mano_dir_name",
        type=str,
        default="hand_mesh_single",
        help="Folder name under base_path that contains MANO hand meshes",
    )
    parser.add_argument("--object_mesh_path", type=str, required=True, help="Object mesh path (.obj/.ply/.glb...)")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--start_frame", type=int, default=-1, help="Optional hand frame start (inclusive), <0 means auto")
    parser.add_argument("--end_frame", type=int, default=-1, help="Optional hand frame end (inclusive), <0 means auto")
    parser.add_argument(
        "--hand_to_object_offset",
        type=int,
        default=1,
        help="object_frame_idx = hand_frame_idx - offset (default: 1, so hand 1 -> object 0)",
    )
    parser.add_argument("--render_width", type=int, default=1920, help="rendered image width")
    parser.add_argument("--render_height", type=int, default=1080, help="rendered image height")
    args = parser.parse_args()

    hand_mesh_dir = os.path.join(args.base_path, args.mano_dir_name)
    object_tracking_dir = os.path.join(args.base_path, "object_tracking_result")
    npz_path = os.path.join(object_tracking_dir, "obj_T_frames.npz")

    if not os.path.isdir(hand_mesh_dir):
        raise FileNotFoundError(f"hand mesh folder not found: {hand_mesh_dir}")

    hand_frame_ids = list_hand_frame_ids(hand_mesh_dir)
    if len(hand_frame_ids) == 0:
        raise RuntimeError(f"No numeric .obj files found in: {hand_mesh_dir}")

    if args.start_frame >= 0:
        hand_frame_ids = [f for f in hand_frame_ids if f >= args.start_frame]
    if args.end_frame >= 0:
        hand_frame_ids = [f for f in hand_frame_ids if f <= args.end_frame]
    if len(hand_frame_ids) == 0:
        raise RuntimeError("No hand frames left after start/end filtering.")

    has_object_tracking = os.path.isdir(object_tracking_dir)
    has_object_mesh = os.path.exists(args.object_mesh_path)
    enable_object = has_object_tracking and has_object_mesh
    if not has_object_tracking:
        print(f"[WARN] object tracking folder not found: {object_tracking_dir}; showing hand only.")
    if not has_object_mesh:
        print(f"[WARN] object mesh not found: {args.object_mesh_path}; showing hand only.")

    npz_lookup = build_npz_transform_lookup(npz_path) if enable_object else {}
    c2r = load_c2r(args.base_path)
    object_mesh = None
    if enable_object:
        object_mesh = trimesh.load(args.object_mesh_path, force="mesh", process=False)
        if not isinstance(object_mesh, trimesh.Trimesh):
            object_mesh = trimesh.util.concatenate(tuple(object_mesh.geometry.values()))

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(dark_mode=False)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/ground",
        width=6.0,
        height=6.0,
        width_segments=24,
        height_segments=24,
        plane="xy",
    )
    render_dir = os.path.join(args.base_path, "viser_renders")
    os.makedirs(render_dir, exist_ok=True)

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        client.camera.near = 0.01
        client.camera.far = 100.0
        client.camera.look_speed = 0.2
        client.camera.move_speed = 0.05

    frame_nodes = []
    frame_mesh_handles: List[List[object]] = []
    object_transforms_by_frame: List[Optional[np.ndarray]] = []
    for hand_frame_idx in hand_frame_ids:
        frame_node = server.scene.add_frame(f"/frames/f{hand_frame_idx}", show_axes=False, visible=False)
        handles_this_frame: List[object] = []
        hand_mesh_path = os.path.join(hand_mesh_dir, f"{hand_frame_idx:05d}.obj")
        hand_mesh = trimesh.load(hand_mesh_path, force="mesh", process=False)
        if not isinstance(hand_mesh, trimesh.Trimesh):
            hand_mesh = trimesh.util.concatenate(tuple(hand_mesh.geometry.values()))
        hand_mesh.apply_transform(c2r)
        hand_mesh.visual = trimesh.visual.ColorVisuals(
            mesh=hand_mesh,
            vertex_colors=np.tile(MANO_HAND_COLOR, (len(hand_mesh.vertices), 1)),
        )

        hand_handle = server.scene.add_mesh_trimesh(
            name=f"/frames/f{hand_frame_idx}/hand",
            mesh=hand_mesh,
        )
        handles_this_frame.append(hand_handle)

        T_obj = None
        if enable_object:
            object_frame_idx = hand_frame_idx - args.hand_to_object_offset
            T_obj = get_object_transform(object_tracking_dir, npz_lookup, object_frame_idx)
            if T_obj is not None:
                T_obj = c2r @ T_obj
        object_transforms_by_frame.append(T_obj)
        frame_nodes.append(frame_node)
        frame_mesh_handles.append(handles_this_frame)

    object_frame = None
    if enable_object and object_mesh is not None:
        object_frame = server.scene.add_frame("/object", show_axes=False, visible=False)
        server.scene.add_mesh_trimesh(
            name="/object/mesh",
            mesh=object_mesh,
        )

    def set_object_pose(frame_list_idx: int) -> None:
        if object_frame is None:
            return
        T_obj = object_transforms_by_frame[frame_list_idx]
        if T_obj is None:
            object_frame.visible = False
            return
        object_frame.visible = True
        object_frame.position = T_obj[:3, 3]
        object_frame.wxyz = trimesh.transformations.quaternion_from_matrix(T_obj)

    current_i = 0
    frame_nodes[current_i].visible = True
    for h in frame_mesh_handles[current_i]:
        h.visible = True
    set_object_pose(current_i)

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(frame_nodes) - 1,
            step=1,
            initial_value=0,
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
        gui_prev = server.gui.add_button("Prev")
        gui_next = server.gui.add_button("Next")
        gui_render_png = server.gui.add_checkbox("Render to PNG", False)
        gui_render_current = server.gui.add_button("Render Current")
        gui_render_all = server.gui.add_button("Render All")

    def set_visible_frame(next_i: int) -> None:
        nonlocal current_i
        if next_i == current_i:
            return
        with server.atomic():
            frame_nodes[current_i].visible = False
            for h in frame_mesh_handles[current_i]:
                h.visible = False
            frame_nodes[next_i].visible = True
            for h in frame_mesh_handles[next_i]:
                h.visible = True
            set_object_pose(next_i)
        current_i = next_i

    @gui_timestep.on_update
    def _(_evt) -> None:
        next_i = int(gui_timestep.value)
        set_visible_frame(next_i)

    @gui_prev.on_click
    def _(_evt) -> None:
        gui_timestep.value = (int(gui_timestep.value) - 1) % len(frame_nodes)

    @gui_next.on_click
    def _(_evt) -> None:
        gui_timestep.value = (int(gui_timestep.value) + 1) % len(frame_nodes)

    def render_one_frame(frame_list_idx: int) -> None:
        gui_timestep.value = frame_list_idx
        time.sleep(0.03)
        clients = list(server.get_clients().values())
        if len(clients) == 0:
            print("[WARN] no connected client; render skipped.")
            return
        client = clients[0]
        rendered_img = client.get_render(height=args.render_height, width=args.render_width)
        hand_frame_idx = hand_frame_ids[frame_list_idx]
        out_path = os.path.join(render_dir, f"{hand_frame_idx:05d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))

    @gui_render_current.on_click
    def _(_evt) -> None:
        render_one_frame(int(gui_timestep.value))
        print(f"[INFO] rendered current frame -> {render_dir}")

    @gui_render_all.on_click
    def _(_evt) -> None:
        was_playing = bool(gui_playing.value)
        gui_playing.value = False
        for i in range(len(frame_nodes)):
            render_one_frame(i)
        gui_playing.value = was_playing
        print(f"[INFO] rendered all frames -> {render_dir}")

    print(f"[INFO] Viser running on port {args.port}")
    print(f"[INFO] hand frames: {hand_frame_ids[0]}..{hand_frame_ids[-1]} (count={len(hand_frame_ids)})")
    if enable_object:
        print(f"[INFO] mapping: object_frame = hand_frame - {args.hand_to_object_offset}")
    else:
        print("[INFO] object trajectory disabled.")
    print(f"[INFO] using C2R from: {os.path.join(args.base_path, 'C2R.npy')}")
    print(f"[INFO] open: http://localhost:{args.port}")

    while True:
        if gui_playing.value:
            gui_timestep.value = (int(gui_timestep.value) + 1) % len(frame_nodes)
            if gui_render_png.value:
                render_one_frame(int(gui_timestep.value))
        time.sleep(1.0 / max(1, int(gui_fps.value)))


if __name__ == "__main__":
    main()