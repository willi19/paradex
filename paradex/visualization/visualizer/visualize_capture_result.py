"""Show reconstructed hands, object, and calibrated cameras in Viser.

The hand MANO meshes in a capture directory are already expressed in the
capture/world frame. Object world poses are read from SAM3 tracking output,
while ``cam_param/extrinsics.json`` stores OpenCV world-to-camera matrices.
This viewer converts only the camera poses before adding them to Viser.

Example:
    python -m paradex.visualization.visualizer.visualize_capture_result

    python -m paradex.visualization.visualizer.visualize_capture_result \
        --capture-dir /path/to/capture --frame 120
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import cv2
from scipy.spatial.transform import Rotation as R

from paradex.visualization.visualizer.viser_capture import CaptureViserViewer


DEFAULT_CAPTURE_DIR = Path(
    "/home/temp_id/shared_data/capture/hand_over/seongho/bimanual/"
    "circular_frying_pan/2"
)
HAND_COLORS = {
    "left": np.array([224, 194, 159, 255], dtype=np.uint8),
    "right": np.array([205, 169, 127, 255], dtype=np.uint8),
}
OBJECT_FALLBACK_COLORS = {
    "orange": np.array([242, 137, 38, 255], dtype=np.uint8),
}


@dataclass
class TrackedObject:
    """Mesh and synchronized world poses for one reconstructed object."""

    name: str
    mesh: trimesh.Trimesh
    poses: dict[int, np.ndarray]
    mesh_path: Path
    double_sided: bool


def _load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh while accepting OBJ files that trimesh reads as a Scene."""
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh contains no geometry: {path}")
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type {type(loaded).__name__}: {path}")
    return loaded


def _colored_mesh(path: Path, color: np.ndarray) -> trimesh.Trimesh:
    mesh = _load_mesh(path).copy()
    mesh.visual.vertex_colors = np.broadcast_to(color, (len(mesh.vertices), 4)).copy()
    return mesh


def _hand_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = _load_mesh(path)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32)


def _homogeneous(extrinsic: Any) -> np.ndarray:
    matrix = np.asarray(extrinsic, dtype=float)
    if matrix.shape == (3, 4):
        return np.vstack((matrix, np.array([0.0, 0.0, 0.0, 1.0])))
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"Expected a 3x4 or 4x4 extrinsic, received {matrix.shape}")


def _hand_frames(capture_dir: Path, side: str) -> dict[int, Path]:
    # SAM3 reprocessing moves the synchronized MANO sequence here. Preserve a
    # fallback for captures produced before that layout change.
    hand_root = capture_dir / "hand_temp"
    if not hand_root.is_dir():
        hand_root = capture_dir / "hand"
    mesh_dir = hand_root / side / "mano"
    frames = {int(path.stem): path for path in mesh_dir.glob("*.obj") if path.stem.isdigit()}
    if not frames:
        raise FileNotFoundError(f"No MANO meshes found in {mesh_dir}")
    return frames


def _resolve_object_mesh(
    capture_dir: Path,
    result: dict[str, Any],
    explicit_path: Path | None,
    foundpose_dir: Path,
) -> tuple[Path, float]:
    if explicit_path is not None:
        if explicit_path.is_file():
            return explicit_path, 1.0
        raise FileNotFoundError(f"Object mesh specified by --object-mesh does not exist: {explicit_path}")

    object_name = result.get("object_name")
    shared_data = next((parent for parent in capture_dir.parents if parent.name == "shared_data"), None)
    # GoTrack poses use the meter-scaled Blender source mesh, not FoundPose's
    # millimetre PLY asset.  Prefer that canonical mesh even if result.json
    # contains the stale path from the machine that produced the capture.
    candidates: list[Path] = []
    if shared_data is not None and object_name:
        candidates.append(shared_data / "mesh_blender" / object_name / f"{object_name}.obj")
    if result.get("mesh"):
        candidates.append(Path(result["mesh"]))

    for candidate in candidates:
        if candidate.is_file():
            return candidate, 1.0

    # Fall back to the capture-local FoundPose asset when the source mesh is
    # unavailable. It is self-contained but stored in millimetres.
    local_ply = (
        capture_dir
        / foundpose_dir.name
        / "assets"
        / str(result.get("object_name", ""))
        / "models"
        / "obj_000001.ply"
    )
    if local_ply.is_file():
        return local_ply, 1e-3

    searched = "\n  ".join(str(candidate) for candidate in candidates) or "(none)"
    raise FileNotFoundError(f"Could not locate the reconstructed object mesh. Searched:\n  {searched}")


def _load_sam3_object_poses(capture_dir: Path, object_name: str) -> dict[int, np.ndarray]:
    """Load SAM3's per-frame object-to-world transforms."""
    pose_path = (
        capture_dir
        / f"sam3_seed_tracking_output_{object_name}"
        / "all_poses_world.npz"
    )
    if not pose_path.is_file():
        raise FileNotFoundError(f"SAM3 poses not found for {object_name}: {pose_path}")

    poses: dict[int, np.ndarray] = {}
    with np.load(pose_path) as archive:
        for key in archive.files:
            if not key.startswith("frame_"):
                continue
            pose = np.asarray(archive[key], dtype=float)
            if pose.shape == (4, 4):
                poses[int(key.removeprefix("frame_"))] = pose
    if not poses:
        raise ValueError(f"SAM3 pose archive has no 4x4 frame poses: {pose_path}")
    return poses


def _load_tracked_objects(capture_dir: Path, explicit_pan_mesh: Path | None) -> list[TrackedObject]:
    """Load every FoundPose object that has a reconstructed mesh in the capture."""
    foundpose_dirs = sorted(
        (path.parent for path in capture_dir.glob("foundpose_*/result.json")),
        key=lambda path: (path.name != "foundpose_pan", path.name),
    )
    tracked_objects: list[TrackedObject] = []
    for foundpose_dir in foundpose_dirs:
        with (foundpose_dir / "result.json").open(encoding="utf-8") as file:
            result = json.load(file)
        object_name = str(result["object_name"])
        mesh_path, mesh_scale = _resolve_object_mesh(
            capture_dir,
            result,
            explicit_pan_mesh if foundpose_dir.name == "foundpose_pan" else None,
            foundpose_dir,
        )
        mesh = _load_mesh(mesh_path).copy()
        mesh.apply_scale(mesh_scale)
        trimesh.repair.fix_normals(mesh, multibody=True)
        if object_name in OBJECT_FALLBACK_COLORS:
            mesh.visual.vertex_colors = np.broadcast_to(
                OBJECT_FALLBACK_COLORS[object_name], (len(mesh.vertices), 4)
            ).copy()

        poses = _load_sam3_object_poses(capture_dir, object_name)
        tracked_objects.append(
            TrackedObject(
                name=object_name,
                mesh=mesh,
                poses=poses,
                mesh_path=mesh_path,
                # The pan needs double-sided rendering; orange is intentionally
                # given a visible semantic colour instead of its source texture.
                double_sided=(
                    object_name in OBJECT_FALLBACK_COLORS
                    or getattr(mesh.visual, "kind", None) != "texture"
                ),
            )
        )
    if not tracked_objects:
        raise FileNotFoundError(f"No FoundPose results found in {capture_dir}")
    return tracked_objects


def _load_up_direction(capture_dir: Path) -> np.ndarray:
    """Use the recorded camera-to-robot calibration to orient the Viser view."""
    c2r_path = capture_dir / "C2R.npy"
    if not c2r_path.is_file():
        return np.array([0.0, 0.0, 1.0])

    c2r = np.asarray(np.load(c2r_path), dtype=float)
    if c2r.shape != (4, 4):
        return np.array([0.0, 0.0, 1.0])
    # The grid is always the world XY plane. Keep the camera orbit's up axis
    # parallel to that grid instead of inheriting the calibration's small tilt.
    # This capture uses -Z as visual up.
    return np.array([0.0, 0.0, 1.0 if c2r[2, 2] >= 0.0 else -1.0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a capture's MANO hands, FoundPose object, and calibrated cameras."
    )
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Initial zero-based synchronized capture frame. Change it later with the Viser Capture/Frame slider.",
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        help="Override the object mesh path when the capture asset is unavailable.",
    )
    parser.add_argument("--camera-size", type=float, default=0.12)
    parser.add_argument(
        "--floor-height",
        type=float,
        default=None,
        help="Grid Z height. Defaults to the lowest point of the first-frame object mesh.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Capture playback speed.")
    parser.add_argument(
        "--render-output",
        type=Path,
        help="MP4 path written by the Render Video button (default: <capture-dir>/rendered_viser.mp4).",
    )
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--paused", action="store_true", help="Start with capture playback paused.")
    parser.add_argument("--no-cameras", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    capture_dir = args.capture_dir.expanduser().resolve()
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"Capture directory does not exist: {capture_dir}")

    left_frames = _hand_frames(capture_dir, "left")
    right_frames = _hand_frames(capture_dir, "right")
    tracked_objects = _load_tracked_objects(capture_dir, args.object_mesh)
    primary_object = next(
        (tracked for tracked in tracked_objects if tracked.name == "circular_frying_pan"),
        tracked_objects[0],
    )
    shared_frames = set.intersection(*(set(tracked.poses) for tracked in tracked_objects))
    frame_indices = sorted(shared_frames)
    if not frame_indices:
        raise ValueError("The tracked objects do not share any synchronized frames.")
    if args.frame not in shared_frames:
        raise ValueError(f"Frame {args.frame} is unavailable. Valid range is {frame_indices[0]}-{frame_indices[-1]}.")

    up_direction = _load_up_direction(capture_dir)
    # Keep the grid in the existing XY orientation. In this capture +up is
    # approximately -Z, so the visual bottom of the pan is its *maximum*
    # world-Z value (not its minimum).
    first_object_pose = primary_object.poses[frame_indices[0]]
    first_object_vertices = trimesh.transform_points(primary_object.mesh.vertices, first_object_pose)
    z_values = first_object_vertices[:, 2]
    auto_floor_height = float(z_values.min() if up_direction[2] >= 0.0 else z_values.max())
    floor_height = auto_floor_height if args.floor_height is None else args.floor_height

    viewer = CaptureViserViewer(
        up_direction=up_direction,
        scene_title=f"{capture_dir.name}: hands + object + cameras",
    )
    viewer.add_floor(
        height=floor_height,
        cell_color=(230, 235, 229),
        section_color=(189, 198, 181),
    )
    viewer.gui_playing.value = False

    for tracked in tracked_objects:
        viewer.add_object(
            tracked.name,
            tracked.mesh,
            tracked.poses[args.frame],
            show_axes=False,
            double_sided=tracked.double_sided,
        )

    hand_handles: dict[str, Any] = {}

    def show_frame(frame_index: int) -> None:
        # Hand OBJ filenames start at one, while GoTrack records start at zero.
        hand_frame = frame_index + 1
        for side, paths in (("left", left_frames), ("right", right_frames)):
            old_handle = hand_handles.get(side)
            if hand_frame not in paths:
                if old_handle is not None:
                    old_handle.visible = False
                continue
            vertices, faces = _hand_geometry(paths[hand_frame])
            if old_handle is None:
                hand_handles[side] = viewer.server.scene.add_mesh_simple(
                    name=f"/hands/{side}/mano_mesh",
                    vertices=vertices,
                    faces=faces,
                    color=tuple(int(component) for component in HAND_COLORS[side][:3]),
                    side="double",
                )
            else:
                if old_handle.faces.shape != faces.shape or not np.array_equal(old_handle.faces, faces):
                    raise ValueError(f"MANO topology changes at {paths[hand_frame]}")
                old_handle.vertices = vertices
                old_handle.visible = True

        for tracked in tracked_objects:
            object_frame = viewer.obj_dict[tracked.name]["frame"]
            object_transform = tracked.poses[frame_index]
            object_frame.position = object_transform[:3, 3]
            xyzw = R.from_matrix(object_transform[:3, :3]).as_quat()
            object_frame.wxyz = xyzw[[3, 0, 1, 2]]
        viewer.server.flush()

    show_frame(args.frame)

    if not args.no_cameras:
        with (capture_dir / "cam_param" / "intrinsics.json").open(encoding="utf-8") as file:
            intrinsics = json.load(file)
        with (capture_dir / "cam_param" / "extrinsics.json").open(encoding="utf-8") as file:
            extrinsics = json.load(file)

        for serial, camera_from_world in extrinsics.items():
            if serial not in intrinsics:
                print(f"Skipping camera {serial}: no intrinsic parameters.")
                continue
            viewer.add_camera(
                name=serial,
                extrinsic=np.linalg.inv(_homogeneous(camera_from_world)),
                intrinsic=intrinsics[serial],
                color=(86, 93, 88),
                size=args.camera_size,
                show_axes=False,
            )

    @viewer.server.on_client_connect
    def _set_reference_view(client) -> None:
        focus = primary_object.poses[args.frame][:3, 3]
        view_direction = np.array([-1.0, -1.0, 0.0])
        view_direction -= up_direction * np.dot(view_direction, up_direction)
        view_direction /= np.linalg.norm(view_direction)
        client.camera.position = focus + 2.5 * view_direction + 1.8 * up_direction
        client.camera.look_at = focus
        client.camera.up_direction = up_direction
        client.camera.fov = 0.9

    with viewer.server.gui.add_folder("Capture", expand_by_default=True):
        frame_slider = viewer.server.gui.add_slider(
            "Frame",
            min=frame_indices[0],
            max=frame_indices[-1],
            step=1,
            initial_value=args.frame,
        )
        capture_playing = viewer.server.gui.add_checkbox("Play", not args.paused)
        capture_fps = viewer.server.gui.add_slider(
            "FPS", min=1.0, max=60.0, step=1.0, initial_value=args.fps
        )
        render_button = viewer.server.gui.add_button("Render Video", icon="movie")
        render_status = viewer.server.gui.add_text("Render Status", "Ready", disabled=True)

    frame_positions = {frame: position for position, frame in enumerate(frame_indices)}
    current_frame = args.frame
    scene_lock = threading.RLock()

    def set_frame(frame: int, *, update_slider: bool) -> None:
        nonlocal current_frame
        with scene_lock:
            if frame == current_frame:
                return
            show_frame(frame)
            current_frame = frame
            if update_slider:
                frame_slider.value = frame

    @frame_slider.on_update
    def _(_) -> None:
        frame = int(frame_slider.value)
        if frame in shared_frames:
            set_frame(frame, update_slider=False)

    @render_button.on_click
    def _render_video(event) -> None:
        """Render the complete sequence from the click-time browser camera pose."""
        client = event.client
        if client is None or render_button.disabled:
            return

        render_button.disabled = True
        was_playing = capture_playing.value
        starting_frame = current_frame
        capture_playing.value = False
        render_status.value = "Rendering…"
        output_path = (args.render_output or capture_dir / "rendered_viser.mp4").expanduser()
        writer: cv2.VideoWriter | None = None

        # Snapshot the user-selected viewpoint once; changing frames must not
        # change the perspective of the output video.
        camera_position = np.asarray(client.camera.position, dtype=float).copy()
        camera_wxyz = np.asarray(client.camera.wxyz, dtype=float).copy()
        camera_fov = float(client.camera.fov)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with scene_lock:
                for position, frame in enumerate(frame_indices, start=1):
                    set_frame(frame, update_slider=True)
                    image = client.get_render(
                        height=args.render_height,
                        width=args.render_width,
                        position=camera_position,
                        wxyz=camera_wxyz,
                        fov=camera_fov,
                        transport_format="jpeg",
                    )
                    if writer is None:
                        height, width = image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(capture_fps.value),
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open video writer: {output_path}")
                    writer.write(cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR))
                    render_status.value = f"Rendering {position}/{len(frame_indices)}"
            render_status.value = f"Saved: {output_path}"
        except Exception as error:
            render_status.value = f"Render failed: {error}"
            raise
        finally:
            if writer is not None:
                writer.release()
            with scene_lock:
                set_frame(starting_frame, update_slider=True)
            capture_playing.value = was_playing
            render_button.disabled = False

    # Send static scene nodes (floor, object, hands, cameras, and controls)
    # immediately. This is necessary when the viewer starts paused because no
    # playback update will otherwise trigger a later flush.
    viewer.server.flush()
    print(
        f"Serving {capture_dir} with {len(frame_indices)} synchronized frames, "
        f"{len(tracked_objects)} object meshes ({', '.join(tracked.mesh_path.name for tracked in tracked_objects)}), and "
        f"{0 if args.no_cameras else len(extrinsics)} camera frustums."
    )
    try:
        while True:
            if capture_playing.value:
                next_frame = frame_indices[(frame_positions[current_frame] + 1) % len(frame_indices)]
                set_frame(next_frame, update_slider=True)
            time.sleep(1.0 / float(capture_fps.value))
    except KeyboardInterrupt:
        viewer.server.stop()


if __name__ == "__main__":
    main()
