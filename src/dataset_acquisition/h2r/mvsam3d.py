"""Capture multi-camera images and estimate an object pose with MV-SAM3D.

The camera capture and projection run in the ``robot`` environment.  SAM3
segmentation and MV-SAM3D/Depth-Anything-3 run in separate conda environments
through subprocesses, so their CUDA/PyTorch dependencies never enter the robot
process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MVSAM3D_REPO = PROJECT_ROOT / "external" / "MV-SAM3D"


def _transform(value: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0.0, 0.0, 0.0, 1.0]))
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 3x4 or 4x4, got {matrix.shape}")
    return matrix


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(-1)[:4]
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        raise ValueError("MV-SAM3D returned a zero quaternion")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("[run]", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _conda_python(env_name: str, *arguments: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", env_name, "python", *arguments]


def _require_cuda(env_name: str) -> None:
    result = subprocess.run(
        _conda_python(
            env_name,
            "-c",
            "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)",
        ),
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"CUDA is not available in conda env '{env_name}'; "
            "MV-SAM3D requires a CUDA-visible NVIDIA GPU"
        )


def _segment_stage(argv: list[str]) -> None:
    """Hidden entry point executed inside the dedicated SAM3 environment."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)

    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        load_from_HF=args.checkpoint is None,
    )
    processor = Sam3Processor(model, confidence_threshold=args.confidence, device=device)
    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(Path(args.images).glob("*.png"), key=lambda path: int(path.stem))
    successful_views = []
    for image_path in image_paths:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        state = processor.set_image(Image.fromarray(rgb))
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=args.prompt)
        masks = state["masks"]
        if masks.numel() == 0:
            print(f"[sam3] {image_path.name}: no '{args.prompt}' found; skipping view")
            continue
        mask = masks.squeeze(1).any(dim=0).cpu().numpy()
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = mask.astype(np.uint8) * 255
        Image.fromarray(rgba).save(output_dir / image_path.name)
        successful_views.append(image_path.stem)
        print(f"[sam3] {image_path.name}: {int(mask.sum())} foreground pixels")

    if not successful_views:
        raise RuntimeError(f"SAM3 found no '{args.prompt}' in any captured view")
    print(
        f"[sam3] segmented {len(successful_views)}/{len(image_paths)} views; "
        "MV-SAM3D will use the successful views"
    )


def _capture(save_dir: Path, rcc_entry: str) -> None:
    from paradex.calibration.utils import save_current_camparam
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.utils.path import shared_dir

    save_dir.mkdir(parents=True, exist_ok=True)
    save_current_camparam(str(save_dir))
    relative = save_dir.resolve().relative_to(Path(shared_dir).resolve())
    controller = remote_camera_controller(rcc_entry)
    try:
        controller.start("image", False, f"shared_data/{relative.as_posix()}/raw")
        controller.stop()
    finally:
        controller.end()


def _prepare_scene(save_dir: Path, scene_dir: Path) -> list[str]:
    source_dir = save_dir / "raw" / "images"
    source_images = sorted(source_dir.glob("*.png"), key=lambda path: path.stem)
    if not source_images:
        raise FileNotFoundError(f"no captured PNG images in {source_dir}")
    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    camera_ids = []
    for index, source in enumerate(source_images):
        shutil.copy2(source, images_dir / f"{index}.png")
        camera_ids.append(source.stem)
    (scene_dir / "camera_order.json").write_text(
        json.dumps(camera_ids, indent=2), encoding="utf-8"
    )
    return camera_ids


def _new_result_dir(repo: Path, scene_name: str, mask_name: str, before: set[Path]) -> Path:
    root = repo / "visualization" / scene_name / mask_name
    candidates = [path for path in root.glob("*") if path.is_dir() and path not in before]
    if not candidates:
        candidates = [path for path in root.glob("*") if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"MV-SAM3D produced no result directory under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _load_mvsam3d_pose(result_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    optimized = result_dir / "pose_optimization" / "optimized_params.npz"
    params_path = optimized if optimized.is_file() else result_dir / "params.npz"
    if not params_path.is_file():
        raise FileNotFoundError(f"MV-SAM3D pose parameters not found in {result_dir}")
    with np.load(params_path, allow_pickle=True) as values:
        scale = np.asarray(values["scale"], dtype=np.float64).reshape(-1)
        rotation = np.asarray(values["rotation"], dtype=np.float64).reshape(-1)
        translation = np.asarray(values["translation"], dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    mesh_path = result_dir / ("result_pose_optimized.glb" if optimized.is_file() else "result.glb")
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    return scale[:3], rotation[:4], translation[:3], mesh_path


def _pose_in_calibrated_world(
    camera_from_world: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation_p3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MV-SAM3D's view-0 PyTorch3D pose to calibrated OpenCV/world frames."""

    p3d_to_cv = np.diag([-1.0, -1.0, 1.0])
    pose_camera = np.eye(4, dtype=np.float64)
    # MV-SAM3D applies its quaternion with row vectors (``points @ R``).
    # OpenCV uses column vectors, so the equivalent rotation is ``R.T``.
    pose_camera[:3, :3] = p3d_to_cv @ _quat_wxyz_to_matrix(rotation_wxyz).T
    pose_camera[:3, 3] = p3d_to_cv @ translation_p3d
    pose_world = np.linalg.inv(camera_from_world) @ pose_camera
    return pose_camera, pose_world


def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"no mesh geometry in {path}")
    return trimesh.util.concatenate(meshes)


def _write_projection_grid(
    save_dir: Path,
    mesh_path: Path,
    pose_world: np.ndarray,
    scale: np.ndarray,
) -> Path:
    import cv2
    from paradex.image.grid import make_image_grid

    intrinsics = json.loads((save_dir / "cam_param" / "intrinsics.json").read_text())
    extrinsics = json.loads((save_dir / "cam_param" / "extrinsics.json").read_text())
    mesh = _load_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Upstream applies this fixed transform before the learned pose.
    z_up_to_y_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    vertices = (vertices @ z_up_to_y_up.T) * scale.reshape(1, 3)
    if len(faces) > 4000:
        faces = faces[np.linspace(0, len(faces) - 1, 4000, dtype=np.int64)]

    overlays = []
    for camera_id in sorted(extrinsics):
        image_path = save_dir / "raw" / "images" / f"{camera_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or camera_id not in intrinsics:
            continue
        camera_from_world = _transform(extrinsics[camera_id], name=f"extrinsics[{camera_id}]")
        camera_from_object = camera_from_world @ pose_world
        rvec, _ = cv2.Rodrigues(camera_from_object[:3, :3])
        camera = intrinsics[camera_id]
        matrix = np.asarray(camera["original_intrinsics"], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(camera.get("dist_params", []), dtype=np.float64)
        projected, _ = cv2.projectPoints(
            vertices, rvec, camera_from_object[:3, 3], matrix, distortion
        )
        projected = np.rint(projected.reshape(-1, 2)).astype(np.int32)
        depth = vertices @ camera_from_object[:3, :3].T + camera_from_object[:3, 3]
        for face in faces:
            if np.all(depth[face, 2] > 0.01):
                cv2.polylines(image, [projected[face]], True, (0, 255, 255), 1, cv2.LINE_AA)

        axis = np.array([[0, 0, 0], [0.08, 0, 0], [0, 0.08, 0], [0, 0, 0.08]], np.float64)
        axis_2d, _ = cv2.projectPoints(
            axis, rvec, camera_from_object[:3, 3], matrix, distortion
        )
        origin, x_axis, y_axis, z_axis = np.rint(axis_2d.reshape(-1, 2)).astype(np.int32)
        for endpoint, color in ((x_axis, (0, 0, 255)), (y_axis, (0, 255, 0)), (z_axis, (255, 0, 0))):
            cv2.line(image, origin, endpoint, color, 3, cv2.LINE_AA)
        cv2.putText(image, camera_id, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        overlays.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not overlays:
        raise RuntimeError("no calibrated camera image could be projected")
    output = save_dir / "projection_grid.png"
    grid = make_image_grid(overlays)
    if not cv2.imwrite(str(output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {output}")
    return output


def main() -> None:
    from paradex.calibration.utils import load_current_C2R
    from paradex.utils.path import shared_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="SAM3 text prompt for the object")
    parser.add_argument("--save_path", default="capture/h2r/mvsam3d")
    parser.add_argument("--rcc_entry", default="image_main.py")
    parser.add_argument("--mvsam3d_repo", default=str(DEFAULT_MVSAM3D_REPO))
    parser.add_argument("--mvsam3d_env", default="sam3d-objects")
    parser.add_argument("--sam3_env", default="sam3")
    parser.add_argument("--gpu", default="1", help="physical CUDA GPU index")
    parser.add_argument("--sam3_checkpoint")
    parser.add_argument("--sam3_confidence", type=float, default=0.3)
    parser.add_argument("--skip_capture", action="store_true", help="reuse --capture_dir")
    parser.add_argument("--capture_dir", help="existing capture directory under shared_data")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    repo = Path(args.mvsam3d_repo).expanduser().resolve()
    if not (repo / "run_inference_weighted.py").is_file():
        raise FileNotFoundError(f"MV-SAM3D repository not found: {repo}")
    _require_cuda(args.mvsam3d_env)

    if args.skip_capture:
        if not args.capture_dir:
            raise ValueError("--skip_capture requires --capture_dir")
        capture_dir = Path(args.capture_dir).expanduser()
        save_dir = (
            capture_dir.resolve()
            if capture_dir.is_absolute()
            else (Path(shared_dir) / capture_dir).resolve()
        )
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(shared_dir) / args.save_path / stamp
        _capture(save_dir, args.rcc_entry)

    scene_dir = save_dir / f"mvsam3d_{save_dir.name}"
    camera_ids = _prepare_scene(save_dir, scene_dir)
    mask_name = "object"
    segment_command = _conda_python(
        args.sam3_env,
        str(Path(__file__).resolve()),
        "_segment",
        "--images", str(scene_dir / "images"),
        "--output", str(scene_dir / mask_name),
        "--prompt", args.prompt,
        "--confidence", str(args.sam3_confidence),
    )
    sam3_checkpoint = (
        Path(args.sam3_checkpoint).expanduser()
        if args.sam3_checkpoint
        else repo / "checkpoints" / "sam3.pt"
    )
    if sam3_checkpoint.is_file():
        segment_command.extend(("--checkpoint", str(sam3_checkpoint)))
    _run(segment_command)
    successful_view_names = [
        path.stem
        for path in sorted(
            (scene_dir / mask_name).glob("*.png"), key=lambda path: int(path.stem)
        )
    ]
    if not successful_view_names:
        raise RuntimeError(f"SAM3 produced no usable '{args.prompt}' masks")

    da3_dir = save_dir / "da3"
    _run(
        _conda_python(
            args.mvsam3d_env,
            "scripts/run_da3.py",
            "--image_dir", str(scene_dir / "images"),
            "--output_dir", str(da3_dir),
        ),
        cwd=repo,
    )

    result_root = repo / "visualization" / scene_dir.name / mask_name
    before = set(result_root.glob("*")) if result_root.is_dir() else set()
    _run(
        _conda_python(
            args.mvsam3d_env,
            "run_inference_weighted.py",
            "--input_path", str(scene_dir),
            "--mask_prompt", mask_name,
            "--da3_output", str(da3_dir / "da3_output.npz"),
            "--merge_da3_glb",
            "--overlay_pointmap",
            "--run_pose_optimization",
        ),
        cwd=repo,
    )
    result_dir = _new_result_dir(repo, scene_dir.name, mask_name, before)
    scale, rotation, translation, mesh_path = _load_mvsam3d_pose(result_dir)
    captured_mesh_path = save_dir / "mvsam3d_result.glb"
    shutil.copy2(mesh_path, captured_mesh_path)

    calibration = json.loads((save_dir / "cam_param" / "extrinsics.json").read_text())
    reference_view_index = int(successful_view_names[0])
    reference_camera = camera_ids[reference_view_index]
    camera_from_world = _transform(calibration[reference_camera], name="reference extrinsic")
    pose_camera, pose_world = _pose_in_calibrated_world(camera_from_world, rotation, translation)
    saved_c2r = save_dir / "C2R.npy"
    c2r_source = np.load(saved_c2r) if saved_c2r.is_file() else load_current_C2R()
    c2r = _transform(c2r_source, name="C2R")
    pose_robot = np.linalg.inv(c2r) @ pose_world
    np.save(save_dir / "C2R.npy", c2r)

    pose_output = {
        "pose_world": pose_world.tolist(),
        "pose_left_cam": pose_camera.tolist(),
        "R_world": pose_world[:3, :3].tolist(),
        "t_world": pose_world[:3, 3].tolist(),
        "scale": scale.tolist(),
        "rotation_wxyz_mvsam3d": rotation.tolist(),
        "translation_mvsam3d_pytorch3d": translation.tolist(),
        "reference_camera": reference_camera,
        "canonical_frame": "MV-SAM3D Y-up canonical frame",
        "generated_mesh": str(captured_mesh_path),
        "result_dir": str(result_dir),
    }
    (save_dir / "object_6d.json").write_text(json.dumps(pose_output, indent=2), encoding="utf-8")
    robot_output = dict(pose_output, pose_robot=pose_robot.tolist())
    (save_dir / "object_6d_robot.json").write_text(json.dumps(robot_output, indent=2), encoding="utf-8")
    grid_path = _write_projection_grid(save_dir, captured_mesh_path, pose_world, scale)
    print(f"saved pose: {save_dir / 'object_6d.json'}")
    print(f"saved projection grid: {grid_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_segment":
        _segment_stage(sys.argv[2:])
    else:
        main()
