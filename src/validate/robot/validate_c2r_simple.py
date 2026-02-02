"""
Simple C2R (camera-to-robot) validation script.

This script validates C2R calibration by projecting the robot mesh onto
captured images from an existing episode.

Usage:
    python src/validate/robot/validate_c2r_simple.py --capture-root hri_inspire_left --object <object_name> --episode 0

Or with a specific capture path:
    python src/validate/robot/validate_c2r_simple.py --capture-path /path/to/capture/episode
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from paradex.calibration.utils import load_camparam, load_current_camparam, load_current_C2R
from paradex.visualization.robot import RobotModule
from paradex.robot.inspire import inspire_action_to_qpos
from paradex.image.projection import BatchRenderer
from paradex.image.overlay import overlay_mask
from paradex.image.grid import make_image_grid
from paradex.utils.path import rsc_path, shared_dir
from paradex.utils.load_data import load_series, resample_to


def load_images_from_episode(episode_path, frame_idx=0, data_type='video'):
    """Load images from video_extracted directory."""
    if data_type=='video':
        image_dir = os.path.join(episode_path, "video_extracted")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"video_extracted directory not found: {image_dir}")
        images = {}
        cam_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

        for cam_serial in cam_dirs:
            cam_dir = os.path.join(image_dir, cam_serial)
            img_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
            if not img_files:
                img_files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))

            if img_files and frame_idx < len(img_files):
                img_path = img_files[frame_idx]
                img = cv2.imread(img_path)
                if img is not None:
                    images[cam_serial] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print(f"  Loaded {cam_serial}: {img.shape} from {os.path.basename(img_path)}")
    else:
        image_dir = os.path.join(episode_path, "raw/images")
        images = {}
        for img_nm in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_nm)
            img = cv2.imread(img_path)
            if img is not None:
                cam_id = img_nm.split('_')[0]  # Assuming filename format: <cam_serial>_frameXXXX.jpg
                images[cam_id] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(f"  Loaded {cam_id}: {img.shape} from {img_nm}")

    return images


def load_robot_qpos_from_episode(episode_path, hand_name, frame_idx=0):
    """Load robot joint positions from recorded data."""
    raw_dir = os.path.join(episode_path, "raw")
    arm_dir = os.path.join(raw_dir, "arm")
    hand_dir = os.path.join(raw_dir, "hand")

    # Load arm qpos
    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))

    # Load hand action/position
    hand_action, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))

    # Resample hand to arm time
    hand_action = resample_to(hand_time, hand_action, arm_time)

    # Convert hand action to joint angles
    if hand_name == "inspire":
        hand_qpos = inspire_action_to_qpos(hand_action)
    else:
        hand_qpos = hand_action

    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

    # Get frame at index
    actual_idx = min(frame_idx, full_qpos.shape[0] - 1)
    print(f"  Total frames: {full_qpos.shape[0]}, using frame {actual_idx}")
    return full_qpos[actual_idx]


def validate_c2r_from_episode(
    capture_path,
    arm_name="xarm",
    hand_name="inspire",
    frame_idx=0,
    device="cuda:0",
    output_dir=None,
    data_type='video'
):
    """
    Validate C2R by projecting robot onto captured images.

    Args:
        capture_path: Path to capture episode directory
        arm_name: Arm type
        hand_name: Hand type
        frame_idx: Frame index to use
        device: CUDA device
        output_dir: Output directory for overlays
    """
    print("=" * 60)
    print("C2R Calibration Validation (from episode)")
    print("=" * 60)
    print(f"Episode path: {capture_path}")

    # Load camera parameters and C2R from episode
    print("\n[1] Loading calibration data...")
    intrinsic, extrinsic = load_camparam(capture_path)
    c2r_path = os.path.join(capture_path, "C2R.npy")
    if os.path.exists(c2r_path):
        c2r = np.load(c2r_path)
        print(f"    Loaded C2R from episode: {c2r_path}")
    else:
        c2r = load_current_C2R()
        print(f"    Using current system C2R")
    print(f"    Cameras: {list(intrinsic.keys())}")

    # Load images
    print(f"\n[2] Loading images (frame {frame_idx})...")
    images = load_images_from_episode(capture_path, frame_idx, data_type=data_type)
    if not images:
        raise RuntimeError("No images found!")

    # Load robot qpos
    print(f"\n[3] Loading robot pose...")
    full_qpos = load_robot_qpos_from_episode(capture_path, hand_name, frame_idx)
    print(f"    Arm qpos: {full_qpos[:6]}")
    print(f"    Hand qpos: {full_qpos[6:]}")

    # Load robot model
    print("\n[4] Loading robot URDF...")
    urdf_path = os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}_left_new.urdf")
    robot = RobotModule(urdf_path)
    robot.update_cfg(full_qpos)
    mesh = robot.get_robot_mesh()
    print(f"    Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Compute extrinsic with C2R
    print("\n[5] Setting up renderer...")
    filtered_intrinsic = {k: v for k, v in intrinsic.items() if k in images}
    extrinsic_c2r = {}
    for cam_id in filtered_intrinsic.keys():
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsic[cam_id]
        extr_full = cam_from_world @ c2r
        extrinsic_c2r[cam_id] = extr_full[:3, :]

    renderer = BatchRenderer(filtered_intrinsic, extrinsic_c2r)

    # Render robot
    print("\n[6] Rendering robot projection...")
    color_dict, mask_dict, depth_dict = renderer.render(mesh)

    # Create overlays
    print("\n[7] Creating overlay images...")
    if output_dir is None:
        output_dir = os.path.join(capture_path, "c2r_validation")
    os.makedirs(output_dir, exist_ok=True)

    overlays = []
    for cam_id in sorted(filtered_intrinsic.keys()):
        if cam_id not in images:
            continue

        image = images[cam_id]
        mask = mask_dict[cam_id].astype(np.float32)

        # Create overlay
        overlay = overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5)
        overlays.append(overlay)

        # Save individual
        out_path = os.path.join(output_dir, f"{cam_id}_overlay.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"    Saved: {out_path}")

    # Create grid
    if overlays:
        grid = make_image_grid(overlays)
        grid_path = os.path.join(output_dir, f"grid_frame{frame_idx:04d}.jpg")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\n    Grid: {grid_path}")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print(f"Output: {output_dir}")
    print("Green overlay should align with the robot in the image.")
    print("=" * 60)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Validate C2R from existing episode")
    parser.add_argument("--capture-path", type=str, help="Direct path to episode directory")
    parser.add_argument("--capture-root", type=str, default="hri_inspire_left",
                        help="Capture root name (e.g., hri_inspire_left)")
    parser.add_argument("--object", type=str, help="Object name")
    parser.add_argument("--episode", type=str, default="0", help="Episode number")
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data_type", type=str, choices=['video', 'images'], default='video',
                        help="Type of data storage: 'video' for video_extracted, 'raw' for raw/images")

    args = parser.parse_args()

    if args.capture_path:
        capture_path = args.capture_path
    elif args.object:
        capture_path = os.path.join(
            shared_dir, "capture", args.capture_root, args.object, str(args.episode)
        )
    else:
        parser.error("Either --capture-path or --object must be specified")

    if not os.path.exists(capture_path):
        raise FileNotFoundError(f"Capture path not found: {capture_path}")

    validate_c2r_from_episode(
        capture_path=capture_path,
        arm_name=args.arm,
        hand_name=args.hand,
        frame_idx=args.frame,
        device=args.device,
        output_dir=args.output_dir,
        data_type=args.data_type
    )


if __name__ == "__main__":
    main()
