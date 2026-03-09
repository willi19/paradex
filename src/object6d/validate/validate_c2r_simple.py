"""
Validate C2R calibration by projecting robot with zero qpos onto existing images.

Usage:
    python src/validate/robot/validate_c2r.py --scene-path /path/to/capture/episode
"""
import argparse
import glob
import os

import cv2
import numpy as np

from paradex.calibration.utils import load_camparam
from paradex.visualization.robot import RobotModule
from paradex.image.projection import BatchRenderer
from paradex.image.overlay import overlay_mask
from paradex.image.image_dict import ImageDict
from paradex.robot.inspire import parse_inspire
import math
from typing import List
import numpy as np



def make_image_grid(images: List[np.ndarray]) -> np.ndarray:
    # Tile images (RGB) into a nearly square grid.
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    base_h, base_w = images[0].shape[:2]
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    grid = np.zeros((rows * base_h, cols * base_w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        if img.shape[:2] != (base_h, base_w):
            import cv2

            img = cv2.resize(img, (base_w, base_h))
        r, c = divmod(idx, cols)
        grid[r * base_h : (r + 1) * base_h, c * base_w : (c + 1) * base_w] = img
    return grid
from paradex.utils.path import rsc_path, shared_dir


def load_images(scene_path, frame_idx=0):
    """Load images from raw/images directory."""
    image_dir = os.path.join(scene_path, "raw", "images")
    if not os.path.exists(image_dir):
        # Fallback to video_extracted
        image_dir = os.path.join(scene_path, "video_extracted")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"No images found in {scene_path}")

    images = {}

    # Check if images are directly in folder (e.g., {serial}.png)
    direct_images = glob.glob(os.path.join(image_dir, "*.png"))
    if not direct_images:
        direct_images = glob.glob(os.path.join(image_dir, "*.jpg"))

    if direct_images:
        # Images are {serial}.png directly in the folder
        for img_path in direct_images:
            cam_serial = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            if img is not None:
                images[cam_serial] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(f"  {cam_serial}: {img.shape}")
    else:
        # Images are in subdirectories per camera
        for cam_serial in os.listdir(image_dir):
            cam_dir = os.path.join(image_dir, cam_serial)
            if not os.path.isdir(cam_dir):
                continue

            img_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
            if not img_files:
                img_files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))

            if img_files:
                idx = min(frame_idx, len(img_files) - 1)
                img = cv2.imread(img_files[idx])
                if img is not None:
                    images[cam_serial] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print(f"  {cam_serial}: {img.shape}")

    return images


def validate_c2r(scene_path, arm="xarm", hand="inspire", frame_idx=0):
    """Project robot with zero qpos onto images from scene_path."""
    print("=" * 50)
    print("C2R Validation - Zero Qpos")
    print("=" * 50)
    print(f"Scene: {scene_path}")

    # Load calibration from scene
    print("\n[1] Loading calibration...")
    intrinsic, extrinsic = load_camparam(scene_path)
    c2r = np.load(os.path.join(scene_path, "C2R.npy"))
    print(f"    Cameras: {list(intrinsic.keys())}")

    # Load images using ImageDict
    print(f"\n[2] Loading images...")

    # Check if undistorted images exist, otherwise create them
    undistort_path = os.path.join(scene_path, "undistort")
    if os.path.exists(os.path.join(undistort_path, "images")):
        print("    Loading undistorted images...")
        img_dict = ImageDict.from_path(undistort_path)
    else:
        print("    Loading raw images and undistorting...")
        img_dict = ImageDict.from_path(scene_path)
        img_dict.set_camparam(intrinsic, extrinsic)
        img_dict = img_dict.undistort(save_path=undistort_path)
        print(f"    Saved undistorted images to {undistort_path}")

    img_dict.set_camparam(intrinsic, extrinsic)

    # Load qpos
    print("\n[3] Loading qpos...")
    full_qpos = np.zeros(12)  # 6 arm + 6 hand (default)

    # Try to load arm pose from scene if it exists
    arm_pose_file = os.path.join(scene_path, "raw", "arm", "position.npy")
    if os.path.exists(arm_pose_file):
        print(f"    Loading arm pose from {arm_pose_file}")
        arm_pose_data = np.load(arm_pose_file)

        # If it's a time series, use the first pose
        if arm_pose_data.ndim == 2:
            arm_pose = arm_pose_data[0]
            print(f"    Using first pose from time series (shape: {arm_pose_data.shape})")
        else:
            arm_pose = arm_pose_data

        # Set arm qpos (first 6 values)
        full_qpos[:6] = arm_pose
        print(f"    Loaded arm qpos: {arm_pose}")
    else:
        print(f"    No arm pose found at {arm_pose_file}, using zero qpos for arm")

    # Try to load hand pose if it exists
    hand_pose_file = os.path.join(scene_path, "raw", "hand", "position.npy")
    hand_action_file = os.path.join(scene_path, "raw", "hand", "action.npy")
    if os.path.exists(hand_action_file):
        print(f"    Loading hand action from {hand_action_file}")
        hand_action_data = np.load(hand_action_file)

        # If it's a time series, use the first action
        if hand_action_data.ndim == 2:
            hand_action = hand_action_data[0:1]  # Keep 2D for parse_inspire
            print(f"    Using first action from time series (shape: {hand_action_data.shape})")
        else:
            hand_action = hand_action_data.reshape(1, -1)

        print(f"    Raw hand action: {hand_action[0]} (Inspire units 0-2000)")

        # Convert from Inspire units (0-2000) to radians using parse_inspire
        hand_qpos_converted = parse_inspire(hand_action)
        hand_qpos = hand_qpos_converted[0]

        # Set hand qpos (last 6 values)
        full_qpos[6:] = hand_qpos
        print(f"    Converted hand qpos: {hand_qpos} (radians)")
    elif os.path.exists(hand_pose_file):
        print(f"    Loading hand pose from {hand_pose_file}")
        hand_pose_data = np.load(hand_pose_file)

        if hand_pose_data.ndim == 2:
            hand_pose = hand_pose_data[0:1]
        else:
            hand_pose = hand_pose_data.reshape(1, -1)

        print(f"    Raw hand pose: {hand_pose[0]} (Inspire units)")
        hand_qpos_converted = parse_inspire(hand_pose)
        full_qpos[6:] = hand_qpos_converted[0]
        print(f"    Converted hand qpos: {hand_qpos_converted[0]} (radians)")
    else:
        print(f"    No hand data found, using zero qpos for hand")
    print(f"    Final qpos: {full_qpos}")
    
    # Load robot
    print("\n[4] Loading robot...")
    urdf_path = os.path.join(rsc_path, "robot", f"{arm}_{hand}.urdf")
    robot = RobotModule(urdf_path)
    robot.update_cfg(full_qpos)
    mesh = robot.get_robot_mesh()
    print(f"    Mesh: {len(mesh.vertices)} verts")

    # Transform mesh from robot base frame to world frame using C2R
    print("    Transforming mesh with C2R...")
    mesh.apply_transform(c2r)

    # Render using ImageDict (handles distortion correctly)
    print("\n[5] Rendering...")
    overlay_img_dict = img_dict.project_mesh(mesh)

    # Save overlays
    print("\n[6] Saving overlays...")
    output_dir = os.path.join(scene_path, "c2r_validation")
    os.makedirs(output_dir, exist_ok=True)

    overlay_img_dict.save(output_dir)
    print(f"    Saved overlays to {output_dir}")

    # Create grid
    overlays = []
    for cam_id in sorted(overlay_img_dict.serial_list):
        img = overlay_img_dict[cam_id]
        overlays.append(img)

    if overlays:
        grid = make_image_grid(overlays)
        grid_path = os.path.join(output_dir, "grid.jpg")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"    Grid: {grid_path}")

    print("\n" + "=" * 50)
    print(f"Done! Output: {output_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire")
    args = parser.parse_args()

    validate_c2r(args.scene_path, args.arm, args.hand)


if __name__ == "__main__":
    main()