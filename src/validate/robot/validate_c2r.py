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
from paradex.image.grid import make_image_grid
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

    # Load images
    print(f"\n[2] Loading images (frame {frame_idx})...")
    images = load_images(scene_path, frame_idx)

    # Zero qpos
    print("\n[3] Using zero qpos...")
    full_qpos = np.zeros(12)  # 6 arm + 6 hand
    print(f"    qpos: {full_qpos}")

    # Load robot
    print("\n[4] Loading robot...")
    urdf_path = os.path.join(rsc_path, "robot", f"{arm}_{hand}_left_new.urdf")
    robot = RobotModule(urdf_path)
    robot.update_cfg(full_qpos)
    mesh = robot.get_robot_mesh()
    print(f"    Mesh: {len(mesh.vertices)} verts")

    # Setup renderer
    print("\n[5] Rendering...")
    print(f"    Intrinsic cams: {list(intrinsic.keys())}")
    print(f"    Image cams: {list(images.keys())}")

    # Match cameras
    filtered_intrinsic = {k: v for k, v in intrinsic.items() if k in images}

    if not filtered_intrinsic:
        print("    No matching cameras! Using all available images...")
        # Use images as-is, assume intrinsic keys match
        filtered_intrinsic = {k: intrinsic[k] for k in intrinsic.keys() if k in images}
        if not filtered_intrinsic:
            # Try using first available intrinsic for each image
            intrinsic_list = list(intrinsic.keys())
            image_list = list(images.keys())
            filtered_intrinsic = {}
            for i, img_cam in enumerate(image_list):
                if i < len(intrinsic_list):
                    filtered_intrinsic[img_cam] = intrinsic[intrinsic_list[i]]
            # Also need to remap extrinsic
            extrinsic = {image_list[i]: extrinsic[intrinsic_list[i]]
                        for i in range(min(len(image_list), len(intrinsic_list)))}

    extrinsic_c2r = {}
    for cam_id in filtered_intrinsic.keys():
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsic[cam_id]
        extr_full = cam_from_world @ c2r
        extrinsic_c2r[cam_id] = extr_full[:3, :]

    print(f"    Rendering {len(filtered_intrinsic)} cameras...")
    renderer = BatchRenderer(filtered_intrinsic, extrinsic_c2r)
    color_dict, mask_dict, depth_dict = renderer.render(mesh)

    # Create overlays
    print("\n[6] Saving overlays...")
    output_dir = os.path.join(scene_path, "c2r_validation")
    os.makedirs(output_dir, exist_ok=True)

    overlays = []
    for cam_id in sorted(filtered_intrinsic.keys()):
        if cam_id not in images:
            continue

        mask = mask_dict[cam_id].astype(np.float32)
        overlay = overlay_mask(images[cam_id], mask, color=(0, 255, 0), alpha=0.5)
        overlays.append(overlay)

        out_path = os.path.join(output_dir, f"{cam_id}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"    {out_path}")

    # Grid
    if overlays:
        grid = make_image_grid(overlays)
        grid_path = os.path.join(output_dir, "grid.jpg")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\n    Grid: {grid_path}")

    print("\n" + "=" * 50)
    print(f"Done! Output: {output_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=str, required=True)
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    validate_c2r(args.scene_path, args.arm, args.hand, args.frame)


if __name__ == "__main__":
    main()
