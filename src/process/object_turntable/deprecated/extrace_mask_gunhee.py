#!/usr/bin/env python3
"""
Preprocess paradex turntable data for gsplat:
- Uses extracted images from extract_video.py
- Uses rotation matrices from get_rotation_gunhee.py
- Downscale images to half resolution
- Adjust intrinsics accordingly
- Generate masks using SAM3
- Save masks and masked images to gsplat/data/paradex_turntable
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import sys
import glob
import contextlib
from scipy.spatial.transform import Rotation
import torch
from PIL import Image
from tqdm import tqdm
import gc

# Add paradex to path if needed
paradex_path = Path(__file__).parent.parent.parent / "paradex"
if str(paradex_path) not in sys.path:
    sys.path.insert(0, str(paradex_path / "src"))

try:
    from paradex.utils.path import shared_dir, home_path
    from paradex.calibration.utils import load_camparam
except ImportError:
    print("Warning: Could not import paradex utilities. Using default paths.")
    shared_dir = os.path.expanduser("~/shared_data")
    home_path = os.path.expanduser("~")

# Add sam3 to path
sam3_path = Path(__file__).parent.parent.parent / "sam3"
if str(sam3_path) not in sys.path:
    sys.path.insert(0, str(sam3_path))

try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SAM3: {e}. Make sure sam3 is in your path.")
    SAM3_AVAILABLE = False


def downscale_image(image, scale=0.5):
    """Downscale image by scale factor."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def adjust_intrinsics(intrinsics, scale=0.5):
    """Adjust camera intrinsics for downscaled image."""
    K = np.array(intrinsics)
    K[0, 0] *= scale  # fx
    K[1, 1] *= scale  # fy
    K[0, 2] *= scale  # cx
    K[1, 2] *= scale  # cy
    return K.tolist()


def combine_masks(masks):
    """Combine multiple masks into a single binary mask."""
    if len(masks) == 0:
        return None
    combined = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        combined = combined | mask
    return combined


def build_sam3_model_and_processor(device=None):
    """
    Build SAM3 model and processor. This can be reused across multiple calls.
    
    Args:
        device: torch.device to use (None = auto-detect)
    
    Returns:
        tuple: (model, processor, device)
    """
    if not SAM3_AVAILABLE:
        raise RuntimeError("SAM3 is not available. Cannot generate masks.")
    
    # Initialize device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Enable TF32 for Ampere GPUs (faster computation)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            device = torch.device("cpu")
    
    print(f"Building SAM3 image model (device: {device})...")
    try:
        # Try to find BPE path
        sam3_root = Path(sam3.__file__).parent.parent
        bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        
        if not bpe_path.exists():
            # Try alternative path
            bpe_path = sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        
        if bpe_path.exists():
            model = build_sam3_image_model(bpe_path=str(bpe_path))
        else:
            print("Warning: BPE path not found, using default")
            model = build_sam3_image_model()
    except Exception as e:
        print(f"Warning: Error building model with BPE, using default: {e}")
        model = build_sam3_image_model()
    
    # Set model to evaluation mode and disable gradient computation globally
    model.eval()
    torch.set_grad_enabled(False)
    
    processor = Sam3Processor(model, confidence_threshold=0.5, device=str(device))
    
    return model, processor, device


def generate_masks_sam3(images_folder, text_prompt="object", gpu=None, model=None, processor=None, device=None):
    """
    Generate masks for all images in a folder using SAM3 image processor.
    This processes each image independently, avoiding memory accumulation.
    
    Args:
        images_folder: Path to folder containing images
        text_prompt: Text prompt for segmentation
        gpu: GPU ID to use (None = use default) - deprecated, use device instead
        model: Optional pre-built SAM3 model (if None, will build one)
        processor: Optional pre-built SAM3 processor (if None, will build one)
        device: Optional torch.device (if None, will auto-detect)
    
    Returns:
        dict: Dictionary mapping frame indices to masks (numpy arrays)
    """
    if not SAM3_AVAILABLE:
        raise RuntimeError("SAM3 is not available. Cannot generate masks.")
    
    images_folder = Path(images_folder)
    if not images_folder.exists():
        raise ValueError(f"Images folder does not exist: {images_folder}")
    
    # Get all jpg files and sort them
    image_files = sorted(glob.glob(str(images_folder / "*.jpg")))
    if len(image_files) == 0:
        raise ValueError(f"No .jpg files found in {images_folder}")
    
    print(f"Found {len(image_files)} images in {images_folder}")
    
    # Build model and processor if not provided
    if model is None or processor is None:
        model, processor, device = build_sam3_model_and_processor(device=device)
    else:
        # Use provided model and processor, but ensure device is set
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        print(f"Using provided SAM3 model and processor (device: {device})")
    
    # Process each image independently
    print(f"Processing {len(image_files)} images with text prompt: '{text_prompt}'...")
    masks_dict = {}
    
    # Process images with proper context management
    for idx, image_path in enumerate(tqdm(image_files, desc="Generating masks")):
        try:
            # Load image
            from PIL import Image
            image = Image.open(image_path)
            
            # Use autocast context manager properly (opens and closes for each image)
            # This ensures autocast state is cleaned up after each image
            autocast_context = (
                torch.autocast("cuda", dtype=torch.bfloat16) 
                if device.type == "cuda" 
                else contextlib.nullcontext()
            )
            
            with autocast_context:
                # Set image and text prompt (inside autocast context)
                inference_state = processor.set_image(image)
                inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
                
                # Get masks from inference state
                # Masks from SAM3 have shape (N, 1, H, W) where N is number of detected objects
                # H, W are at original image resolution
                if "masks" in inference_state and inference_state["masks"].numel() > 0:
                    masks = inference_state["masks"].cpu().numpy()  # Shape: (N, 1, H, W) or (N, H, W)
                    
                    # Handle different mask shapes
                    if len(masks.shape) == 4:
                        # Shape: (N, 1, H, W) - squeeze out channel dimension
                        masks = masks.squeeze(1)  # Now (N, H, W)
                    elif len(masks.shape) == 3:
                        # Shape: (N, H, W) - already correct
                        pass
                    else:
                        print(f"  Warning: Unexpected mask shape for image {idx} ({Path(image_path).name}): {masks.shape}")
                        continue
                    
                    # Combine all masks with OR operation (if multiple objects detected)
                    if masks.shape[0] > 1:
                        combined_mask = np.any(masks, axis=0)  # Shape: (H, W)
                    else:
                        combined_mask = masks[0]  # Shape: (H, W)
                    
                    # Verify mask is valid (2D, not empty)
                    if combined_mask.size > 0 and len(combined_mask.shape) == 2:
                        # Store mask at original resolution - we'll resize later to match downscaled image
                        masks_dict[idx] = combined_mask
                    else:
                        print(f"  Warning: Invalid mask shape for image {idx} ({Path(image_path).name}): {combined_mask.shape}")
                else:
                    # No masks found for this image
                    print(f"  Warning: No masks found for image {idx} ({Path(image_path).name})")
            
            # Reset prompts and clean up inference state (outside autocast context)
            processor.reset_all_prompts(inference_state)
            del inference_state  # Explicitly delete to free GPU memory
            
        except Exception as e:
            print(f"  Error processing image {idx} ({Path(image_path).name}): {e}")
            # Try to clean up even on error
            if 'inference_state' in locals():
                try:
                    processor.reset_all_prompts(inference_state)
                    del inference_state
                except:
                    pass
            continue
        
        # Clear GPU cache periodically (every 10 images)
        if (idx + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Note: We don't delete model/processor here if they were provided
    # They should be cleaned up by the caller when done with all processing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"Successfully generated masks for {len(masks_dict)}/{len(image_files)} images")
    
    return masks_dict, image_files


def compute_camera_pose(extrinsic, rotation_matrix):
    """
    Compute camera pose (camera-to-world) from base extrinsic and rotation matrix.
    
    Args:
        extrinsic: Base camera extrinsic (3x4 or 4x4) - world-to-camera
        rotation_matrix: Rotation matrix for this frame (4x4) - turntable rotation (not c2w)
    
    Returns:
        camtoworld: Camera-to-world transformation (4x4)
        w2c: World-to-camera transformation (3x4) - matching original format
    """
    # Convert extrinsic to 4x4 if needed
    if extrinsic.shape == (3, 4):
        w2c_base = np.eye(4)
        w2c_base[:3] = extrinsic
    else:
        w2c_base = extrinsic
    
    # Apply turntable rotation: multiply rot_mat in front of extrinsic (matching generate_colmap.py)
    # rot_mat @ extrinsic gives the new world-to-camera transformation
    new_w2c = w2c_base @ np.linalg.inv(rotation_matrix)
    
    # Convert to camera-to-world
    camtoworld = np.linalg.inv(new_w2c)
    
    # Return w2c in 3x4 format (matching original extrinsics.json format)
    w2c_3x4 = new_w2c[:3]
    
    return camtoworld, w2c_3x4


def preprocess_turntable_data(data_dir, object_name=None, prompt=None, scale=0.5, frame_skip=1, camera_prefix=None):
    """
    Preprocess turntable data directory.
    
    Args:
        data_dir: Path to turntable data directory (e.g., pepper_tuna/2025-12-02_19-12-45)
        object_name: Object name for directory/file naming (auto-detected from directory name if None)
        prompt: Text prompt for SAM3 segmentation (defaults to object_name if None)
        scale: Downscaling factor (default 0.5 for half resolution)
        frame_skip: Process every Nth frame (default 1 = all frames)
        camera_prefix: Filter to only process cameras with this prefix. Can be:
            - A single string (e.g., "23280285")
            - A list of strings (e.g., ["23280285", "23022633"])
            - A comma-separated string (e.g., "23280285,23022633")
            - None to process all cameras
    """
    data_dir = Path(data_dir).resolve()
    
    # Check if this is a timestamped directory or the base directory
    if (data_dir / "images").exists():
        base_dir = data_dir
        # This is a timestamp directory (e.g., 2025-12-02_19-12-45)
        # Object name should be the parent directory name
        if object_name is None:
            object_name = data_dir.parent.name
    else:
        # Look for timestamped subdirectories
        subdirs = [d for d in data_dir.iterdir() if d.is_dir() and (d / "images").exists()]
        if len(subdirs) == 0:
            raise ValueError(f"No valid data directory found in {data_dir}")
        if len(subdirs) > 1:
            print(f"Warning: Multiple timestamped directories found, using {subdirs[0]}")
        base_dir = subdirs[0]
        # This is an object directory (e.g., pepper_tuna)
        # Object name should be the directory name itself
        if object_name is None:
            object_name = data_dir.name
    
    # Auto-detect object name from directory path if still None (fallback)
    if object_name is None:
        # Try to extract from the path: look for directories under /paradex_turntable/ or /object_turntable/
        parts = data_dir.parts
        try:
            turntable_idx = -1
            for i, part in enumerate(parts):
                if 'turntable' in part.lower():
                    turntable_idx = i
                    break
            if turntable_idx >= 0 and turntable_idx + 1 < len(parts):
                object_name = parts[turntable_idx + 1]
            else:
                object_name = data_dir.name
        except (ValueError, IndexError):
            # Fallback: use directory name
            object_name = data_dir.name
    
    print(f"Using object name: {object_name}")
    
    # Use prompt for SAM3 segmentation (default to object_name if not provided)
    if prompt is None:
        prompt = object_name
    print(f"Using SAM3 prompt: {prompt}")
    
    images_dir = base_dir / "images"
    
    # Load rotation matrices from local directory
    rot_path = base_dir / "rot.npy"
    
    if not rot_path.exists():
        raise ValueError(f"Rotation matrices not found: {rot_path}")
    
    rot_matrices = np.load(rot_path)
    print(f"Loaded rotation matrices from: {rot_path}")
    
    # Load camera parameters - try base_dir first (matching original), then shared_dir
    cam_param_dir = base_dir / "cam_param"
    
    # If not in base_dir, try shared_dir
    if not cam_param_dir.exists():
        demo_path_shared = Path(shared_dir) / "capture" / "object_turntable" / object_name / base_dir.name
        cam_param_dir = demo_path_shared / "cam_param"
    
    if not cam_param_dir.exists():
        raise ValueError(f"Camera parameters directory not found. Tried:\n  - {base_dir / 'cam_param'}\n  - {cam_param_dir}")
    
    intrinsics_path = cam_param_dir / "intrinsics.json"
    extrinsics_path = cam_param_dir / "extrinsics.json"
    
    if not intrinsics_path.exists():
        raise ValueError(f"Intrinsics file not found: {intrinsics_path}")
    if not extrinsics_path.exists():
        raise ValueError(f"Extrinsics file not found: {extrinsics_path}")
    
    with open(intrinsics_path, 'r') as f:
        intrinsics_data = json.load(f)
    
    with open(extrinsics_path, 'r') as f:
        extrinsics_data = json.load(f)
    
    # Output directory: gsplat/data/paradex_turntable/{object_name}
    output_base = Path("/home/gunhee/2025/3drecon/gsplat/data/paradex_turntable")
    output_dir = output_base / object_name
    
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    masked_images_dir = output_dir / "masked_images"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    masked_images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Get all camera serial numbers from base_dir/images (source)
    source_images_dir = base_dir / "images"
    camera_serials = [d.name for d in source_images_dir.iterdir() if d.is_dir()]
    
    # Filter by camera prefix(es) if provided
    if camera_prefix is not None:
        # Handle multiple formats: single string, list, or comma-separated string
        if isinstance(camera_prefix, str):
            # Check if it's comma-separated
            if ',' in camera_prefix:
                prefixes = [p.strip() for p in camera_prefix.split(',')]
            else:
                prefixes = [camera_prefix]
        elif isinstance(camera_prefix, list):
            prefixes = camera_prefix
        else:
            prefixes = [str(camera_prefix)]
        
        # Filter cameras to match any of the prefixes
        filtered_cameras = []
        for cam in camera_serials:
            if any(cam.startswith(prefix) for prefix in prefixes):
                filtered_cameras.append(cam)
        
        camera_serials = filtered_cameras
        
        if len(camera_serials) == 0:
            available_cameras = sorted([d.name for d in source_images_dir.iterdir() if d.is_dir()])
            raise ValueError(f"No cameras found matching prefix(es) {prefixes}. Available cameras: {available_cameras}")
        print(f"Filtered to {len(camera_serials)} camera(s) matching prefix(es) {prefixes}: {camera_serials}")
    
    camera_serials = sorted(camera_serials)
    
    print(f"Found {len(camera_serials)} camera(s) to process")
    
    # Build SAM3 model and processor once for all cameras (reused across cameras)
    sam3_model = None
    sam3_processor = None
    sam3_device = None
    if SAM3_AVAILABLE:
        print("Building SAM3 model and processor (will be reused for all cameras)...")
        sam3_model, sam3_processor, sam3_device = build_sam3_model_and_processor()
    
    # Process each camera
    processed_images = []
    
    # Store computed extrinsics and intrinsics per image (matching image filenames)
    computed_extrinsics = {}
    computed_intrinsics = {}
    
    try:
        for camera_serial in camera_serials:
            if camera_serial not in intrinsics_data:
                print(f"Warning: Camera {camera_serial} not found in intrinsics, skipping")
                continue
            
            if camera_serial not in extrinsics_data:
                print(f"Warning: Camera {camera_serial} not found in extrinsics, skipping")
                continue
            
            camera_images_dir = source_images_dir / camera_serial
            frame_files = sorted(camera_images_dir.glob("frame_*.jpg"))
            
            if len(frame_files) == 0:
                print(f"Warning: No frames found for camera {camera_serial}, skipping")
                continue
            
            print(f"Processing camera {camera_serial} ({len(frame_files)} frames)...")
            
            # Get base extrinsic for this camera
            extrinsic_base = np.array(extrinsics_data[camera_serial])  # 3x4
            
            # First, filter frames to determine which ones we'll actually process
            # This reduces the number of frames SAM3 needs to process
            print(f"  Filtering frames for camera {camera_serial}...")
            valid_frames = []  # List of (frame_file, frame_num) tuples
            
            for frame_file in frame_files[::frame_skip]:
                frame_num = int(frame_file.stem.split('_')[1])
                
                # Check if we have rotation matrix for this frame
                if frame_num > len(rot_matrices):
                    continue  # Skip frames without rotation matrices
                
                # Check if image can be loaded
                image = cv2.imread(str(frame_file))
                if image is None:
                    continue  # Skip unloadable images
                
                valid_frames.append((frame_file, frame_num))
            
            if len(valid_frames) == 0:
                print(f"  No valid frames found for camera {camera_serial}, skipping")
                continue
        
            print(f"  Found {len(valid_frames)} valid frames (out of {len(frame_files)} total, after frame_skip={frame_skip})")
            use_frame_suffix = len(valid_frames) > 1
            
            # Generate masks using SAM3 for only the valid frames
            print(f"  Generating masks using SAM3 for {len(valid_frames)} valid frames...")
            frame_to_mask_map = {}  # Maps frame_num to mask
            try:
                # Create temporary folder with only valid frames for SAM3
                import tempfile
                import shutil
                temp_images_dir = Path(tempfile.mkdtemp())
                
                # Copy only valid images to temp directory for SAM3 processing
                # Store mapping from SAM3 index to original frame number
                sam3_index_to_frame = {}
                for sam3_idx, (frame_file, frame_num) in enumerate(valid_frames):
                    temp_image_name = f"{sam3_idx:06d}.jpg"
                    shutil.copy2(frame_file, temp_images_dir / temp_image_name)
                    sam3_index_to_frame[sam3_idx] = frame_num
                
                # Generate masks using SAM3 image processor
                # This processes each image independently, so no memory accumulation
                # Reuse the pre-built model and processor for efficiency
                masks_dict, _ = generate_masks_sam3(
                    str(temp_images_dir), 
                    text_prompt=prompt,
                    model=sam3_model,
                    processor=sam3_processor,
                    device=sam3_device
                )
                
                # Map SAM3 indices back to frame numbers
                for sam3_idx, mask in masks_dict.items():
                    if sam3_idx in sam3_index_to_frame:
                        frame_num = sam3_index_to_frame[sam3_idx]
                        frame_to_mask_map[frame_num] = mask
                
                # Clean up temp directory
                shutil.rmtree(temp_images_dir)
                
            except Exception as e:
                print(f"  Error generating masks with SAM3: {e}")
                import traceback
                traceback.print_exc()
                # Clear GPU cache even on error
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                continue
            
            # Clear GPU cache after processing each camera
            print(f"  Clearing GPU cache after processing camera {camera_serial}...")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Process only the valid frames (already filtered and have masks)
            for frame_file, frame_num in valid_frames:
                # Get rotation matrix (frame_num is 1-based, array is 0-based)
                rot_matrix = rot_matrices[frame_num - 1]
                
                # Load image (we already checked it's loadable, but load again for processing)
                image = cv2.imread(str(frame_file))
                if image is None:
                    print(f"Warning: Could not load image {frame_file}, skipping")
                    continue
                
                original_h, original_w = image.shape[:2]
                downscaled_image = downscale_image(image, scale)
                new_h, new_w = downscaled_image.shape[:2]
                
                # Adjust intrinsics
                if "intrinsics_undistort" in intrinsics_data[camera_serial]:
                    original_intrinsics = intrinsics_data[camera_serial]["intrinsics_undistort"]
                else:
                    original_intrinsics = intrinsics_data[camera_serial]["original_intrinsics"]
                adjusted_intrinsics = adjust_intrinsics(original_intrinsics, scale)
                
                # Compute camera pose
                camtoworld, w2c_3x4 = compute_camera_pose(extrinsic_base, rot_matrix)
                
                # Create image name matching original format: {camera_serial}.jpg
                # For multiple frames (regardless of frame_skip), we'll use frame number as suffix
                # to avoid overwriting files
                if use_frame_suffix:
                    # Multiple frames: use frame number as suffix
                    image_filename = f"{camera_serial}_{frame_num:06d}.jpg"
                else:
                    # Single frame: use camera serial only (matching original)
                    image_filename = f"{camera_serial}.jpg"
                mask_filename = image_filename
                
                # Get mask from SAM3 results using frame number
                if frame_num not in frame_to_mask_map:
                    print(f"Skipping {image_filename}: mask generation failed for frame {frame_num}")
                    continue
                
                mask_bool = frame_to_mask_map[frame_num]
                
                # Verify mask is valid
                if mask_bool is None or mask_bool.size == 0:
                    print(f"Skipping {image_filename}: invalid mask for frame {frame_num}")
                    continue
                
                # Verify mask has valid shape (should be 2D at original image resolution)
                if len(mask_bool.shape) != 2 or mask_bool.shape[0] == 0 or mask_bool.shape[1] == 0:
                    print(f"Skipping {image_filename}: invalid mask shape {mask_bool.shape} for frame {frame_num}")
                    continue
                
                # Mask is at original image resolution (original_h, original_w)
                # We need to resize it to match the downscaled image size (new_h, new_w)
                # Always resize since mask is at original resolution and image is downscaled
                try:
                    # Convert boolean mask to uint8 for cv2.resize
                    if mask_bool.dtype == bool:
                        mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                    else:
                        mask_uint8 = (mask_bool > 0.5).astype(np.uint8) * 255
                    
                    # Resize mask to match downscaled image size
                    mask_resized = cv2.resize(
                        mask_uint8,
                        (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    
                    # Convert back to boolean
                    mask_bool = mask_resized > 127
                    
                except cv2.error as e:
                    print(f"Skipping {image_filename}: error resizing mask from {mask_bool.shape} to ({new_h}, {new_w}): {e}")
                    continue
                
                # Convert to uint8 mask
                mask = (mask_bool.astype(np.uint8) * 255)
                
                # Verify mask is valid before saving
                if mask.size == 0 or mask.shape != downscaled_image.shape[:2]:
                    print(f"Skipping {image_filename}: invalid mask shape after resize")
                    continue
                
                # Check if mask has any foreground pixels (not completely empty)
                if mask.sum() == 0:
                    print(f"Skipping {image_filename}: mask is empty (no foreground pixels)")
                    continue
                
                # Save image, mask, and masked image
                cv2.imwrite(str(images_dir / image_filename), downscaled_image)
                cv2.imwrite(str(masks_dir / mask_filename), mask)
                
                # Create and save masked image (background set to black)
                masked_image = downscaled_image.copy()
                mask_bool = mask > 127
                masked_image[~mask_bool] = [0, 0, 0]
                cv2.imwrite(str(masked_images_dir / image_filename), masked_image)
                
                # Store extrinsics and intrinsics per image (matching filename without extension) - only if mask succeeded
                image_key = Path(image_filename).stem  # Remove .jpg extension
                computed_extrinsics[image_key] = w2c_3x4.tolist()
                computed_intrinsics[image_key] = {
                    "intrinsics_adjusted": adjusted_intrinsics,
                    "width_adjusted": new_w,
                    "height_adjusted": new_h,
                }
                
                # Store data for tracking
                processed_images.append({
                    "camera_id": camera_serial,
                    "frame_num": frame_num,
                    "image_name": image_filename,
                })
        
        # Save intrinsics and extrinsics per image (matching image filenames)
        adjusted_intrinsics_path = output_dir / "intrinsics_adjusted.json"
        with open(adjusted_intrinsics_path, 'w') as f:
            json.dump(computed_intrinsics, f, indent=2)
        print(f"\nSaved adjusted intrinsics (per image) to {adjusted_intrinsics_path}")
        
        # Save computed extrinsics (w2c) per image (matching image filenames)
        computed_extrinsics_path = output_dir / "extrinsics.json"
        with open(computed_extrinsics_path, 'w') as f:
            json.dump(computed_extrinsics, f, indent=2)
        print(f"Saved computed extrinsics (w2c, per image) to {computed_extrinsics_path}")
        
        # Copy original intrinsics for reference
        import shutil
        shutil.copy(intrinsics_path, output_dir / "intrinsics.json")
    
    finally:
        # Clean up SAM3 model and processor
        if sam3_model is not None:
            del sam3_processor
            del sam3_model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Cleaned up SAM3 model and processor")
    
    print(f"\nPreprocessing complete! Processed {len(processed_images)} images.")
    print(f"Images saved to: {images_dir}")
    print(f"Masks saved to: {masks_dir}")
    print(f"Masked images saved to: {masked_images_dir}")
    
    return processed_images


def preprocess_all_turntable_data(turntable_base_dir=None, prompt=None, scale=0.5, frame_skip=1, camera_prefix=None, object_names=None):
    """
    Preprocess turntable object directories.
    
    Args:
        turntable_base_dir: Base directory containing object directories (default: ~/paradex_download/capture/object_turntable)
        prompt: Text prompt for SAM3 segmentation (defaults to object_name for each object if None)
        scale: Downscaling factor (default 0.5 for half resolution)
        frame_skip: Process every Nth frame (default 1 = all frames)
        camera_prefix: Filter to only process cameras with this prefix. Can be:
            - A single string (e.g., "23280285")
            - A list of strings (e.g., ["23280285", "23022633"])
            - A comma-separated string (e.g., "23280285,23022633")
            - None to process all cameras
        object_names: List of object names to process (None = process all objects)
    """
    if turntable_base_dir is None:
        turntable_base_dir = Path(home_path) / "paradex_download" / "capture" / "object_turntable"
    else:
        turntable_base_dir = Path(turntable_base_dir)
    
    if not turntable_base_dir.exists():
        raise ValueError(f"Turntable base directory not found: {turntable_base_dir}")
    
    # Find all object directories
    object_dirs = []
    for obj_dir in turntable_base_dir.iterdir():
        if not obj_dir.is_dir():
            continue
        
        # Filter by object_names if specified
        if object_names is not None and obj_dir.name not in object_names:
            continue
        
        # Check for timestamped subdirectories with images
        for subdir in obj_dir.iterdir():
            if subdir.is_dir() and (subdir / "images").exists():
                object_dirs.append(subdir)
    
    if len(object_dirs) == 0:
        print(f"No object directories found in {turntable_base_dir}")
        return
    
    print(f"Found {len(object_dirs)} object directories to process:")
    for obj_dir in object_dirs:
        print(f"  - {obj_dir.parent.name}/{obj_dir.name}")
    
    print("\n" + "="*60)
    
    for obj_dir in sorted(object_dirs):
        print(f"\nProcessing: {obj_dir.parent.name}/{obj_dir.name}")
        print("="*60)
        try:
            processed = preprocess_turntable_data(obj_dir, object_name=None, prompt=prompt, scale=scale, frame_skip=frame_skip, camera_prefix=camera_prefix)
            print(f"✓ Successfully processed {len(processed)} images for {obj_dir.parent.name}/{obj_dir.name}")
        except Exception as e:
            print(f"✗ Error processing {obj_dir.parent.name}/{obj_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Batch processing complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess paradex turntable data for nvdiffrec")
    parser.add_argument("data_dir", type=str, nargs="?", default=None, 
                       help="Path to turntable data directory (e.g., pepper_tuna/2025-12-02_19-12-45). If not provided, processes all directories")
    parser.add_argument("--object_name", type=str, default=None, 
                       help="Object name for directory/file naming (auto-detected from directory name if not provided)")
    parser.add_argument("--object_names", type=str, nargs="*", default=None,
                       help="List of object names to process (e.g., --object_names blue_metal_bottle pepper_tuna). If not provided, processes all objects")
    parser.add_argument("--prompt", type=str, default="object on the checkerboard, excluding checkerboard",
                       help="Text prompt for SAM3 segmentation (defaults to object_name if not provided)")
    parser.add_argument("--scale", type=float, default=0.5, 
                       help="Downscaling factor (default: 0.5)")
    parser.add_argument("--frame_skip", type=int, default=5,
                       help="Process every Nth frame (default: 5)")
    parser.add_argument("--turntable_base", type=str, default=None,
                       help="Base directory containing object directories (default: ~/paradex_download/capture/object_turntable)")
    parser.add_argument("--camera-prefix", type=str, default=None,
                       help="Filter to only process images from cameras matching this prefix. "
                            "Can be a single prefix (e.g., '23280285') or comma-separated multiple prefixes "
                            "(e.g., '23280285,23022633'). Default: '23022633'")
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        # Process all directories (optionally filtered by object_names)
        preprocess_all_turntable_data(args.turntable_base, args.prompt, args.scale, args.frame_skip, args.camera_prefix, args.object_names)
    else:
        # Process single directory
        preprocess_turntable_data(args.data_dir, args.object_name, args.prompt, args.scale, args.frame_skip, args.camera_prefix)

