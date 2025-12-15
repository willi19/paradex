import torch
torch.set_grad_enabled(False)
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import gc

from paradex.utils.path import home_path, shared_dir
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, merge_charuco_detection
from paradex.image.image_dict import ImageDict
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.image.aruco import get_board_cor

gpus_to_use = range(torch.cuda.device_count())
video_predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

def propagate_in_video(predictor, session_id, start_frame_idx):
    # we will just propagate from the start frame to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_idx,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
        del response

    return outputs_per_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

def process_video(root_dir, serial_num, video_predictor=video_predictor):
    image_dir = os.path.join(root_dir, "images")
    outdir = os.path.join(root_dir, "masks")
    outdir_img = os.path.join(root_dir, "masked_images")
    # import pdb; pdb.set_trace()
    
    new_img_path = os.path.join(root_dir, "selected", serial_num)
    if os.path.exists(new_img_path):
        shutil.rmtree(new_img_path)
        
    if os.path.exists(os.path.join(outdir, serial_num)):
        shutil.rmtree(os.path.join(outdir, serial_num))
    if os.path.exists(os.path.join(outdir_img, serial_num)):
        shutil.rmtree(os.path.join(outdir_img, serial_num))

    new_img_list = []
    os.makedirs(new_img_path, exist_ok=True)
    
    for charuco_file in os.listdir(os.path.join(root_dir, "charuco_3d")):
        if "_id" not in charuco_file:
            continue
        idx = charuco_file.split("_id")[0]
        if not os.path.exists(os.path.join(image_dir, serial_num, f"frame_{idx}.jpg")) or int(idx) % 40 != 1:
            continue
        cv2.imwrite(os.path.join(new_img_path, f"{idx}.jpg"), cv2.imread(os.path.join(image_dir, serial_num, f"frame_{idx}.jpg")))
        new_img_list.append(f"frame_{idx}.jpg")

    new_img_list.sort()
    
    img_for_vis = [cv2.imread(os.path.join(new_img_path, img_name)) for img_name in sorted(os.listdir(new_img_path))]
    video_path = os.path.join(new_img_path)
    # Start a session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    
    start_frame_idx = -1
    for frame_id in range(len(img_for_vis)):
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_id, # Arbitrary frame index
                text=f"object on the checkerboard, excluding checkerboard",
            )
        )
        out = response["outputs"]
        if len(out['out_obj_ids']) > 0:
            start_frame_idx = frame_id
            print(f"Object detected in frame {frame_id} of serial {serial_num}, using as start frame.")
            break
        video_predictor.reset_session(session_id)

    if start_frame_idx == -1:
        print(f"No object detected in any frame of serial {serial_num}, skipping.")
        video_predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        return
    # import pdb; pdb.set_trace()
    outputs_per_frame = propagate_in_video(video_predictor, session_id, start_frame_idx)
    vis_frame_stride = 1
    for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
        out = outputs_per_frame[frame_idx]
        if len(out['out_obj_ids']) == 0:
            continue
        mask = (out['out_binary_masks'][0] * 255).astype("uint8")
        output_path = os.path.join(outdir, serial_num, new_img_list[frame_idx].replace(".jpg", ".png"))
        os.makedirs(os.path.join(outdir, serial_num), exist_ok=True)
        cv2.imwrite(output_path, mask)
        
        masked_img = cv2.bitwise_and(img_for_vis[frame_idx], img_for_vis[frame_idx], mask=mask)
        output_img_path = os.path.join(outdir_img, serial_num, new_img_list[frame_idx])
        os.makedirs(os.path.join(outdir_img, serial_num), exist_ok=True)
        cv2.imwrite(output_img_path, masked_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    

        
def load_mask(root_dir, predictor=video_predictor):

    outdir = os.path.join(root_dir, "masks")
    outdir_img = os.path.join(root_dir, "masked_images")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_img, exist_ok=True)
    
    obj_name = os.path.basename(os.path.dirname(root_dir))
    print(f"Processing object: {obj_name}")

    image_dir = os.path.join(root_dir, "images")
    for serial_num in os.listdir(image_dir):
        process_video(root_dir, serial_num)
    
        
        
        
root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
error_log = []
for i, obj_name in enumerate(sorted(os.listdir(root_dir))):
    obj_path = os.path.join(root_dir, obj_name)
    for index in os.listdir(obj_path):
        demo_path = os.path.join("capture/object_turntable", obj_name, index)
        try:
            load_mask(os.path.join(home_path, "paradex_download", demo_path), predictor=video_predictor)
            print(f"Finished processing {demo_path}")
        except Exception as e:
            print(f"Error processing {demo_path}: {e}")
            error_log.append((demo_path, i))
        