import json
import os
import time
import torch
import numpy as np
from pathlib import Path

from paradex.object_detection.default_config import name2prompt


def process_one_mask(args_tuple):
    matcherto3d, obj_name, midx, tg_mask, last_image, serial_num, template_dict, \
                                paircount_threshold, saveimg, NAS_IMG_SAVEDIR = args_tuple
    
    tg_mask = np.repeat(tg_mask[..., None], 3, axis=2).astype(np.int64) * 255.0

    src_3d_dict, tg_2d_dict, org_2d_dict = matcherto3d.match_img2template(
        last_image, tg_mask,
        template_dict[obj_name], paircount_threshold, batch_size=24,
        draw=saveimg, use_crop=True,
        image_name=str(NAS_IMG_SAVEDIR / f'matching_{serial_num}_{midx}.png')
    )

    pair_count = 0
    src_3d_points, tg_2d_points, src_cam_ids = [], [], []
    for cam_id in src_3d_dict:
        if len(src_3d_dict[cam_id]) > 0:
            pair_count += len(src_3d_dict[cam_id])
            src_3d_points.append(src_3d_dict[cam_id])
            tg_2d_points.append(tg_2d_dict[cam_id])
            src_cam_ids.append([cam_id] * len(src_3d_dict[cam_id]))

    if pair_count > 0:
        src_3d_points = np.vstack(src_3d_points).astype(np.float64)
        tg_2d_points = np.vstack(tg_2d_points).astype(np.float64)
        result = {
            'count': pair_count,
            'combined_src_3d': src_3d_points.tolist(),
            'combined_tg_2d': tg_2d_points.tolist(),
            'src_arr_cam_ids': np.hstack(src_cam_ids).tolist()
        }
    else:
        result = {'count': 0}

    return obj_name, midx, result