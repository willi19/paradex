'''
    Config and args file for object optim
'''
import time
import cv2
import numpy as np 
import os, sys
from pathlib import Path
PROJECT_DIR = Path(__file__).absolute().parent.parent
print(f'PROJECT_DIR {PROJECT_DIR}')
sys.path.insert(0, str(PROJECT_DIR))

from paradex.object_detection.obj_utils.io import get_binary_mask
from paradex.object_detection.obj_utils.scene import Scene
from paradex.model.yolo_world_module import check_mask
from paradex.object_detection.obj_utils.vis_utils import make_grid_image_np, putText, crop_and_resize_by_mask

# TODO: 저장해둔 object 이름들
from paradex.utils.file_io import shared_dir
obj_list = ['pringles']
template_path = {"pringles":os.path.join(shared_dir, "capture/marker_object/pringles/0")}
ELOFTR_CKPT_PATH = os.path.join(shared_dir, 'thirdparty/EfficientLoFTR/ckpts/eloftr_outdoor.ckpt')
# 파일 형태로 저장해두면 더 좋을 듯

hide_list = ['22641005','22645021','23280594','23180202','22641023','23029839','22640993']

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--obj_name', default=None)# for parallel processing 
    parser.add_argument('--scene_path', type=str, required=True)
    parser.add_argument('--tg_cams', type=str, nargs='+')
    
    parser.add_argument('--object_scale', type=float)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--iou_weight', type=float, default=0.5)
    parser.add_argument('--tg_weight', type=float, default=10.0)

    parser.add_argument('--rescale_factor', type=float, default=0.25)

    parser.add_argument('--fibonacci_numb', type=int, default=1000)
    parser.add_argument('--initial_cam_id', type=str, default='22645029')

    parser.add_argument('--confidence', type=float, default=0.002)
    parser.add_argument('--wo_simplify', action='store_true')

    parser.add_argument('--renderer_type', type=str, default='nvdiffrast', choices=['pytorch3d','nvdiffrast'])
    
    parser.add_argument('--mask_logic', type=str, default='precomputed', choices=['precomputed','yolo'])

    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--use_min", action='store_true')
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--use_sil', action='store_true')

    parser.add_argument("--reload_mask", action='store_true')
    parser.add_argument("--force_rerender", action='store_true', help='rerender object rotation candidate. \
                                                                use this option camera extrinsic parameter change')
    
    parser.add_argument('--dinofeature', action='store_true')
    parser.add_argument('--localfeature', action='store_true')
    parser.add_argument('--stepsize', type=int, default=1)
    args = parser.parse_args()

    return args

# using yolo or predefined. 
def load_obj_masks(org_scene:Scene, reload_mask = False, debug=False, confidence=0.002):
    if reload_mask:
        assert org_scene.mask_detector is not None, 'Scene should have mask detector.'

    if reload_mask:
        obj_name = org_scene.obj_nm

        yolo_module = org_scene.mask_detector
        mask_generation_st = time.time()
        detection_results = {}
        results_img = []
        for cam_id in org_scene.cam_ids:

            bgr_img = org_scene.get_image(cam_id, 0)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2BGR)
            detections = yolo_module.process_img(rgb_img)
            mask = detections.mask
            
            if len(mask)>0 and not check_mask(mask[0]):
                detections.confidence *= 0.0

            detection_results[cam_id] = detections

            if debug:
                canvas = yolo_module.annotate_image(bgr_img, detections, categories=yolo_module.categories, with_confidence=True)
                canvas = putText(canvas, cam_id, color=(0,0, 255))
                results_img.append(canvas)
                cv2.imwrite('test.png', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        if debug:
            cv2.imwrite(f'debug_grid_{obj_name}.png', make_grid_image_np(np.array(results_img), 4,6))

        confidence_dict = {cam_id: detection_results[cam_id].confidence.item() \
                           for cam_id in detection_results if detection_results[cam_id].confidence and detection_results[cam_id].confidence > confidence and cam_id not in hide_list}
        cam_N = 10
        top_n_cams2confidence = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)[:cam_N]
        top_n_cams = [cam_id for cam_id, confidence in top_n_cams2confidence]
        top_n_confidence = [confidence for cam_id, confidence in top_n_cams2confidence]

        mask_dict_org = {cam_id: get_binary_mask(detection_results[cam_id].mask[0]) for cam_id in top_n_cams}
        # mask_dict_scaled = {cam_id: cv2.resize( get_binary_mask(detection_results[cam_id].mask[0]).astype(np.uint8), dsize=(rescaled_scene.width, rescaled_scene.height), interpolation=cv2.INTER_LINEAR) for cam_id in top_n_cams}
    
        mask_generation_ed = time.time()
        print(f"GT mask generation {mask_generation_ed-mask_generation_st} sec")
    else:
        mask_dict_org = {}
        for cam_id in org_scene.cam_ids:
            mask = get_binary_mask(org_scene.get_mask(cam_id, fidx=0)[...,0])

            if check_mask(mask):
                mask_dict_org[cam_id] = mask
            

        top_n_cams = [key for key in mask_dict_org]
        top_n_confidence = [1 for key in mask_dict_org]

    
    return mask_dict_org, top_n_cams, top_n_confidence