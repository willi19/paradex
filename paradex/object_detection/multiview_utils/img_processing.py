import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import sys
from pathlib import Path
PROJECT_PATH = Path(__file__).parent.parent
sys.path.append(PROJECT_PATH)

from paradex.object_detection.obj_utils.vis_utils import overlay_mask, make_grid_image_np
from copy import deepcopy

SRC_COLOR = (255,0,0)
SET_COLOR = (0,0,255)

def make_square(img, background_color=(0,0,0)):
    h, w = img.shape[:2]
    square_size = max(h, w)
    square_img = np.full((square_size, square_size, 3), background_color, dtype=np.uint8)
    
    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2

    square_img[y_offset:y_offset+h, x_offset:x_offset+w]=img

    return square_img


def crop_with_mask(rgb_img, mask, output_L, padding=5, background_color=(0,0,0), only_foreground=False):
    """
    Crop an RGB image using a mask, add padding, and resize.

    Parameters:
    - output_size: tuple (width, height) for resizing
    - padding: number of pixels to pad around the mask
    - background_color: tuple (B,G,R) for padding background
    """

    # Ensure mask is binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find coordinates of the mask
    ys, xs, _ = np.where(mask > 0)

    # Calculate bounding box with padding
    x_min = max(xs.min() - padding, 0)
    x_max = min(xs.max() + padding, rgb_img.shape[1] - 1)
    y_min = max(ys.min() - padding, 0)
    y_max = min(ys.max() + padding, rgb_img.shape[0] - 1)

    mid_before = ((x_min+x_max)/2,(y_min+y_max)/2)
    
    # Crop the image
    cropped = rgb_img[y_min:y_max+1, x_min:x_max+1]
    mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]

    if only_foreground:
        canvas = np.full_like(cropped, background_color)
        mask_bool = mask_cropped>0
        canvas[mask_bool] = cropped[mask_bool]
        cropped = canvas


    squared_cropped_img = make_square(cropped)   
    squared_cropped_mask = make_square(mask_cropped)
    # Resize
    final_img_resized = cv2.resize(squared_cropped_img, (output_L,output_L), interpolation=cv2.INTER_AREA)
    final_mask_resized = cv2.resize(squared_cropped_mask, (output_L,output_L), interpolation=cv2.INTER_AREA)

    mid_after = (squared_cropped_img.shape[0]/2, squared_cropped_img.shape[0]/2)
    scale = output_L/squared_cropped_img.shape[0]
    similarity_T = np.array([
        [scale, 0, (mid_after[0]-mid_before[0]) * scale],
    [0, scale,  (mid_after[1]-mid_before[1]) * scale],
        [0,0,1]
    ], dtype=np.float32)

    return final_img_resized, final_mask_resized, similarity_T


import matplotlib.pyplot as plt
cmap = plt.get_cmap("jet") # jet, viridis, plasma, coolwarm

def get_length_color(normalized_length):
    color = cmap(normalized_length)[:3]  
    # 0~255 BGR 변환
    color_bgr = tuple(int(c*255) for c in color[::-1])  
    return color_bgr

def draw_pairs_wtext(canvas, pts1, pts2, normalized_lengths, text_input):

    for (p1, p2, l) in zip(pts1, pts2, normalized_lengths):
        # cmap(l)은 (R,G,B,A) → 0~1 float
        color = cmap(l)[:3]  
        # 0~255 BGR 변환
        color_bgr = tuple(int(c*255) for c in color[::-1])  
        cv2.line(canvas, tuple(p1), tuple(p2), color_bgr, 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text_input, font, scale, thickness)
    x = canvas.shape[1] - text_w - 10
    y = text_h + 10
    cv2.putText(canvas, text_input, (x, y), font, scale, (255, 0, 0), thickness, cv2.LINE_AA)
    
    return canvas


def draw_match(img0_resized, img1_resized, pair_output):
    def get_color(score):
        '''
            Map score to RGB color
            score:0~1
        '''
        r = int(255 * (1 - score))
        g = int(255 * score)
        b = 0
        # return (1-score, score, 0)
        return (r,g,b)

    height0, width0 = img0_resized.shape[:2]
    height1, width1 = img1_resized.shape[:2]
    plot_image = np.zeros((max(height0, height1), width0 + width1, 3), dtype=np.uint8)
    plot_image[:height0, :width0] = img0_resized
    plot_image[:height1, width0:] = img1_resized

    plot_image_pil = Image.fromarray(plot_image)
    draw = ImageDraw.Draw(plot_image_pil)

    keypoints0_x, keypoints0_y = pair_output["keypoints0"].astype(np.int64).T
    keypoints1_x, keypoints1_y = pair_output["keypoints1"].astype(np.int64).T
    
    if 'matching_scores' not in pair_output:
        pair_output['matching_scores'] = np.ones(len(keypoints0_x)).astype(np.float64)

    for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
        keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, pair_output["matching_scores"]
    ):
        if matching_score>0.5:
            color = get_color(matching_score)
            draw.line(
                (keypoint0_x, keypoint0_y, keypoint1_x + width0, keypoint1_y),
                fill=color,
                width=3,
            )
            draw.ellipse((keypoint0_x - 2, keypoint0_y - 2, keypoint0_x + 2, keypoint0_y + 2), fill="black")
            draw.ellipse(
                (keypoint1_x + width0 - 2, keypoint1_y - 2, keypoint1_x + width0 + 2, keypoint1_y + 2),
                fill="black",
            )

    return cv2.cvtColor(np.array(plot_image_pil), cv2.COLOR_BGR2RGB)
    

def draw_inliers(src_img, tg_img, src_keypoints, tg_keypoints, tmp_inlier_flag):

    # scores = np.ones((tg_keypoints.shape[0])).astype(np.float64)
    inlier_percentage = tmp_inlier_flag.sum()/len(tmp_inlier_flag)
    # print(f'Inlier Percentage from {len(tmp_inlier_flag)} number of points: {inlier_percentage}')
    pair_output = {'keypoints0':src_keypoints, 'keypoints1':tg_keypoints}
    match_img = draw_match(src_img, tg_img, pair_output)
    pair_output_inlier = {'keypoints0':src_keypoints[tmp_inlier_flag], 'keypoints1':tg_keypoints[tmp_inlier_flag]}
    match_img_inliers = draw_match(src_img, tg_img, pair_output_inlier)

    return match_img, match_img_inliers


def draw_text(canvas, text_input, position="center"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    thickness = 2
    color = (255, 0, 0)

    # 텍스트 크기 계산
    (text_w, text_h), baseline = cv2.getTextSize(text_input, font, scale, thickness)

    # 기본 y좌표 (상단)
    y = text_h + 10  

    # 위치별 x좌표 계산
    if position == "left":
        x = 10
    elif position == "center":
        x = (canvas.shape[1] - text_w) // 2
    elif position == "right":
        x = canvas.shape[1] - text_w - 10
    else:
        raise ValueError("position must be 'left', 'center', or 'right'")

    # 텍스트 그리기
    cv2.putText(canvas, text_input, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    return canvas


def rendersil_obj2allview(scene, obj_dict, obj_T, img_bucket, highlight:dict=None):
    
    # Transfer object dicationary 
    transformed_obj = deepcopy(obj_dict)
    transformed_verts = torch.einsum('mn, bjn -> bjm', obj_T[:3,:3], \
                                        transformed_obj['verts'].clone().detach())+ obj_T[:3,3]
    transformed_obj['verts'] = transformed_verts
    batch_rendered = scene.batch_render(transformed_obj)

    rendered_rgb, rendered_sil = batch_rendered
    rendered_sil = rendered_sil.squeeze()

    img_dict = {}
    imgs = []
    # visualize on image
    for cidx, cam_id in enumerate(scene.cam_ids):
        bgr_img = img_bucket[cam_id]

        mask=rendered_sil[cidx].detach().cpu().numpy()
        overlaid = overlay_mask(bgr_img, mask=(mask>0))
        overlaid =  draw_text(overlaid, cam_id)
        if highlight is not None and cam_id in highlight:
            h,w = overlaid.shape[:2]
            cv2.rectangle(overlaid, (0, 0), (w-1, h-1), highlight[cam_id], 10)
        imgs.append(overlaid)
        img_dict[cam_id] = overlaid

    return cv2.cvtColor(make_grid_image_np(np.stack(imgs), 4, 6),cv2.COLOR_BGR2RGB), img_dict
    


