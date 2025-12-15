import math
import cv2
import numpy as np

def get_optimal_font_scale(text, target_width, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
    """텍스트가 target_width에 맞는 최적의 font scale 찾기"""
    for scale in np.arange(0.3, 3.0, 0.1):
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        if text_size[0] > target_width:
            return max(0.3, scale - 0.1)  # 한 단계 작게
    return 3.0

def merge_image(image_dict, image_text={}, put_text=True):
    name_list = sorted(list(image_dict.keys()))
    num_images = len(name_list)
    
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)
    border_px = 10
    
    new_W = 2048 // grid_cols
    new_H = 1536 // grid_rows
    
    grid_image = np.ones((1536+border_px*(grid_rows-1), new_W*grid_cols+border_px*(grid_cols-1), 3), dtype=np.uint8) * 255

    for idx, img_name in enumerate(name_list):
        img = image_dict[img_name].copy()
        
        # 텍스트 준비
        
        # 이미지 너비의 90%에 맞는 font scale 자동 계산
        
        if put_text:
            target_width = int(img.shape[1] * 0.5)
            thickness = max(1, img.shape[1] // 500)  # 이미지 크기에 비례하는 두께
            txt = f"{img_name}"
            font_scale = get_optimal_font_scale(txt, target_width, thickness=thickness)
            if img_name in image_text:
                txt += f" {image_text[img_name]}"
            # 텍스트 크기 계산
            text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = 10
            text_y = text_size[1] + 10  # 텍스트 높이 + 여백
            
            # 텍스트 배경 (가독성 향상)
            cv2.rectangle(img, 
                        (text_x - 5, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5),
                        (0, 0, 0), -1)
            
            # 텍스트 그리기
            cv2.putText(img, txt, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
            
        # 리사이즈 및 배치
        resized_img = cv2.resize(img, (new_W, new_H))
        
        r_idx = idx // grid_cols
        c_idx = idx % grid_cols

        r_start = r_idx * (new_H + border_px)
        c_start = c_idx * (new_W + border_px)
        grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img
    
    return grid_image