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

def merge_image(
    image_dict,
    image_text={},
    put_text=True,
    grid_cols=None,
    preserve_aspect=False,
    target_height=None,
):
    name_list = sorted(list(image_dict.keys()))
    num_images = len(name_list)
    
    grid_cols = math.ceil(math.sqrt(num_images)) if grid_cols is None else int(grid_cols)
    if grid_cols <= 0:
        raise ValueError("grid_cols must be positive")
    grid_rows = math.ceil(num_images / grid_cols)
    border_px = 10
    
    new_W = 2048 // grid_cols
    if preserve_aspect:
        if target_height is None:
            new_H = max(
                max(
                    1,
                    int(
                        round(
                            new_W
                            * image_dict[name].shape[0]
                            / image_dict[name].shape[1]
                        )
                    ),
                )
                for name in name_list
            )
        else:
            available_height = int(target_height) - border_px * (grid_rows - 1)
            if available_height < grid_rows:
                raise ValueError("target_height is too small for the image grid")
            new_H = available_height // grid_rows
            # Let the output window become narrower instead of padding every
            # camera into the legacy 2048px-wide canvas.
            new_W = max(
                max(
                    1,
                    int(
                        round(
                            new_H
                            * image_dict[name].shape[1]
                            / image_dict[name].shape[0]
                        )
                    ),
                )
                for name in name_list
            )
        canvas_height = new_H * grid_rows
    else:
        new_H = 1200 // grid_rows #1536
        canvas_height = 1200
    
    grid_image = np.ones((canvas_height+border_px*(grid_rows-1), new_W*grid_cols+border_px*(grid_cols-1), 3), dtype=np.uint8) * 255

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
        if preserve_aspect:
            scale = min(new_W / img.shape[1], new_H / img.shape[0])
            resized_W = max(1, int(round(img.shape[1] * scale)))
            resized_H = max(1, int(round(img.shape[0] * scale)))
            resized = cv2.resize(img, (resized_W, resized_H))
            resized_img = np.ones((new_H, new_W, 3), dtype=np.uint8) * 255
            y_offset = (new_H - resized_H) // 2
            x_offset = (new_W - resized_W) // 2
            resized_img[
                y_offset:y_offset + resized_H,
                x_offset:x_offset + resized_W,
            ] = resized
        else:
            resized_img = cv2.resize(img, (new_W, new_H))
        
        r_idx = idx // grid_cols
        c_idx = idx % grid_cols

        r_start = r_idx * (new_H + border_px)
        c_start = c_idx * (new_W + border_px)
        grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img
    
    return grid_image
