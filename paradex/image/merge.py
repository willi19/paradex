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


def merge_image(image_dict, image_text=None, put_text=True, max_total_size=(2048, 1536), border_px=10):
    """Merge multiple images into a grid while preserving aspect ratio.

    - The grid cell size is computed from max_total_size divided by grid cols/rows.
    - Images are resized to fit into the cell while preserving aspect ratio.
    - Images are not upscaled beyond their native size (to avoid over-blurring).
    - Each image is centered inside a same-size padded cell.

    Args:
        image_dict: dict{name: image(np.ndarray)}
        image_text: dict{name: text}
        put_text: whether to overlay text on each image
        max_total_size: (max_width, max_height) of the final canvas
        border_px: pixels between cells
    """
    if image_text is None:
        image_text = {}

    name_list = sorted(list(image_dict.keys()))
    num_images = len(name_list)
    if num_images == 0:
        return np.ones((10, 10, 3), dtype=np.uint8) * 255

    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)

    max_W, max_H = max_total_size

    # Determine cell target size based on overall max and grid division
    cell_w_target = max_W // grid_cols
    cell_h_target = max_H // grid_rows

    # Don't force enlargement: cap cell target to the maximum native image size
    max_input_w = max(int(image_dict[n].shape[1]) for n in name_list)
    max_input_h = max(int(image_dict[n].shape[0]) for n in name_list)
    cell_w_target = min(cell_w_target, max_input_w)
    cell_h_target = min(cell_h_target, max_input_h)

    canvas_h = cell_h_target * grid_rows + border_px * (grid_rows - 1)
    canvas_w = cell_w_target * grid_cols + border_px * (grid_cols - 1)
    grid_image = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for idx, img_name in enumerate(name_list):
        img = image_dict[img_name].copy()
        h, w = img.shape[:2]

        # Compute scale to fit into cell, preserving aspect ratio. Do not upscale (>1).
        scale = min(cell_w_target / w, cell_h_target / h, 1.0)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(img, (new_w, new_h))

        # Prepare cell background and center the resized image
        cell = np.ones((cell_h_target, cell_w_target, 3), dtype=np.uint8) * 255
        y0 = (cell_h_target - new_h) // 2
        x0 = (cell_w_target - new_w) // 2
        cell[y0:y0+new_h, x0:x0+new_w] = resized

        # Draw text onto the cell (scale relative to cell size)
        if put_text:
            txt = img_name
            if img_name in image_text:
                txt += f" {image_text[img_name]}"

            # choose font scale based on cell width
            thickness = max(1, cell_w_target // 500)
            target_width = int(cell_w_target * 0.5)
            font_scale = get_optimal_font_scale(txt, target_width, thickness=thickness)
            text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = 10
            text_y = text_size[1] + 10

            # text background
            cv2.rectangle(cell, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(cell, txt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

        r_idx = idx // grid_cols
        c_idx = idx % grid_cols

        r_start = r_idx * (cell_h_target + border_px)
        c_start = c_idx * (cell_w_target + border_px)
        grid_image[r_start:r_start+cell_h_target, c_start:c_start+cell_w_target] = cell

    print(grid_image.shape, flush=True)
    return grid_image