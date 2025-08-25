import cv2
import numpy as np

def draw_rectangle(image, rect_corners, color=(0, 255, 0), thickness=2):
    """
    이미지에 회전된 사각형을 그리는 함수
    
    Args:
        image: 입력 이미지
        rect_corners: 사각형의 4개 모서리 좌표
        color: 선 색상 (B, G, R)
        thickness: 선 두께
    
    Returns:
        image: 사각형이 그려진 이미지
    """
    if rect_corners is not None:
        cv2.drawContours(image, [rect_corners], 0, color, thickness)
        
        # 모서리에 점 표시
        for corner in rect_corners:
            cv2.circle(image, tuple(corner), 3, color, -1)
            
    return image

def draw_mask(image, mask, color):
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    # 마스크 영역을 선명한 색상으로 칠하기
    mask_3channel = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    colored_mask = np.zeros_like(mask_3channel)
    colored_mask[mask_uint8 > 0] = color
    
    # 더 강한 오버레이 적용 (0.3 -> 0.6으로 증가)
    image = cv2.addWeighted(image, 1.0, colored_mask, 0.6, 0)

def draw_points(image, point, color=(255,255,0)):
    h, w, _ = image.shape
    for v in point:
        if v[0] > w or v[0] < 0 or v[1] > h or v[1] < 0:
            continue
        image = cv2.circle(image, (int(v[0]), int(v[1])), 5, color, -1)
    return image