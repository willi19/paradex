import cv2
import numpy as np

def get_minimum_rotated_rectangle_from_mask(mask):
    """
    Segmentation mask로부터 최소 회전 사각형을 구하는 함수
    
    Args:
        mask: 2D boolean array 또는 0/255 값을 가진 이미지
    
    Returns:
        rect_corners: 회전된 사각형의 4개 모서리 좌표 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        rect_info: (center, (width, height), angle) - cv2.minAreaRect 결과
        axis_aligned_bbox: [x1, y1, x2, y2] - 축에 정렬된 bounding box
        area_reduction: 면적 감소율 (%)
    """
    
    # mask를 uint8 타입으로 변환
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, 0
    
    # 가장 큰 컨투어 선택 (mask에서는 보통 하나지만 안전하게)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 최소 회전 사각형 계산
    rect_info = cv2.minAreaRect(largest_contour)
    center, (width, height), angle = rect_info
    
    # 사각형의 4개 모서리 좌표 계산
    rect_corners = cv2.boxPoints(rect_info)
    rect_corners = rect_corners.astype(np.int32)  # 정수로 변환
    
    # 축에 정렬된 bounding box 계산 (비교용)
    x_coords = rect_corners[:, 0]
    y_coords = rect_corners[:, 1]
    axis_aligned_bbox = [np.min(x_coords), np.min(y_coords), 
                        np.max(x_coords), np.max(y_coords)]
    
    # 면적 비교
    rotated_area = width * height
    aligned_area = (axis_aligned_bbox[2] - axis_aligned_bbox[0]) * (axis_aligned_bbox[3] - axis_aligned_bbox[1])
    area_reduction = ((aligned_area - rotated_area) / aligned_area * 100) if aligned_area > 0 else 0
    
    return rect_corners, rect_info, axis_aligned_bbox, area_reduction
