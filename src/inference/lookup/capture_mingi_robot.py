import numpy as np
import cv2
import json
import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

from paradex.utils.file_io import shared_dir, load_current_camparam, load_latest_C2R
from paradex.model.yolo_world_module import YOLO_MODULE
from paradex.image.undistort import undistort_img
from paradex.image.merge import merge_image
from paradex.image.projection import get_cammtx

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

def draw_rotated_rectangle(image, rect_corners, color=(0, 255, 0), thickness=2):
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
        for i, corner in enumerate(rect_corners):
            cv2.circle(image, tuple(corner), 3, color, -1)
            cv2.putText(image, str(i), tuple(corner + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image

def visualize_mask_and_rectangle(image, mask, rect_corners, rect_info, save_path=None):
    """
    마스크와 최소 회전 사각형을 시각화하는 함수
    """
    # 이미지 복사
    vis_img = image.copy()
    
    # 마스크 오버레이 (반투명)
    if mask is not None:
        if mask.dtype == bool:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask.astype(np.uint8)
        
        # 마스크를 컬러로 변환
        mask_colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        
        # 반투명 오버레이
        vis_img = cv2.addWeighted(vis_img, 0.7, mask_colored, 0.3, 0)
    
    # 회전된 사각형 그리기
    if rect_corners is not None:
        vis_img = draw_rotated_rectangle(vis_img, rect_corners, (0, 255, 0), 2)
        
        # 사각형 정보 텍스트 추가
        if rect_info is not None:
            center, (width, height), angle = rect_info
            info_text = f"Size: {width:.1f}x{height:.1f}, Angle: {angle:.1f}°"
            cv2.putText(vis_img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img

def undistort_image_dir(root_path, intrinsic):
    if not os.path.exists(os.path.join(root_path, "undistorted_images")):
        img_list = os.listdir(os.path.join(root_path, "image"))
        os.makedirs(os.path.join(root_path, "undistorted_images"), exist_ok=True)
        for img_name in img_list:
            serial_num = img_name.split('.')[0]
            img = cv2.imread(os.path.join(root_path, "image", f"{serial_num}.png"))
            undistorted_img = undistort_img(img, intrinsic[serial_num])
            cv2.imwrite(os.path.join(root_path, "undistorted_images", f"{serial_num}.png"), undistorted_img)

# 메인 코드
obj_name = "pringles_case"
yolo_module = YOLO_MODULE(categories=obj_name)

intrinsic, extrinsic = load_current_camparam()
proj_mtx = get_cammtx(intrinsic, extrinsic)

for index in os.listdir(f"{shared_dir}/multi_pringles"):
    root_path = os.path.join(f"{shared_dir}/multi_pringles/{index}")
    
    undistort_image_dir(root_path, intrinsic)
    img_list = os.listdir(os.path.join(root_path, "undistorted_images"))
    
    C2R = load_latest_C2R()
    C2R = np.linalg.inv(C2R)  # convert to camera coordinate system
    
    device = torch.device("cuda:0")
    
    # DO YOLO
    detection_results = {}
    results_img = {}
    rectangle_info = {}  # 각 이미지의 회전 사각형 정보 저장
    
    print(f"Processing folder {index}...")
    
    for img_name in tqdm.tqdm(img_list):
        serial_num = img_name.split('.')[0]
        
        rgb_img = cv2.imread(os.path.join(root_path, "undistorted_images", img_name))
        
        detections = yolo_module.process_img(rgb_img, with_segmentation=True, top_1=False)
        
        if detections.xyxy.size > 0:
            detections.bbox_center = detections.xyxy[:, :2] + (detections.xyxy[:, 2:] - detections.xyxy[:, :2]) / 2
            
            # ===== 새로운 부분: 모든 마스크에서 최소 회전 사각형 계산 =====
            if hasattr(detections, 'mask') and detections.mask is not None:
                num_detections = detections.mask.shape[0]
                print(f"Camera {serial_num}: Found {num_detections} detections")
                
                # 각 detection별 회전 사각형 정보 저장
                rectangle_info[serial_num] = []
                
                for detection_idx in range(num_detections):
                    mask = detections.mask[detection_idx]  # shape: (H, W)
                    
                    # 최소 회전 사각형 계산
                    rect_corners, rect_info, axis_aligned_bbox, area_reduction = get_minimum_rotated_rectangle_from_mask(mask)
                    
                    if rect_corners is not None:
                        # 정보 출력
                        center, (width, height), angle = rect_info
                        print(f"  Detection {detection_idx}:")
                        print(f"    Rotated rectangle: {width:.1f} x {height:.1f}")
                        print(f"    Rotation angle: {angle:.1f}°")
                        print(f"    Area reduction vs axis-aligned: {area_reduction:.1f}%")
                        
                        # 회전 사각형 정보 저장
                        rect_data = {
                            'detection_idx': detection_idx,
                            'corners': rect_corners.tolist(),
                            'center': center,
                            'size': (width, height),
                            'angle': angle,
                            'area_reduction': area_reduction,
                            'original_bbox': detections.xyxy[detection_idx].tolist() if detection_idx < len(detections.xyxy) else None
                        }
                        rectangle_info[serial_num].append(rect_data)
                        
                        # 기존 bbox를 회전 사각형의 axis-aligned bbox로 업데이트 (선택사항)
                        # if detection_idx < len(detections.xyxy):
                        #     detections.xyxy[detection_idx] = np.array(axis_aligned_bbox)
                    else:
                        print(f"  Detection {detection_idx}: No valid mask found")
                
                # 빈 리스트면 제거
                if not rectangle_info[serial_num]:
                    del rectangle_info[serial_num]
        
        detection_results[serial_num] = detections
        
        # 시각화: 원본 + 마스크 + 회전 사각형
        canvas = yolo_module.annotate_image(rgb_img, detections, categories=yolo_module.categories, with_confidence=True)
        
        # 모든 회전된 사각형 그리기
        if serial_num in rectangle_info:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # 여러 색상
            
            for idx, rect_data in enumerate(rectangle_info[serial_num]):
                rect_corners = np.array(rect_data['corners'], dtype=np.int32)
                color = colors[idx % len(colors)]  # 색상 순환
                
                # 회전된 사각형 그리기
                canvas = draw_rotated_rectangle(canvas, rect_corners, color, 3)
                
                # 기존 bbox도 그리기 (비교용, 점선 스타일로)
                if rect_data['original_bbox'] is not None:
                    x1, y1, x2, y2 = [int(coord) for coord in rect_data['original_bbox']]
                    # 점선 효과를 위해 여러 작은 사각형으로 그리기
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
                
                # 정보 텍스트 추가 (detection별로)
                angle = rect_data['angle']
                area_reduction = rect_data['area_reduction']
                text_y = 30 + idx * 25
                cv2.putText(canvas, f"Det{idx}: {angle:.1f}° ({area_reduction:.1f}% smaller)", 
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 상세한 시각화 이미지도 저장 (모든 detection에 대해)
        if serial_num in rectangle_info and hasattr(detections, 'mask') and detections.mask is not None:
            os.makedirs(os.path.join("detailed_visualizations", str(index)), exist_ok=True)
            
            for idx, rect_data in enumerate(rectangle_info[serial_num]):
                detailed_vis = visualize_mask_and_rectangle(
                    rgb_img, detections.mask[rect_data['detection_idx']], 
                    np.array(rect_data['corners'], dtype=np.int32), 
                    (rect_data['center'], rect_data['size'], rect_data['angle'])
                )
                cv2.imwrite(os.path.join("detailed_visualizations", str(index), f"{serial_num}_det{idx}_detailed.png"), detailed_vis)
        
        os.makedirs(os.path.join("annotated_images", str(index)), exist_ok=True)
        cv2.imwrite(os.path.join("annotated_images", str(index), f"{serial_num}.png"), canvas)
        results_img[serial_num] = canvas.copy()
    
    # 회전 사각형 정보를 JSON으로 저장
    if rectangle_info:
        json_path = os.path.join(root_path, "rotated_rectangles.json")
        with open(json_path, 'w') as f:
            json.dump(rectangle_info, f, indent=2)
        print(f"Saved rotated rectangle info to {json_path}")
    
    grid_img = merge_image(results_img)
    cv2.imwrite(f'grid_{index}.png', grid_img)
    print(f"Processed folder {index}, saved grid_{index}.png")