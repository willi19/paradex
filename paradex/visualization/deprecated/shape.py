import cv2
import numpy as np

from paradex.geometry.conversion import project


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

def draw_pose_axes(img, robot_pose, camera_matrix, axis_length=0.01, text=None, color=(255,0,0)):
    """
    Draw robot pose as 3D coordinate axes on the image
    
    Args:
        img: Input image
        robot_pose: 4x4 robot transformation matrix  
        camera_matrix: Camera projection matrix (3x4)
        axis_length: Length of coordinate axes in meters
    """
    # Define coordinate axes in 3D
    origin = np.array([0, 0, 0, 1])  # Origin point
    x_axis = np.array([axis_length, 0, 0, 1])  # X axis (red)
    y_axis = np.array([0, axis_length, 0, 1])  # Y axis (green) 
    z_axis = np.array([0, 0, axis_length, 1])  # Z axis (blue)
    
    # Transform axes to robot coordinate frame
    axes_3d = np.array([origin, x_axis, y_axis, z_axis]).T  # 4x4
    axes_robot = robot_pose @ axes_3d  # Transform to world coordinates
    
    # Project to 2D image coordinates
    axes_2d = project(camera_matrix, axes_robot[:3].T)  # Project 3D points
    
    if len(axes_2d) == 4:  # Ensure all points projected successfully
        origin_2d = tuple(map(int, axes_2d[0]))
        x_end_2d = tuple(map(int, axes_2d[1])) 
        y_end_2d = tuple(map(int, axes_2d[2]))
        z_end_2d = tuple(map(int, axes_2d[3]))
        
        # Draw coordinate axes with different colors
        cv2.arrowedLine(img, origin_2d, x_end_2d, (0, 0, 255), 3)    # X: Red
        cv2.arrowedLine(img, origin_2d, y_end_2d, (0, 255, 0), 3)    # Y: Green  
        cv2.arrowedLine(img, origin_2d, z_end_2d, (255, 0, 0), 3)    # Z: Blue
        
        # Add text labels
        cv2.putText(img, 'X', x_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, 'Y', y_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
        cv2.putText(img, 'Z', z_end_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.circle(img, origin_2d, 5, color, -1)
        if text is not None:
            cv2.putText(img, text, (origin_2d[0]+30, origin_2d[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)