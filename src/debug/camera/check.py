import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import tqdm

from paradex.utils.file_io import shared_dir, load_current_camparam, load_latest_C2R
from paradex.image.merge import merge_image
from paradex.image.projection import get_cammtx, project_point

from paradex.visualization.shape import draw_mask, draw_rectangle, draw_points

import numpy as np
from scipy.spatial import ConvexHull
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def check_visibility(proj_mtx, voxel):
    count_grid = np.zeros((voxel.shape[1]), dtype=np.float32)
    for serial_num, proj in proj_mtx.items():
        w = 2048
        h = 1536
        image_points_homo = proj @ voxel

        valid_depth = image_points_homo[2, :] > 0
        image_points = image_points_homo[:2] / image_points_homo[2:3]
        image_points[:, valid_depth] = (image_points_homo[:2, valid_depth] / 
                                       image_points_homo[2:3, valid_depth])

        u_coords = image_points[0, :]
        v_coords = image_points[1, :]

        valid_u = (u_coords >= 0) & (u_coords < w)
        valid_v = (v_coords >= 0) & (v_coords < h)
        valid_projection = valid_depth & valid_u & valid_v
        count_grid[valid_projection] += 1
    
    return count_grid        

def get_visibility_color_turbo(count, max_count=24):
    """
    Turbo 색상맵 스타일 (파란색 -> 시안 -> 녹색 -> 노란색 -> 빨간색)
    Google의 Turbo 컬러맵을 모방
    """
    if count == 0:
        return (128, 128, 128)  # 회색
    
    ratio = 1 - count / max_count
    
    # Turbo 색상맵 근사 (5단계)
    if ratio < 0.2:
        # 진한 파란색 -> 밝은 파란색
        t = ratio / 0.2
        r = int(48 + (0 - 48) * t)
        g = int(18 + (90 - 18) * t)
        b = int(59 + (255 - 59) * t)
    elif ratio < 0.4:
        # 밝은 파란색 -> 시안
        t = (ratio - 0.2) / 0.2
        r = int(0 + (0 - 0) * t)
        g = int(90 + (255 - 90) * t)
        b = int(255 + (255 - 255) * t)
    elif ratio < 0.6:
        # 시안 -> 녹색
        t = (ratio - 0.4) / 0.2
        r = int(0 + (0 - 0) * t)
        g = int(255 + (255 - 255) * t)
        b = int(255 + (0 - 255) * t)
    elif ratio < 0.8:
        # 녹색 -> 노란색
        t = (ratio - 0.6) / 0.2
        r = int(0 + (255 - 0) * t)
        g = int(255 + (255 - 255) * t)
        b = int(0 + (0 - 0) * t)
    else:
        # 노란색 -> 빨간색
        t = (ratio - 0.8) / 0.2
        r = int(255 + (255 - 255) * t)
        g = int(255 + (0 - 255) * t)
        b = int(0 + (0 - 0) * t)
    
    return (b, g, r)  # BGR 순서

def create_reference_cube_image(axis, slice_value, search_range, texture_image, size=(150, 150)):
    """
    현재 슬라이스 위치를 보여주는 3D 큐브 참조 이미지 생성 (텍스트 잘림 해결)
    """
    # 여백을 고려한 figure 생성
    fig = plt.figure(figsize=(size[0]/10, size[1]/10), dpi=100)
    
    # 여백 설정으로 텍스트 잘림 방지
    ax = fig.add_subplot(111, projection='3d')
    
    # 큐브의 꼭짓점 정의
    x_range = [search_range[0][0], search_range[0][1]]
    y_range = [search_range[1][0], search_range[1][1]]  
    z_range = [search_range[2][0], search_range[2][1]]

    x_length = abs(x_range[1] - x_range[0])
    y_length = abs(y_range[1] - y_range[0])
    z_length = abs(z_range[1] - z_range[0])
    
    # 큐브 모서리 그리기
    vertices = np.array([[x_range[0], y_range[0], z_range[0]],
                        [x_range[1], y_range[0], z_range[0]],
                        [x_range[1], y_range[1], z_range[0]],
                        [x_range[0], y_range[1], z_range[0]],
                        [x_range[0], y_range[0], z_range[1]],
                        [x_range[1], y_range[0], z_range[1]],
                        [x_range[1], y_range[1], z_range[1]],
                        [x_range[0], y_range[1], z_range[1]]])
    
    # 큐브 면 정의
    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]],  # 바닥
             [vertices[4], vertices[5], vertices[6], vertices[7]],  # 위
             [vertices[0], vertices[1], vertices[5], vertices[4]],  # 앞
             [vertices[2], vertices[3], vertices[7], vertices[6]],  # 뒤
             [vertices[1], vertices[2], vertices[6], vertices[5]],  # 오른쪽
             [vertices[4], vertices[7], vertices[3], vertices[0]]]  # 왼쪽
    
    # 큐브 그리기
    cube = Poly3DCollection(faces, alpha=0.1, facecolor='lightblue', edgecolor='black', linewidth=0.5)
    ax.add_collection(cube)   

    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
    h, w = texture_image.shape[:2]
    texture_image = cv2.resize(texture_image,(w*2, h*2))
    # 이미지 크기 조정 (너무 크면 줄이기)
    # if texture_image.shape[0] > 256 or texture_image.shape[1] > 256:
    #     scale = min(256 / texture_image.shape[0], 256 / texture_image.shape[1])
    #     new_w = int(texture_image.shape[1] * scale)
    #     new_h = int(texture_image.shape[0] * scale)
    #     texture_image = cv2.resize(texture_image, (new_w, new_h))
    
    # 이미지를 0-1 범위로 정규화
    texture_image = texture_image.astype(np.float32) / 255.0
    
    # 3D 평면에 이미지 텍스처 적용
    # matplotlib의 plot_surface를 사용하여 텍스처 매핑
    if axis == 0:  # X축 슬라이스 (Y-Z 평면)
        Y, Z = np.meshgrid(np.linspace(y_range[0], y_range[1], texture_image.shape[1]),
                            np.linspace(z_range[1], z_range[0], texture_image.shape[0]))  # Z는 뒤집어야 함
        X = np.full_like(Y, slice_value)
        ax.plot_surface(X, Y, Z, facecolors=texture_image, alpha=1.0, antialiased=True)
        
    elif axis == 1:  # Y축 슬라이스 (X-Z 평면)
        X, Z = np.meshgrid(np.linspace(x_range[0], x_range[1], texture_image.shape[1]),
                            np.linspace(z_range[1], z_range[0], texture_image.shape[0]))
        Y = np.full_like(X, slice_value)
        ax.plot_surface(X, Y, Z, facecolors=texture_image, alpha=1.0, antialiased=True)
        
    else:  # Z축 슬라이스 (X-Y 평면)
        X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], texture_image.shape[1]),
                            np.linspace(y_range[0], y_range[1], texture_image.shape[0])) 
        Z = np.full_like(X, slice_value) 
        ax.plot_surface(X, Y, Z, facecolors=texture_image, alpha=1.0, antialiased=True)
            
    # 축 범위 설정
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_box_aspect([x_length, y_length, z_length])

    ax.set_xlabel('X (m)', fontsize=30, labelpad=40)
    ax.set_ylabel('Y (m)', fontsize=30, labelpad=40)
    ax.set_zlabel('Z (m)', fontsize=30, labelpad=25)
    
    ax.tick_params(axis='x', labelsize=20, pad=8)
    ax.tick_params(axis='y', labelsize=20, pad=8)
    ax.tick_params(axis='z', labelsize=20, pad=8)
    
    # 여백 최소화하면서 텍스트 공간 확보
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # 뷰 각도 설정 (보기 좋게)
    ax.view_init(elev=20, azim=225)
    
    # 이미지로 변환
    fig.canvas.draw()
    
    # ARGB 대신 RGB로 변경
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
    
    plt.close(fig)
    
    # BGR로 변환 (OpenCV 형식)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def add_ruler_axes(img, axis_names, search_range, redundant_axis, tick_interval_m=0.05):
    """
    이미지 가장자리에 눈금자 스타일의 축을 추가
    
    Args:
        img: 원본 이미지
        axis_names: 축 이름들 ['x', 'y', 'z']
        search_range: 각 축의 범위 (m 단위)
        redundant_axis: 현재 표시되는 두 축의 인덱스
        tick_interval_m: 눈금 간격 (m 단위)
    """
    h, w = img.shape[:2]
    img = cv2.resize(img, (w*2, h * 2))
    h, w = img.shape[:2]
    
    # 축 정보 계산
    axis0_range_m = search_range[redundant_axis[0]]  # 가로축 (이미지 x방향)
    axis1_range_m = search_range[redundant_axis[1]]  # 세로축 (이미지 y방향)
    
    axis0_length_m = (axis0_range_m[1] - axis0_range_m[0]) 
    axis1_length_m = (axis1_range_m[1] - axis1_range_m[0])
    
    # 픽셀당 m 계산
    pixels_per_m_x = w / axis0_length_m
    pixels_per_m_y = h / axis1_length_m
    
    # 축 영역을 위한 여백 크기
    margin_bottom = 60  # 가로축용
    margin_top = 10  # 가로축용

    margin_left = 60    # 세로축용
    margin_right = 10    # 세로축용
    
    # 새로운 캔버스 생성 (여백 포함)
    new_h = h + margin_bottom + margin_top
    new_w = w + margin_left + margin_right
    canvas = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    
    # 원본 이미지를 여백을 두고 배치
    canvas[margin_top:margin_top+h, margin_left:margin_left+w] = img
    
    # === 가로축 (하단) 그리기 ===
    axis_y = margin_top + h + 5  # 축의 y 위치
    
    # 주축선 그리기
    cv2.line(canvas, (margin_left, axis_y), (margin_left + w, axis_y), (0, 0, 0), 2)
    
    # 가로축 눈금 및 레이블
    axis0_start_m = axis0_range_m[0]
    axis0_end_m = axis0_range_m[1]
    
    # 눈금 위치 계산
    tick_start = int(np.ceil(axis0_start_m / tick_interval_m)) * tick_interval_m
    tick_positions_m = np.arange(tick_start, axis0_end_m + tick_interval_m, tick_interval_m)
    for tick_m in tick_positions_m:
        tick_m = round(tick_m, 1)
        if axis0_start_m <= tick_m <= axis0_end_m:
            # 픽셀 위치 계산
            pixel_x = margin_left + int((tick_m - axis0_start_m) * pixels_per_m_x)
            
            # 눈금선 그리기
            cv2.line(canvas, (pixel_x, axis_y - 5), (pixel_x, axis_y + 5), (0, 0, 0), 2)
            
            # 숫자 레이블
            label = f'{tick_m:.0f}' if tick_m.is_integer() else f'{tick_m:.1f}'
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = pixel_x - text_size[0] // 2
            cv2.putText(canvas, label, (text_x, axis_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 가로축 이름
    axis_name_x = axis_names[redundant_axis[0]].upper()
    cv2.putText(canvas, f'{axis_name_x} (m)', (margin_left + w // 2, axis_y + 30 + margin_top), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # === 세로축 (왼쪽) 그리기 ===
    axis_x = margin_left - 5  # 축의 x 위치
    
    # 주축선 그리기  
    cv2.line(canvas, (axis_x, margin_top), (axis_x, margin_top+h), (0, 0, 0), 2)
    
    # 세로축 눈금 및 레이블 (주의: 이미지는 y가 아래로 증가하지만 실제 좌표는 위로 증가)
    axis1_start_m = axis1_range_m[0] 
    axis1_end_m = axis1_range_m[1] 
    
    # 눈금 위치 계산
    tick_start = int(np.ceil(axis1_start_m / tick_interval_m)) * tick_interval_m
    tick_positions_cm = np.arange(tick_start, axis1_end_m + tick_interval_m, tick_interval_m)
    
    for tick_cm in tick_positions_cm:
        tick_m = round(tick_cm, 1)
        if axis1_start_m <= tick_cm <= axis1_end_m:
            # 픽셀 위치 계산 (y축은 뒤집혀 있음)
            pixel_y = h - int((tick_cm - axis1_start_m) * pixels_per_m_y) + margin_top
            
            # 눈금선 그리기
            cv2.line(canvas, (axis_x - 5, pixel_y), (axis_x + 5, pixel_y), (0, 0, 0), 2)
            
            # 숫자 레이블
            label = f'{tick_cm:.0f}' if tick_cm.is_integer() else f'{tick_cm:.1f}'
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = axis_x - text_size[0] - 5
            cv2.putText(canvas, label, (text_x, pixel_y + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 세로축 이름 (세로로 표시)
    axis_name_y = axis_names[redundant_axis[1]].upper()
    cv2.putText(canvas, f'{axis_name_y} (m)', (5, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # 원점 표시
    cv2.circle(canvas, (axis_x, axis_y), 3, (255, 0, 0), -1)
    
    return canvas

def create_colorbar_legend(color_list, max_count=23):
    """
    색상 범례 추가
    """
    color_l = 30
    h, w = color_l * (max_count + 1), color_l
    img = np.zeros((h, w, 3))
    # 색상 그라디언트
    for c in range(max_count + 1):
        img[color_l * c : color_l * (c+1), :, :] = color_list[max_count - c]
    return img

def add_colorbar(img, color_bar, width):
    h, w = img.shape[:2]
    h_color, w_color = color_bar.shape[:2]
    if h < h_color:
        # print(w, h, h_color, w_color, (h_color / h) * w)
        w = round((h_color / h) * w)
        h = h_color
        img = cv2.resize(img, (w, h))
    
    color_bar_img = np.ones((h, width, 3)) * 255

    buffer = (width - w_color) // 2
    buffer_h = (h - h_color) // 2
    color_bar_img[buffer_h:h_color + buffer_h, buffer:buffer + w_color] = color_bar

    img = np.concatenate([img, color_bar_img], axis=1)
    return img

def add_cubeimg(img, cube_img):
    h, w = img.shape[:2]
    h_cube, w_cube = cube_img.shape[:2]
    if h < h_cube:
        w = (h_cube / h) * w
        h = h_cube
        img = cv2.resize(img, (w, h))
    
    cube_add_img = np.ones((h, w_cube + 20, 3)) * 255

    buffer = (w - w_cube) // 2
    buffer_h = (h - h_cube) // 2
    cube_add_img[buffer_h:h_cube + buffer_h, 10:w_cube + 10] = cube_img

    img = np.concatenate([img, cube_add_img], axis=1)
    return img

# 메인 코드
intrinsic, extrinsic = load_current_camparam("20250812_162600")
proj_mtx = get_cammtx(intrinsic, extrinsic)
C2R = load_latest_C2R()

camera_points = []
for serial_num, ext in extrinsic.items():
    camera_center = - np.linalg.inv(ext[:3, :3]) @ ext[:3, 3]
    camera_center = np.linalg.inv(C2R[:3, :3]) @ camera_center + np.linalg.inv(C2R)[:3,3]
    camera_points.append(camera_center)
camera_points = np.array(camera_points)

search_range = [(0, 1.1), (-1.1, 1.1), (-0.06, 0.4)]
search_axis = [np.linspace(search_range[i][0], search_range[i][1], int((search_range[i][1] - search_range[i][0]) * 200) + 1) for i in range(3)]

X, Y, Z = np.meshgrid(search_axis[0], search_axis[1], search_axis[2])
robot_points_homo = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), np.ones(X.ravel().shape)])
world_points_homo = (C2R @ robot_points_homo)

count = check_visibility(proj_mtx, world_points_homo)

points = [camera_points]
label = ["camera"]

axis_names = ["x", "y", "z"]

color_list = [get_visibility_color_turbo(c, 23) for c in range(24)]
color_bar = create_colorbar_legend(color_list, 23)

for axis, axis_name in enumerate(axis_names):
    os.makedirs(f"debug/{axis_name}", exist_ok=True)
    redundant_axis = []
    for tmp in range(3):
        if tmp != axis:
            redundant_axis.append(tmp)
    
    value = np.round(robot_points_homo[axis] * 200).astype(np.int32)

    cor = robot_points_homo.copy()
    for tmp in range(2):
        cor[tmp] -= search_range[tmp][0]
    cor[2] = search_range[2][1] - cor[2]
    cor = (cor * 200).astype(np.int32)
    cor = cor[redundant_axis]
    
    for v in tqdm.tqdm(range(int(search_range[axis][0] * 200), int(search_range[axis][1] * 200 + 1))):
        ind = (value == v)
        w_img = int((search_range[redundant_axis[0]][1] - search_range[redundant_axis[0]][0])*200)+1
        h_img = int((search_range[redundant_axis[1]][1] - search_range[redundant_axis[1]][0])*200)+1
        
        # 메인 이미지 생성
        img = np.ones((h_img, w_img, 3), dtype=np.uint8) * 255
        
        if np.sum(ind) == 0:
            print(f"No points found for {axis_name}={v/200:.2f}")
            continue

        for c in range(24):
            color = color_list[c]
            pts_is = cor[:, ind & (count == c)].T
            img = draw_points(img, pts_is, color)

        
        # 현재 슬라이스 값 (cm 단위)
        slice_value_cm = v / 200.0 * 100  # m to cm
        slice_value_m = v / 200.0  # m 단위
        
        # 스케일 바 추가
        # img = add_ruler_axes(img, axis_names, search_range, redundant_axis, tick_interval_m=0.1)
        
        # # 참조 큐브 이미지 생성 및 합성
        img = create_reference_cube_image(axis, slice_value_m, search_range, img.copy())
        # img = add_cubeimg(img, cube_img)
        img = add_colorbar(img, color_bar, 50)
        filename = f"debug/{axis_name}/{v - round(search_range[axis][0]*200):03d}.png"
        cv2.imwrite(filename, img)