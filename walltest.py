import os
import numpy as np
import cv2

from paradex.utils.path import shared_dir
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import draw_aruco
from paradex.calibration.utils import load_current_C2R

marker_ids = [522, 523]  # 위쪽 양끝

img_dict = ImageDict.from_path(os.path.join(shared_dir, "RSS2026_Mingi", "demo_wall_test", "20260128_230457"))
if not os.path.exists(os.path.join(shared_dir, "RSS2026_Mingi", "demo_wall_test", "20260128_230457", "images")):
    img_dict.undistort()

img_dict = ImageDict.from_path(os.path.join(shared_dir, "RSS2026_Mingi", "demo_wall_test", "20260128_230457"))
kypt_2d, kypt_3d = img_dict.triangulate_markers()

# 1. 위쪽 marker 중점들
marker_centers = []
for mid, pts3d in kypt_3d.items():
    if mid in marker_ids:
        marker_centers.append(pts3d.mean(axis=0))
marker_centers = np.array(marker_centers)

# 2. z축은 robot base 기준
c2r = load_current_C2R()
z_dir = c2r[:3, 2]
z_dir = z_dir / np.linalg.norm(z_dir)
origin = c2r[:3, 3]

# 3. x축: 두 marker 연결 방향
x_dir = marker_centers[1] - marker_centers[0]
x_dir = x_dir / np.linalg.norm(x_dir)

# 4. plane normal (두께 방향)
normal = np.cross(x_dir, z_dir)
normal = normal / np.linalg.norm(normal)

# 5. 좌표계
u = x_dir
v = z_dir
w = -normal  # 두께 방향

# 6. 경계
margin = 0.03
thickness = 0.01  # 1cm 두께

top_centers_local_u = np.dot(marker_centers - marker_centers.mean(axis=0), u)
min_u = top_centers_local_u.min() - margin
max_u = top_centers_local_u.max() + margin

marker_z = np.dot(marker_centers.mean(axis=0) - origin, v) + margin

center_x = marker_centers.mean(axis=0)
center_x = center_x - np.dot(center_x - origin, v) * v

# 7. Cuboid 8개 꼭짓점 (앞면 4개 + 뒷면 4개)
front_corners = np.array([
    center_x + min_u * u,
    center_x + max_u * u,
    center_x + max_u * u + marker_z * v,
    center_x + min_u * u + marker_z * v,
])
back_corners = front_corners + thickness * w

# 8. 6개 면 정의 (각 면은 4개 꼭짓점)
faces = [
    front_corners[[0, 1, 2, 3]],  # 앞면
    back_corners[[1, 0, 3, 2]],   # 뒷면 (반대 순서)
    np.array([front_corners[0], back_corners[0], back_corners[3], front_corners[3]]),  # 왼쪽
    np.array([front_corners[1], front_corners[2], back_corners[2], back_corners[1]]),  # 오른쪽
    np.array([front_corners[3], front_corners[2], back_corners[2], back_corners[3]]),  # 위
    np.array([front_corners[0], back_corners[0], back_corners[1], front_corners[1]]),  # 아래
]

# 9. 각 이미지에 overlay
for mid, pts3d in kypt_3d.items():
    if mid in marker_ids:
        pts2d = img_dict.project_pointcloud(pts3d)
        for serial, pts in pts2d.items():
            img_dict.images[serial] = draw_aruco(img_dict.images[serial], pts.reshape(-1, 4, 2), [mid])

for serial in img_dict.images.keys():
    overlay = img_dict.images[serial].copy()
    
    # 각 면 그리기
    for face in faces:
        pts2d = img_dict.project_pointcloud(face)[serial].astype(np.int32)
        cv2.fillPoly(overlay, [pts2d], color=(0, 255, 0))
    
    img_dict.images[serial] = cv2.addWeighted(overlay, 0.3, img_dict.images[serial], 0.7, 0)
    
    # 엣지 그리기
    all_corners = np.vstack([front_corners, back_corners])
    pts2d = img_dict.project_pointcloud(all_corners)[serial].astype(np.int32)
    
    # 앞면 엣지
    for i in range(4):
        cv2.line(img_dict.images[serial], tuple(pts2d[i]), tuple(pts2d[(i+1)%4]), (0, 255, 0), 2)
    # 뒷면 엣지
    for i in range(4):
        cv2.line(img_dict.images[serial], tuple(pts2d[4+i]), tuple(pts2d[4+(i+1)%4]), (0, 255, 0), 2)
    # 연결 엣지
    for i in range(4):
        cv2.line(img_dict.images[serial], tuple(pts2d[i]), tuple(pts2d[4+i]), (0, 255, 0), 2)

img_dict.save("cuboid_overlay")