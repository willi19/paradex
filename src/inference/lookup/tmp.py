import cv2
import numpy as np
import os

from paradex.utils.file_io import shared_dir

root_dir = os.path.join(shared_dir, "inference", "lookup", "pringles")

succ = [0, 0]
precision = [0, 0]

def draw_cross(img, center, size=15, color=(0,0,0), thickness=2):
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)  # 가로선
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)  # 세로선
    
for index in os.listdir(root_dir):
    top_view = np.ones((800, 1600, 3), dtype=np.uint8) * 255
    if not os.path.exists(os.path.join(root_dir, index, "place_6D.npy")):
        continue
    
    cur_6D = np.load(os.path.join(root_dir, index, "place_6D.npy"))
    place_6D = np.load(os.path.join(root_dir, index, "target_6D.npy"))
    
            
    cur_pos = (1600 - int(cur_6D[1,3]*1000), 800 - int(cur_6D[0,3]*1000))  # 스케일 조정
    place_pos = (1600 - int(place_6D[1,3]*1000), 800 - int(place_6D[0,3]*1000))

    success = "success" if cur_6D[2,2] > 0.7 else "fail"
    cv2.circle(top_view, cur_pos, 35, (0,0,255), 1)
    draw_cross(top_view, cur_pos, size=15, color=(0,0,255), thickness=2)


    cv2.circle(top_view, place_pos, 35, (0,255,0), 1)
    draw_cross(top_view, place_pos, size=15, color=(0,255,0), thickness=2)

    # 거리 계산 (mm 단위)
    distance_mm = np.linalg.norm(np.array(cur_6D[:2,3]) - np.array(place_6D[:2,3])) * 1000
    if cur_6D[2,2] > 0.7:
        succ[int(index)%2] += 1
        precision[int(index)%2] += distance_mm

    # 두 점 사이에 선 그리기
    cv2.line(top_view, cur_pos, place_pos, (255, 255, 255), 2)

    # 텍스트 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # 색상 설명
    cv2.putText(top_view, "Red: Current Position", (50, 50), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(top_view, "Green: Target Position", (50, 90), font, font_scale, (0, 255, 0), thickness)

    # 거리 정보
    cv2.putText(top_view, f"Distance: {distance_mm:.1f} mm", (50, 130), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(top_view, "Scale: 0.8m x 1.6m", (50, 750), font, 0.8, (128, 128, 128), 1)
    cv2.putText(top_view, f"{success}", (50, 700), font, 0.8, (128, 128, 128), 1)
    cv2.arrowedLine(top_view, (800, 750), (800, 600), (0,0,0), 5, tipLength=0.25)

    cv2.imwrite(f"{root_dir}/{index}/top_view.png",top_view)
    
for i in range(2):
    print(succ[i], precision[i] / succ[i])