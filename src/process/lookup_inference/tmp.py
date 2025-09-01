import cv2
import numpy as np
import os
import trimesh

from paradex.utils.file_io import shared_dir, rsc_path

def draw_cross(img, center, size=15, color=(0,0,0), thickness=2):
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)  # 가로선
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)  # 세로선
    
def is_stand(obj_T):
    return obj_T[2,2] > 0.7

def get_position(obj_T):
    return (800 - int(obj_T[1, 3] * 1000), 800 - int(obj_T[0, 3] * 1000))

root_dir = os.path.join(shared_dir, "inference", "lookup", "pringles")
mesh = trimesh.load(os.path.join(rsc_path, "object", "pringles","pringles.obj"))

exp_dict = {
    "left":list(range(0, 20, 2)),
    "right":list(range(1, 20, 2)),
    "random_stand":list(range(42, 52)),
    "fixed_fallen":list(range(22, 32)),
    "random_fallen":list(range(32, 42)),
    "stack_stand":list(range(52,62)),
    "palm(27)":list(range(63, 70)),
    "palm(1)":list(range(70, 77)),
    "tripod(54)":list(range(77, 84)),    
    "tripod(9)":list(range(84, 91)),
    "tip(2)":list(range(91, 98)),
    "tip(35)":list(range(98, 105)),
    "palm(1)_heavy":list(range(105, 112)),
    "tripod(9)_heavy":list(range(112, 117)),
    "tip(35)_heavy":list(range(117, 122)),
    "lay_heavy":list(range(122, 124)),
    "lay_light":list(range(124, 129)),
    "tip(35)_light":list(range(129, 135)),
    "tripod(9)_light":list(range(135, 140)),
    "palm(1)_light":list(range(140, 145))
}

succ = {name:0 for name in list(exp_dict.keys())}
precision = {name:0 for name in list(exp_dict.keys())}


for exp_name, index_list in exp_dict.items():
    top_view = np.ones((800, 1600, 3), dtype=np.uint8) * 255
    cv2.arrowedLine(top_view, (800, 750), (800, 600), (0,0,0), 5, tipLength=0.25)
    
    for index in index_list:
        index_path = os.path.join(root_dir, str(index))
        success = True
        
        if not os.path.exists(os.path.join(index_path, "place_6D.npy")):
            success = False
        
        else:
            cur_6D = np.load(os.path.join(index_path, "place_6D.npy"))
            place_6D = np.load(os.path.join(index_path, "target_6D.npy"))
        
            if not is_stand(cur_6D):
                success = False
                
        # plot pick position
        pick_6D = np.load(os.path.join(index_path, "pick_6D.npy"))
        pick_pos = get_position(pick_6D)
        
        color = (0, 255, 0) if success else (0, 0 ,255)
        target_color = (255, 0, 0)
        
        if pick_6D[2, 2] < 0.7:  # 넘어진 상태
            w, h = 192, 70
            z_dir = pick_6D[:2,2]
            z_per = np.array([-z_dir[1], z_dir[0]])
            
            top_left = pick_pos - w/2 * z_dir - h/2 * z_per
            top_right = pick_pos - w/2 * z_dir + h/2 * z_per
            bottom_left = pick_pos + w/2 * z_dir - h/2 * z_per
            bottom_right = pick_pos + w/2 * z_dir + h/2 * z_per
            # Convert to integer coordinates
            corners = np.array([
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ], dtype=np.int32)
            # print(corners)
            # Draw the rotated rectangle
            cv2.polylines(top_view, [corners], True, color, 2)
        else:  # 서 있는 상태
            cv2.circle(top_view, pick_pos, 35, color, 2)

        if not success:
            continue
        
        cur_pos = get_position(cur_6D)  # 스케일 조정
        place_pos = get_position(place_6D)
        cv2.circle(top_view, cur_pos, 2, color, -1)
        # draw_cross(top_view, cur_pos, size=10, color=(0,0,255), thickness=2)

        cv2.circle(top_view, place_pos, 35, target_color, 1)
        draw_cross(top_view, place_pos, size=10, color=target_color, thickness=1)

        # 거리 계산 (mm 단위)
        distance_mm = np.linalg.norm(np.array(cur_6D[:2,3]) - np.array(place_6D[:2,3])) * 1000
        
        succ[exp_name] += 1
        precision[exp_name] += distance_mm

        # 두 점 사이에 선 그리기
        cv2.line(top_view, cur_pos, pick_pos, color, 1)
        cv2.line(top_view, cur_pos, place_pos, target_color, 1)

    # 텍스트 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # 색상 설명
    cv2.putText(top_view, f"Exp : {exp_name}", (50, 50), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(top_view, "Red: Current Position", (50, 90), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(top_view, "Blue: Target Position", (50, 140), font, font_scale, target_color, thickness)

    # 거리 정보
    if succ[exp_name] != 0:
        cv2.putText(top_view, f"Distance: {precision[exp_name] / succ[exp_name]:.1f} mm", (50, 650), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(top_view, "Scale: 0.8m x 1.6m", (50, 750), font, 0.8, (0, 0, 0), 1)
    cv2.putText(top_view, f"Success rate : {succ[exp_name] / len(index_list)}", (50, 700), font, 0.8, (0, 0, 0), 1)
    
    cv2.imwrite(f"{exp_name}.png", top_view)
    if succ[exp_name] == 0:
        print(exp_name, succ[exp_name] / len(index_list))
        continue    
    print(exp_name, succ[exp_name] / len(index_list), precision[exp_name] / succ[exp_name])