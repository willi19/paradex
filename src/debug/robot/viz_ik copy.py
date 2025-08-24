import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import time

from paradex.simulator import IsaacSimulator

ik_data = {i:np.load(f"data/ik_{i}.npy", allow_pickle=True).item() for i in range(1, 5)}
ik_data[0] = np.load(f"data/ik_{0}.npy", allow_pickle=True).item()[0]
# sim = IsaacSimulator(headless=False)

# hand_name = "inspire"
# arm_name = "xarm"

# sim.load_robot_asset(arm_name, hand_name)
# sim.add_env("tmp",env_info = {"robot":{},
#                                 "robot_vis":{"new":(arm_name, hand_name)},
#                                 "object":{},
#                                 "object_vis":{}})
# ind = 5
# action = -1

# while True:    
#     for v, data in ik_data[ind].items():
#         if v == "pos":
#             continue
#         if data[1]:
#             action = data[0]
#             action[6:] = 0
#             sim.step("tmp", {"robot":{},
#                     "robot_vis":{"new":action},
#                     "object_vis":{}
#                     })
#             sim.tick()
#             time.sleep(0.3)
for ind in ik_data.keys():
    pos = ik_data[ind]["pos"]
    tz_dict = {}

    for v, data in ik_data[ind].items():
        if v == "pos":
            continue
        tx, ty, tz = v
        succ = data[1]
        recom_succ = data[2]

        if tz not in tz_dict:
            tz_dict[tz] = []
        tz_dict[tz].append((tx,ty,succ,recom_succ))
        # 시각화: tz별 이미지
    for tz in sorted(tz_dict.keys()):
        points = tz_dict[tz]
        plt.figure(figsize=(3,6))
        for tx, ty, success, recom_succ in points:
            # 스케일링 (필요 시, 예: 정규화 등)
            x = tx * 100  # 예: 이미지 정중앙 기준
            y = ty * 100
            color='red' 
            if (success and recom_succ):
                 color = 'green'
            elif success:
                color = 'blue'

            plt.scatter(x, y, color=color, s=4)
        plt.arrow(0,0,10,0, head_width=3, head_length=3, fc='blue', ec='blue', width=1.0)
        plt.title(f"IK Success Map @ tz={tz:.2f}")
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("tx")
        plt.ylabel("ty")
        os.makedirs(f"debug_ik/{ind}", exist_ok=True)
        plt.savefig(f"debug_ik/{ind}/ik_success_tz_{int(tz*100)}.png", dpi=200)
        plt.close()

for ind in ik_data.keys():
    # 해당 ind의 모든 tz 이미지를 찾음
    image_paths = sorted(glob.glob(f"debug_ik/{ind}/ik_success_tz_*.png"))
    images = [cv2.imread(p) for p in image_paths if os.path.exists(p)]

    # 모두 읽었는지 확인
    if len(images) == 0:
        print(f"[!] No images found for ind={ind}")
        continue

    # 높이가 다르면 resize (안 맞을 경우 오류 방지)
    min_height = min(img.shape[0] for img in images)
    resized_images = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in images]

    # 가로로 붙이기
    combined = cv2.hconcat(resized_images)

    # 저장
    cv2.imwrite(f"debug_ik/{ind}/combined.png", combined)
    print(f"[✓] Saved combined image for ind={ind}")