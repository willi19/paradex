import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import time
import tqdm

from paradex.simulator import IsaacSimulator
from collections import defaultdict

ik_data = {i:np.load(f"data/ik_{i}.npy", allow_pickle=True).item() for i in range(1, 5)}
ik_data[0] = np.load(f"data/ik_{0}.npy", allow_pickle=True).item()[0]
sim = IsaacSimulator(headless=True)

hand_name = "inspire"
arm_name = "xarm"

sim.load_robot_asset(arm_name, None)

for ind in range(5):
    sim.add_env(str(ind), env_info = {"robot":{},
                                    "robot_vis":{"new":(arm_name, None)},
                                    "object":{},
                                    "object_vis":{}})
    del ik_data[ind]["pos"]  # Remove the 'pos' key to avoid processing it
    sim.load_camera(str(ind))
    sim.set_videopath(str(ind), f"ik_vis")

# Step 1: group (tz, tx) → [(v, data), ...]
grouped = defaultdict(list)
for v, data in ik_data[0].items():
    if v == "pos":
        continue
    grouped[(v[2], v[0])].append(v)  # group by tz, tx
for tz, tx in sorted(grouped.keys())[:50]:
    group = grouped[(tz, tx)]
    # Step 3: sort by ty, left→right if tx even, right→left if tx odd
    reverse = (int(tx * 50) % 2 == 0)  # float → int to handle float parity
    sorted_group = sorted(group, key=lambda x: x[1], reverse=reverse)
    grouped[(tz, tx)] = sorted_group  # Update the group with sorted values

# Step 2: sort by tz, then tx
for tz, tx in tqdm.tqdm(sorted(grouped.keys())[:50]):
    sorted_group = grouped[(tz, tx)]
    
    for v in tqdm.tqdm(sorted_group):
        for ind in range(5):
            data = ik_data[ind][v]
            succ = data[1]
            if not succ:
                continue
            action = data[0][:6]
            action[6:] = 0
            sim.step(str(ind), {
                "robot": {},
                "robot_vis": {"new": action},
                "object_vis": {}
            })
        sim.tick()
        # time.sleep(0.03)
sim.terminate()