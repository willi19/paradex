import os
import numpy as np

from paradex.utils.path import shared_dir

index = "1"
# lookup_table_path = os.path.join(shared_dir, "capture", "lookup")
index_path = os.path.join("lookup", "pringles", index)

pick_traj = np.load(f"{index_path}/refined_pick_action.npy")
place_traj = np.load(f"{index_path}/refined_place_action.npy")

pick_hand_traj = np.load(f"{index_path}/refined_pick_hand.npy")
place_hand_traj = np.load(f"{index_path}/refined_place_hand.npy")