import os
import numpy as np
import cv2

from paradex.utils.path import shared_dir
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import draw_aruco
from paradex.calibration.utils import load_current_C2R
from paradex.visualization.visualizer.viser import ViserViewer

c2r = load_current_C2R()

def transform_c2r(marker_pos):
    marker_pos_h = np.hstack([marker_pos, np.ones((marker_pos.shape[0], 1))])
    marker_pos_r = (np.linalg.inv(c2r) @ marker_pos_h.T).T[:, :3]
    return marker_pos_r

marker_ids = [518, 519, 520, 521]
# box 1
# 25 9.5 30.7
# 519 520

# box 2
# 32.5 26 30
# 518 521

img_dict = ImageDict.from_path(os.path.join(shared_dir, "RSS2026_Mingi", "test_shelf", "icecream_scoop", "20260130_232641"))
if not os.path.exists(os.path.join(shared_dir, "RSS2026_Mingi", "test_shelf", "icecream_scoop", "20260130_232641", "images")):
    img_dict.undistort()
    img_dict = ImageDict.from_path(os.path.join(shared_dir, "RSS2026_Mingi", "test_shelf", "icecream_scoop", "20260130_232641"))

kypt_2d, kypt_3d = img_dict.triangulate_markers()
# box 1
# 25 9.5 30.7
# 519 520
p1 = transform_c2r(kypt_3d[519])
p2 = transform_c2r(kypt_3d[520])
print(p1, p2)
# print(np.linalg.norm(p1.mean(axis=0) - p2.mean(axis=0)))
# vis = ViserViewer()