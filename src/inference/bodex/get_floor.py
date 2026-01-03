
from datetime import datetime
import os
import cv2
import numpy as np
import trimesh
import transforms3d

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import detect_aruco
from paradex.utils.path import shared_dir
from paradex.calibration.utils import save_current_camparam, load_current_C2R
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.io.robot_controller.gui_controller import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.path import rsc_path
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.robot.utils import get_robot_urdf_path

rcc = remote_camera_controller("object6d")

c2r = load_current_C2R()
    
filename = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(shared_dir, "inference", "grasp_eval", filename), exist_ok=True)

rcc.start("image", False, \
    f"shared_data/inference/grasp_eval/{filename}/raw")
rcc.stop()
rcc.end()

save_current_camparam(os.path.join(shared_dir, "inference", "grasp_eval", filename))

raw_img_dict = ImageDict.from_path(os.path.join(shared_dir, "inference", "grasp_eval", filename))
raw_img_dict.undistort(os.path.join(shared_dir, "inference", "grasp_eval", filename))

img_dict = ImageDict.from_path(os.path.join(shared_dir, "inference", "grasp_eval", filename))
charuco_3d = img_dict.triangulate_charuco()["1"]["checkerCorner"]
print(charuco_3d)
import pdb; pdb.set_trace()
charuco_3d_h = np.concatenate([charuco_3d, np.ones((charuco_3d.shape[0], 1))], axis=1)
charuco_3d_robot = (np.linalg.inv(c2r) @ charuco_3d_h.T).T[:, :3]

print(charuco_3d_robot)