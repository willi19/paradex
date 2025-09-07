import numpy as np
import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from paradex.image.aruco import detect_aruco, triangulate_marker, draw_aruco
from paradex.utils.file_io import shared_dir, find_latest_directory, rsc_path, load_c2r, get_robot_urdf_path
from paradex.geometry.math import rigid_transform_3D
from paradex.robot import RobotWrapper
from paradex.utils.file_io import eef_calib_path, load_camparam
from paradex.image.projection import get_cammtx
from paradex.image.undistort import undistort_img
from paradex.geometry.Tsai_Lenz import solve, solve_axb_pytorch
from paradex.geometry.conversion import project
from paradex.geometry.coordinate import DEVICE2WRIST
from paradex.visualization_.renderer import BatchRenderer
from paradex.visualization_.robot_module import Robot_Module
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.image.overlay import overlay_mask

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(eef_calib_path)
    
eef_calib_path = os.path.join(shared_dir, "eef", args.name)
index_list = os.listdir(os.path.join(eef_calib_path))

robot = RobotWrapper(
    os.path.join(rsc_path, "robot", "allegro", "allegro.urdf")
)
arm = RobotWrapper(
    os.path.join(rsc_path, "robot", "xarm.urdf")
)

# finger_index["index_proximal"] = 8
# finger_index["middle_proximal"] = 8
# finger_index["ring_proximal"] = 28
# 11:"ring_proximal", 14:"middle_proximal", 13:"index_proximal"}#,

finger_marker = {464:"thumb_distal", 183:"ring_distal", 179:"index_distal", 0:"thumb_proximal", 1:"index_proximal", 2:"middle_proximal", 3:"ring_proximal", 4:"thumb_medial", 5:"index_medial", 6:"middle_medial", 7:"ring_medial"}
finger_id_list = list(finger_marker.keys())
finger_index = {link_name:robot.get_link_index(link_name) for link_name in set(finger_marker.values())}

c2r = load_c2r(os.path.join(eef_calib_path, "0"))
    
link_index = arm.get_link_index("link6")
intrinsic, extrinsic = load_camparam(os.path.join(eef_calib_path, "0"))
cammtx = get_cammtx(intrinsic, extrinsic)

robot_cor = []
cam_cor = []
hand_action = []
qpos = []

for idx in index_list:
    robot_cor.append(np.load(os.path.join(eef_calib_path, idx, "robot.npy")))
    hand_action.append(np.load(os.path.join(eef_calib_path, idx, "hand.npy")))
    qpos.append(np.load(os.path.join(eef_calib_path, idx, "qpos.npy")))
    
    arm.compute_forward_kinematics(qpos[-1])
    robot_cor.append(arm.get_link_pose(link_index))
    
    img_dir = os.path.join(eef_calib_path, idx, "image")
    
    img_dict = {}
    for img_name in os.listdir(img_dir):
        img_dict[img_name.split(".")[0]] = cv2.imread(os.path.join(img_dir, img_name))
    
    for serial_num, img in img_dict.items():
        if serial_num not in cammtx:
            continue
        
        undist_img = undistort_img(img, intrinsic[serial_num])
        
        os.makedirs(os.path.join(eef_calib_path, idx, "undistort"), exist_ok=True)
        cv2.imwrite(os.path.join(eef_calib_path, idx, "undistort", f"{serial_num}.png"), undist_img)
            
X = np.load("/home/temp_id/shared_data/eef/20250907_184610/0/eef.npy")

extrinsic_list = []
intrinsic_list = []

serial_list = os.listdir(os.path.join(eef_calib_path, idx, "image"))
serial_list.sort()

for serial_name in serial_list:
    sn = serial_name.split(".")[0]
    extmat = extrinsic[sn]
    extrinsic_list.append(extmat @ c2r)        
    intrinsic_list.append(intrinsic[sn]['intrinsics_undistort'])

hand_action = np.array(hand_action)
qpos = []
for i in range(len(index_list)):
    wrist_6d = robot_cor[i] @ X
    euler = R.from_matrix(wrist_6d[:3,:3]).as_euler('zyx')
    
    q = np.zeros(22)
    q[0:3] = wrist_6d[0:3,3]
    q[5] = euler[0]
    q[4] = euler[1]
    q[3] = euler[2]
    q[6] = -0.1# hand_action[i][4]
    q[7] = hand_action[i][1] + 0.296
    print(hand_action[i])
    qpos.append(q)
    # qpos.append(np.concatenate([wrist_6d[0:3,3], R.from_matrix(wrist_6d[0:3,0:3]).as_euler('zyx'), hand_action[i]], axis=0))
qpos = np.array(qpos).astype(np.float32)
# qpos = np.concatenate([qpos, hand_action], axis=1)
rm = Robot_Module(get_robot_urdf_path(None, "allegro"), state=qpos)

renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')

for fi, fid in enumerate(index_list):
    img_dir = os.path.join(eef_calib_path, fid, "undistort")
    robot_mesh_list = rm.get_mesh(int(fid))
    robot_mesh = robot_mesh_list[0]
    for i in range(1, len(robot_mesh_list)):
        # robot_mesh += robot_mesh_list[i]
        if "index" in rm.link_list[i]:
            robot_mesh += robot_mesh_list[i]
    # transform X open3d
    # robot_mesh.transform(X @ robot_cor[fi] )
    img_dict = {img_name:cv2.imread(os.path.join(img_dir, img_name)) for img_name in serial_list}
    os.makedirs(os.path.join(eef_calib_path, fid, "overlay"), exist_ok=True)
    
    frame, mask = project_mesh_nvdiff(robot_mesh, renderer)
    mask = mask.detach().cpu().numpy()[:,:,:,0].astype(np.bool_)
        
    for i, img_name in enumerate(serial_list):
        overlay_mask(img_dict[img_name], mask[i], 0.7, np.array((0, 255, 0)))
    
    for img_name in serial_list:
        cv2.imwrite(os.path.join(eef_calib_path, fid, "overlay", img_name), img_dict[img_name])    
        