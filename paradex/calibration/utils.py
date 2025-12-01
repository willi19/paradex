import json
import os
import numpy as np
import shutil
import cv2

from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_directory
from paradex.utils.system import config_dir

cam_param_dir = os.path.join(shared_dir, "cam_param")
handeye_calib_path = os.path.join(shared_dir, "handeye_calibration")
eef_calib_path = os.path.join(shared_dir, "eef")
extrinsic_dir = os.path.join(shared_dir, "extrinsic")

def load_current_camparam(name=None):
    if name == None:
        name = find_latest_directory(cam_param_dir)
    intrinsic_data = json.load(open(os.path.join(cam_param_dir, name, "intrinsics.json")))
    intrinsic = {}
    for serial, values in intrinsic_data.items():
        intrinsic[serial] = {
            "original_intrinsics": np.array(values["original_intrinsics"]).reshape(3, 3),
            "intrinsics_undistort": np.array(values["intrinsics_undistort"]).reshape(3, 3),
            "dist_params": np.array(values["dist_params"]),
            "height": values["height"],  # Scalar values remain unchanged
            "width": values["width"],
        }
        
    extrinsic_data = json.load(open(os.path.join(cam_param_dir, name, "extrinsics.json")))
    extrinsic = {}
    for serial, values in extrinsic_data.items():
        extrinsic[serial] = np.array(values).reshape(3, 4)
    return intrinsic, extrinsic

def load_camparam(demo_path):
    intrinsic_data = json.load(open(os.path.join(demo_path, "cam_param", "intrinsics.json"), "r"))
    intrinsic = {}
    for serial, values in intrinsic_data.items():
        intrinsic[serial] = {
            "original_intrinsics": np.array(values["original_intrinsics"]).reshape(3, 3),
            "intrinsics_undistort": np.array(values["intrinsics_undistort"]).reshape(3, 3),
            "dist_params": np.array(values["dist_params"]),
            "height": values["height"],  # Scalar values remain unchanged
            "width": values["width"],
        }
    extrinsic_data = json.load(open(os.path.join(demo_path, "cam_param", "extrinsics.json"), "r"))
    extrinsic = {}
    for serial, values in extrinsic_data.items():
        extrinsic[serial] = np.array(values).reshape(3, 4)
    return intrinsic, extrinsic

def load_current_intrinsic():
    intrinsics = {}
    intrinsic_path = os.path.join(shared_dir, "intrinsic")
    cam_list = os.listdir(intrinsic_path)
    for cam_name in cam_list:
        param_path = os.path.join(intrinsic_path, cam_name, "param")
        if os.path.exists(param_path) and len(os.listdir(param_path)) > 0:
            param_file = find_latest_directory(param_path)
            param = json.load(open(os.path.join(param_path, param_file), "r"))
            cammtx = np.array(param["K"]).reshape(3, 3)
            dist_params = np.array(param["distortion"]).reshape(1, 5)
            w, h = param["width"], param["height"]

            new_cammtx, roi = cv2.getOptimalNewCameraMatrix(cammtx, dist_params, (w, h), 1, (w, h))
            intrinsics[cam_name] = {
                "original_intrinsics": cammtx,
                "intrinsics_undistort": new_cammtx,
                "intrinsics_warped": new_cammtx,
                "dist_params": dist_params,
                "height": h,  # Scalar values remain unchanged
                "width": w,
            }
    return intrinsics

def load_c2r(demo_path):
    C2R = np.load(os.path.join(demo_path, "C2R.npy"))
    return C2R

def load_eef(demo_path):
    eef= np.load(os.path.join(demo_path, "eef.npy"))
    return eef

def save_current_camparam(save_path):
    camparam_dir = os.path.join(shared_dir, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_dir, "cam_param", camparam_name)

    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"), dirs_exist_ok=True)
    
def load_current_C2R():
    name = find_latest_directory(handeye_calib_path)
    index_list = sorted(os.listdir(os.path.join(handeye_calib_path, name)))
    
    return load_c2r(os.path.join(handeye_calib_path, name, index_list[0]))

def save_current_C2R(save_path):
    c2r = load_current_C2R()
    np.save(os.path.join(save_path, "C2R.npy"), c2r)
    
def load_current_eef():
    name = find_latest_directory(eef_calib_path)
    return load_eef(os.path.join(eef_calib_path, name, "0"))

def get_cammtx(intrinsic, extrinsic):
    cammat = {}
    for serial_num in list(intrinsic.keys()):
        int_mat = intrinsic[serial_num]["intrinsics_undistort"]
        ext_mat = extrinsic[serial_num]
        cammat[serial_num] = int_mat @ ext_mat
    return cammat

def get_handeye_calib_traj(arm_name: str):
    return os.path.join(config_dir, "hecalib", arm_name)