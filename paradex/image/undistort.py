import numpy as np
import cv2
import os
import json

def precomute_undistort_map(intrinsic):
    cammtx = np.array(intrinsic["original_intrinsics"])
    dist_coef = np.array(intrinsic["dist_params"])
    wh = (intrinsic["width"], intrinsic["height"])
    new_cammtx = intrinsic['intrinsics_undistort']
    
    mapx, mapy = cv2.initUndistortRectifyMap(cammtx, dist_coef, None, new_cammtx, wh, cv2.CV_32FC1)
    return new_cammtx, mapx, mapy

def apply_undistort_map(img, mapx, mapy):
    undistorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return undistorted_img

def undistort_img(img, intrinsic):
    """
    Undistort image using intrinsic parameters.

    Args:
        img (np.ndarray): Image to undistort
        intrinsic (dict): Intrinsic parameters

    Returns:
        undistorted_img (np.ndarray): Undistorted image
    """
    undistorted_img = cv2.undistort(img, np.array(intrinsic["original_intrinsics"]), np.array(intrinsic["dist_params"]), None, np.array(intrinsic["intrinsics_undistort"]))
    return undistorted_img

def undistort_points(pts, intrinsic):
    normalized_undistorted_pts = cv2.undistortPoints(pts, intrinsic["original_intrinsics"], intrinsic["dist_params"])
    new_cammtx = intrinsic["intrinsics_undistort"]
    undistorted_pts = normalized_undistorted_pts.squeeze() * np.array([[new_cammtx[0,0], new_cammtx[1,1]]]) + np.array([[new_cammtx[0,2], new_cammtx[1,2]]])
    return undistorted_pts

# Remap corner pixels to new locations
def remap_corners(corners, cammtx, dist_coef, sn, img=None):
    w,h = 2048, 1536
    new_cammtx, roi = cv2.getOptimalNewCameraMatrix(cammtx, dist_coef, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(cammtx, dist_coef, None, new_cammtx, (w, h), 5) # Last argument is image representation mapping option
    if img is not None:
        warped = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # actual remmaping corners
    normalized_coords = cv2.undistortPoints(corners, cammtx, dist_coef)
    mapped_pixels = normalized_coords.squeeze() * np.array([[new_cammtx[0,0], new_cammtx[1,1]]]) + np.array([[new_cammtx[0,2], new_cammtx[1,2]]])

    # for idx in range(mapped_pixels.shape[0]):
    #     cv2.circle(warped, (int(mapped_pixels[idx,0]), int(mapped_pixels[idx,1])), radius=1, color=(255,0,0), thickness=1)
    # cv2.imwrite(f"/home/capture2/Videos/{sn}_overliad.png", warped)
    return mapped_pixels, new_cammtx
