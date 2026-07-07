import numpy as np
import cv2
import os
import json

def precomute_undistort_map(intrinsic):
    """Precompute a CPU ``cv2.remap`` undistortion map for one camera.

    Parameters
    ----------
    intrinsic : dict
        Per-camera params with ``original_intrinsics``, ``dist_params``,
        ``intrinsics_undistort``, ``width``, ``height``.

    Returns
    -------
    tuple
        ``(new_cammtx, mapx, mapy)`` — the undistorted intrinsics and the ``CV_16SC2``
        remap tables for :func:`apply_undistort_map`.
    """
    cammtx = np.array(intrinsic["original_intrinsics"])
    dist_coef = np.array(intrinsic["dist_params"])
    wh = (intrinsic["width"], intrinsic["height"])
    new_cammtx = intrinsic['intrinsics_undistort']
    mapx, mapy = cv2.initUndistortRectifyMap(cammtx, dist_coef, None, new_cammtx, wh, cv2.CV_16SC2)
    return new_cammtx, mapx, mapy

def apply_undistort_map(img, mapx, mapy):
    """Undistort an image with precomputed remap tables (``cv2.remap``).

    Parameters
    ----------
    img : numpy.ndarray
        Distorted input frame.
    mapx, mapy : numpy.ndarray
        Remap tables from :func:`precomute_undistort_map`.

    Returns
    -------
    numpy.ndarray
        The undistorted image.
    """
    undistorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return undistorted_img


def precompute_undistort_map_torch(intrinsic, device='cuda'):
    """GPU undistort grid for torch.nn.functional.grid_sample.

    Returns (new_cammtx, grid_tensor of shape (1, H, W, 2) on `device`).
    """
    import torch
    cammtx = np.array(intrinsic["original_intrinsics"])
    dist_coef = np.array(intrinsic["dist_params"])
    w, h = intrinsic["width"], intrinsic["height"]
    new_cammtx = intrinsic['intrinsics_undistort']
    mapx, mapy = cv2.initUndistortRectifyMap(
        cammtx, dist_coef, None, new_cammtx, (w, h), cv2.CV_32FC1)
    grid_x = (mapx / (w - 1)) * 2 - 1
    grid_y = (mapy / (h - 1)) * 2 - 1
    grid = np.stack([grid_x, grid_y], axis=-1)[None]
    grid_tensor = torch.from_numpy(grid).to(device=device, dtype=torch.float32)
    return new_cammtx, grid_tensor


def apply_undistort_torch(img_bgr, grid_tensor):
    """Undistort a uint8 HxWx3 BGR frame via grid_sample on the grid's device."""
    import torch
    device = grid_tensor.device
    t = (torch.from_numpy(img_bgr)
         .to(device=device)
         .permute(2, 0, 1)
         .unsqueeze(0)
         .float())
    out = torch.nn.functional.grid_sample(
        t, grid_tensor, mode='bilinear', padding_mode='zeros', align_corners=True)
    return (out.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy())

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
    """Map distorted pixel coordinates to undistorted pixel coordinates.

    Runs ``cv2.undistortPoints`` then re-applies ``intrinsics_undistort`` so the
    result is in undistorted-image pixels (not normalized coordinates).

    Parameters
    ----------
    pts : numpy.ndarray
        Distorted points, ``(N, 1, 2)`` or ``(N, 2)``.
    intrinsic : dict
        Per-camera params (``original_intrinsics``, ``dist_params``,
        ``intrinsics_undistort``).

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` undistorted pixel coordinates.
    """
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
