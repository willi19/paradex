import pycolmap
import numpy as np
import os
import cv2

def get_two_view_geometries(cam1, cam2, pix1, pix2, indices, pair): # get tuple of cameras
    pycam1 = pycolmap.Camera(model="OPENCV", width=cam1["width"], height=cam1["height"], params=list(cam1["params"].reshape(-1)))
    pycam2 = pycolmap.Camera(model="OPENCV", width=cam2["width"], height=cam2["height"], params=list(cam2["params"].reshape(-1)))
    E = pycolmap.estimate_essential_matrix(pix1, pix2, pycam1, pycam2)
    
    F = pycolmap.estimate_fundamental_matrix(pix1, pix2)
    H = pycolmap.estimate_homography_matrix(pix1, pix2)
    if E is None or F is None or H is None:
        return None
    # database is shared resource here
    return pair[0], pair[1], indices, F['F'], E['E'], H['H'], 3 # ways to getting two-view-geometries


def load_colmap_camparam(path):
    reconstruction = pycolmap.Reconstruction(os.path.join(path, "0", "colmap"))
    cameras, images = dict(), dict()
    for camera_id, camera in reconstruction.cameras.items():
        cameras[camera_id] = camera # distortion params
    for image_id, image in reconstruction.images.items():
        images[image_id] = image # distortion params    

    intrinsics = dict()
    extrinsics = dict()
    
    for imid in images:
        serialnum = images[imid].name.split("_")[0][:-4]
        camid  = images[imid].camera_id

        w, h = cameras[camid].width, cameras[camid].height
        fx, fy, cx, cy, k1, k2, p1, p2 = cameras[camid].params

        cammtx = np.array([[fx,0.,cx],[0.,fy, cy], [0.,0.,1.]])
        dist_params = np.array([k1,k2,p1,p2])
        new_cammtx, roi = cv2.getOptimalNewCameraMatrix(cammtx, dist_params, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(cammtx, dist_params, None, new_cammtx, (w, h), 5) # Last argument is image representation mapping option
        x,y,w,h = roi

        intrinsics[serialnum] = dict()
        # Save into parameters
        intrinsics[serialnum]["original_intrinsics"] = cammtx # calibrated
        intrinsics[serialnum]["intrinsics_undistort"] = new_cammtx # adjusted as pinhole
        # intrinsics[serialnum]["Intrinsics_warped"] = list(new_cammtx.reshape(-1))
        # intrinsics[serialnum]["Intrinsics_warped"][2] -= x   # check this to get undistorted information
        # intrinsics[serialnum]["Intrinsics_warped"][5] -= y
        intrinsics[serialnum]["height"] = h 
        intrinsics[serialnum]["width"] = w
        intrinsics[serialnum]["dist_params"] = dist_params


        extrinsics[serialnum] = dict()
        
        cam_pose = images[imid].cam_from_world().matrix()
        
        extrinsics[serialnum] = cam_pose.tolist()
    
    return intrinsics, extrinsics