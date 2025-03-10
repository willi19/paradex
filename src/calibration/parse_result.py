import pycolmap
import numpy as np
import json
import os
import argparse 
import cv2

home_dir = os.path.expanduser("~")
download_dir = os.path.join(home_dir, "download","calibration")
config_dir = "config"

def find_latest_directory():
    dirs = [d for d in os.listdir(download_dir)] 
    if not dirs:
        print("No valid directories found.")
        return
        
        # Sort directories based on name (assuming names are time-based)
    latest_dir = max(dirs, key=str)
    
    return latest_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--name", type=str, help="Name of the directory to detect keypoint.")
    parser.add_argument("--latest", action="store_true", help="Split the latest video files.")

    args = parser.parse_args()

    if not args.latest and not args.name:
        print("Please specify either --latest or --name.")
        exit()
    
    if args.latest:
        name = find_latest_directory()
    else:
        name = args.name

    reconstruction = pycolmap.Reconstruction(os.path.join(download_dir, name, "0", "colmap", "0"))
    cameras, images = dict(), dict()
    for camera_id, camera in reconstruction.cameras.items():
        cameras[camera_id] = camera # distortion params
    for image_id, image in reconstruction.images.items():
        images[image_id] = image # distortion params    

    intrinsics = dict()
    for imid in images:
        serialnum = images[imid].name.split("_")[0][:-4]
        camid  = images[imid].camera_id
        print(f"camid : {camid}, imageid : {imid}")

        w, h = cameras[camid].width, cameras[camid].height
        fx, fy, cx, cy, k1, k2, p1, p2 = cameras[camid].params
        cammtx = np.array([[fx,0.,cx],[0.,fy, cy], [0.,0.,1.]])
        dist_params = np.array([k1,k2,p1,p2])
        new_cammtx, roi = cv2.getOptimalNewCameraMatrix(cammtx, dist_params, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(cammtx, dist_params, None, new_cammtx, (w, h), 5) # Last argument is image representation mapping option
        x,y,w,h = roi

        # Save into parameters
        intrinsics[serialnum]["original_intrinsics"] = list(cammtx.reshape(-1)) # calibrated
        intrinsics[serialnum]["Intrinsics"] = list(new_cammtx.reshape(-1)) # adjusted as pinhole
        intrinsics[serialnum]["Intrinsics_warped"] = list(new_cammtx.reshape(-1))
        intrinsics[serialnum]["Intrinsics_warped"][2] -= x   # check this to get undistorted information
        intrinsics[serialnum]["Intrinsics_warped"][5] -= y
        intrinsics[serialnum]["height"] = h 
        intrinsics[serialnum]["width"] = w
        intrinsics[serialnum]["dist_param"] = list(dist_params)        


    json.dump(intrinsics, open("/home/robot/Desktop/tools/flir_python/intrinsics_adjusted.json", "w"), indent='\t')
