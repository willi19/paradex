import json
import pycolmap
import numpy as np
import cv2
import os
from os.path import join as pjoin
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from .viz_utils import pyramid, simpleViewer
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)


with open("./config/charuco_info.json", "r") as f:
    boardinfo = json.load(f)

board_idx_list = [1, 2, 3, 4]
cb_list = [boardinfo[str(board_idx)] for board_idx in board_idx_list]

board_list = [aruco.CharucoBoard( (cb["numX"], cb["numY"]), cb["checkerLength"], cb["markerLength"], aruco_dict, np.array(cb["markerIDs"])) for cb in cb_list]
board_info_list = [(board, int(cb["numMarker"]), bi) for cb, board, bi in zip(cb_list, board_list, board_idx_list)]

corners3d = board_info_list[0][0].getChessboardCorners()
arucoDetector = aruco.ArucoDetector(aruco_dict)
arucoDetector_tuned = aruco.ArucoDetector(aruco_dict)
params = arucoDetector_tuned.getDetectorParameters()
# # set detect parameters
def create_text_3d(text, position, font_size=0.01, color=[1, 1, 1]):
    text_3d = o3d.t.geometry.TriangleMesh.create_text(text).to_legacy()
    text_3d.compute_vertex_normals()
    text_3d.translate(position)
    # text_3d.scale(font_size, position)
    text_3d.paint_uniform_color(color)
    return text_3d

def pyramid(translation, rotation, img=None, focal_length=0.01, color=[1,0,0]): # in order to get list of points. use image size
    #points, res = [[-0.25,0.25,0.5], [-0.25,-0.25,0.5], [0.25, -0.25, 0.5], [0.25,0.25, 0.5], [0,0,0]], []
    # Use sensor size of BFLY-31s4c
    h, w = 5.3/100, 7.07/100 #0.5*H/(H+W), 0.5*W/(W+H) # apply image ratio
    # Opencv : z positive
    # scaled for better viz
    points, res = [[w,-h, 2*focal_length], [w, h, 2*focal_length], [-w,h, 2*focal_length], [-w,-h, 2*focal_length], [0,0,0]], []
    result = []
    for p in points:
        tmp = rotation.T @ (np.array(p) - translation) # world2cam matrix에 대해 cam -> world 전환
        res.append(tmp)
    sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.005)
    sphere.translate(res[4])
    sphere.paint_uniform_color(tuple(color))
    lines = [
    [0, 1],
    [0, 3],
    [1, 2],
    [2, 3],
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
    ]
    colors = [color for i in range(len(lines))]
    # define texture
    vertices = np.array([res[3], res[0], res[1], res[2]])
    indices = np.array([[2, 1, 0], [0, 3, 2], [1,2,3], [3,0,1]])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(res),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)    
    # result.append(line_set)
    # result.append(sphere)

    return line_set #line_set



def triangulate(img_pts, projections):
    """
    N : number of images with same marker
    img_pts : (N, 2) array
    projections : list of 3x4 matrices of length N
    """
    N = img_pts.shape[0]
    assert img_pts.shape == (N, 2)
    assert projections.shape == (N, 3, 4)

    curX = img_pts[:, 0]
    curY = img_pts[:, 1]

    # Prepare the projections
    P0 = projections[:, 0]
    P1 = projections[:, 1]
    P2 = projections[:, 2]

    # Construct the matrix A using broadcasting
    A = []
    for i in range(N):
        A1 = (curY[i] * P2[i]) - P1[i]
        A2 = (curX[i] * P2[i]) - P0[i]
        A.append(A1)
        A.append(A2)
    A = np.array(A)
    # Perform SVD
    U, S, Vt = np.linalg.svd(A)
    kp3d = Vt[-1, 0:3] / Vt[-1, 3]  # Take the last row and normalize
    return kp3d

def detect_charuco_features(root_dir):
    global board_info_list

    ret = {"detected_corners": {}, "detected_markers": {}, "detected_ids": {}, "detected_mids": {}}
    
    frame_path = pjoin(root_dir, "frames")
    scene_list = os.listdir(frame_path)
    scene_list.sort()


    for scene_idx, scene_name in enumerate(scene_list):
        scene_path = pjoin(frame_path, scene_name)
        
        for img_name_tot in os.listdir(scene_path):
            img_name = img_name_tot.split(".")[0]
            img_path = pjoin(scene_path, img_name_tot)
            img = cv2.imread(img_path)
            for board_idx, b in enumerate(board_info_list):
                cur_board, cur_board_id = b[0], b[2]

                charCorner = np.load(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_charCorner.npy"))
                charIDs = np.load(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_charIDs.npy"))
                markerCorner = np.load(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_markerCorner.npy"))
                markerIDs = np.load(pjoin(root_dir, "keypoints", scene_name, f"{img_name}_{cur_board_id}_markerIDs.npy"))

                if img_name not in ret["detected_corners"]:
                    ret["detected_corners"][img_name] = []
                    ret["detected_ids"][img_name] = []
                    ret["detected_markers"][img_name] = []
                    ret["detected_mids"][img_name] = []

                if charIDs.shape != (0,):
                    ret["detected_corners"][img_name].append(charCorner[:,0,:])
                    ret["detected_ids"][img_name].append((charIDs + 70*len(board_info_list)*scene_idx + 70*cur_board_id)[:,0])
                    for val in markerCorner:
                        ret["detected_markers"][img_name].append(val)
                    ret["detected_mids"][img_name].append(markerIDs)


    for img_name in list(ret["detected_corners"].keys()):    
        if len(ret["detected_corners"][img_name]) > 0:
            ret["detected_corners"][img_name] = np.concatenate(ret["detected_corners"][img_name], axis=0)
            ret["detected_ids"][img_name] = np.concatenate(ret["detected_ids"][img_name], axis=0)
            #detected_markers = np.concatenate(detected_markers, axis=0)
            ret["detected_mids"][img_name] = np.concatenate(ret["detected_mids"][img_name], axis=0)

            np.save(pjoin(root_dir, "keypoints", f"{img_name}_detected_corners.npy"), ret["detected_corners"][img_name])
            np.save(pjoin(root_dir, "keypoints", f"{img_name}_detected_ids.npy"), ret["detected_ids"][img_name])
            np.save(pjoin(root_dir, "keypoints", f"{img_name}_detected_mids.npy"), ret["detected_mids"][img_name])
    return ret


if __name__ == "__main__":
    reconstruction = pycolmap.Reconstruction("/home/capture18/captures1/calib_0221_2/out")
    cameras, images, points3D = {}, {}, {}
    for camera_id, camera in reconstruction.cameras.items():
        cameras[camera_id] = camera # distortion params
    for image_id, image in reconstruction.images.items():
        images[image_id] = image # distortion params    
    for point3D_id, point3D in reconstruction.points3D.items():
        points3D[point3D_id] = point3D

    # ret = detect_charuco_features("/home/capture18/captures1/calib_0221_2")
    # print(ret)
    intrinsics = dict() # gather intrinsic info 
    kypt2D = dict()
    for imid in images:
        serialnum = images[imid].name[:-4]
        intrinsics[serialnum] = dict()
        camid  = images[imid].camera_id

        w, h = cameras[camid].width, cameras[camid].height
        fx, fy, cx, cy, k1, k2, p1, p2 = cameras[camid].params
        cammtx = np.array([[fx,0.,cx],[0.,fy, cy], [0.,0.,1.]])
        dist_params = np.array([k1,k2,p1,p2])
        new_cammtx, roi = cv2.getOptimalNewCameraMatrix(cammtx, dist_params, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(cammtx, dist_params, None, new_cammtx, (w, h), 5) # Last argument is image representation mapping option
        x,y,w,h = roi

        # Save into parameters
        intrinsics[serialnum]["original_intrinsics"] = cammtx.copy() # calibrated
        intrinsics[serialnum]["Intrinsics"] = new_cammtx.copy()# list(new_cammtx.reshape(-1)) # adjusted as pinhole
        intrinsics[serialnum]["height"] = h 
        intrinsics[serialnum]["width"] = w
        intrinsics[serialnum]["dist_param"] = list(dist_params) 

        kypt2D_img = np.load(f"/home/capture18/captures1/calib_0221_2/keypoints/{serialnum}_detected_corners.npy")
        kyptIdx = np.load(f"/home/capture18/captures1/calib_0221_2/keypoints/{serialnum}_detected_ids.npy")
        
        for idx, kypt in zip(kyptIdx, kypt2D_img):
            if f"{idx}" not in kypt2D:
                kypt2D[f"{idx}"] = []
            kypt2D[f"{idx}"].append((kypt, camid))
    # Triangulate keypoints
    triangulated_points = {}
    for idx, observations in kypt2D.items():
        if len(observations) < 2:
            continue
        points_2d = np.array([obs[0] for obs in observations])
        cam_ids = [obs[1] for obs in observations]
        proj_matrices = []
        
        for camid in cam_ids:
            image = images[camid]
            Proj = image.cam_from_world.matrix()
            P = intrinsics[image.name[:-4]]["Intrinsics"] @ Proj
            proj_matrices.append(P)
        
        triangulated_points[idx] = triangulate(points_2d, np.array(proj_matrices))
         

    # Compute reprojection errors
    reprojection_errors = {}
    for idx, point_3d in triangulated_points.items():
        for obs in kypt2D[idx]:
            kypt, camid = obs
            image = images[camid]
            P = intrinsics[image.name[:-4]]["Intrinsics"] @ image.cam_from_world.matrix()
            projected = P @ np.hstack((point_3d, 1)).reshape(4, 1)
            projected = (projected[:2] / projected[2]).flatten()
            error = np.linalg.norm(projected - kypt)
            if camid not in reprojection_errors:
                reprojection_errors[camid] = []
            reprojection_errors[camid].append(error)

    # Compute mean reprojection error per camera
    mean_reprojection_errors = {camid: np.mean(errors) for camid, errors in reprojection_errors.items()}

    print("Triangulation and reprojection error computation complete.")
    print("Mean reprojection errors:", mean_reprojection_errors)

    obj_list = []

    for camid, image in images.items():
        cam = cameras[image.camera_id]
        Proj = image.cam_from_world.matrix()
        t = Proj[:3, 3]
        Rot = Proj[:3, :3]

        focal_length = cam.params[0] / 50000

        # text_3d = create_text_3d(f"Camera {camid} : {mean_reprojection_errors[camid]}", t, font_size=0.01)
        # obj_list.append(text_3d)
        err = mean_reprojection_errors[camid]
        obj_list.append(pyramid(t, Rot, focal_length=focal_length, color=np.array([0.9,0.9,0.9])  - np.array([1,1,1]) * err / 14 ))


    for idx, point in triangulated_points.items():
        sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.005)
        sphere.translate(point)
        sphere.paint_uniform_color((1,0,0))
        obj_list.append(sphere)

        if len(obj_list) > 200:
            break
    viz = simpleViewer("render", 2048, 1536, obj_list)
    