from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import cv2
import numpy as np

from paradex.utils.path import home_path, shared_dir
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, merge_charuco_detection
from paradex.image.image_dict import ImageDict
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.image.aruco import get_board_cor, detect_charuco

device = "cuda:0"
sam2_checkpoint = f"{home_path}/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

def load_mask(root_dir, predictor):
    outdir = os.path.join(root_dir, "masks")
    outdir_img = os.path.join(root_dir, "masked_images")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_img, exist_ok=True)

    charuco_path = os.path.join(root_dir, "charuco_3d")
    
    img_dir = os.path.join(root_dir, "images")
    tot_serial_list = os.listdir(img_dir)
    
    intrinsic, extrinsic = load_camparam(os.path.join(root_dir))
    board_cor = get_board_cor()['1']
    
    bids = board_cor['checkerIDs'].flatten()
    bcorners = board_cor['checkerCorner']

    obj_charocu_id = np.ones(70)

    for charuco_file in os.listdir(charuco_path):
        if not charuco_file.endswith("_cor.npy"):
            continue
        charuco_id = np.load(os.path.join(charuco_path, charuco_file.replace("_cor.npy", "_id.npy")))
        obj_charocu_id[charuco_id] = 0

    for charuco_file in os.listdir(charuco_path):
        if not charuco_file.endswith("_cor.npy"):
            continue
        
        charuco_3d = np.load(os.path.join(charuco_path, charuco_file))
        charuco_id = np.load(os.path.join(charuco_path, charuco_file.replace("_cor.npy", "_id.npy")))
        
        frame_idx = charuco_file.split("_")[0]
        serial_list = [s for s in tot_serial_list if os.path.exists(os.path.join(img_dir, s, f"frame_{frame_idx}.jpg"))]
        
        intrinsic_partial = {s: intrinsic[s] for s in serial_list}
        extrinsic_partial = {s: extrinsic[s] for s in serial_list}
        imgs = {serial:cv2.imread(os.path.join(img_dir, serial, f"frame_{frame_idx}.jpg")) for serial in serial_list}
        img_dict = ImageDict(imgs, intrinsic_partial, extrinsic_partial)
        
        ids_comm, bids_comm = find_common_indices(charuco_id, bids)
        obj_points = bcorners[bids_comm]
        img_points = charuco_3d[ids_comm]   
        
        board_pose = SOLVE_XA_B(obj_points, img_points)
        bids_occluded = np.setdiff1d(bids, charuco_id)

        bbox_pts = np.concatenate([bcorners[obj_charocu_id==1], bcorners[obj_charocu_id==1] + np.array([0,0,-0.05])], axis=0)
        bbox_pts_hom = np.hstack([bbox_pts, np.ones((bbox_pts.shape[0], 1))])
        bbox_pts_world = board_pose @ bbox_pts_hom.T
        bbox_pts_world = bbox_pts_world.T[:, :3]  # slightly above the board
        proj_center = img_dict.project_pointcloud(bbox_pts_world)
        
        query_pts = np.mean(bbox_pts, axis=0).reshape(1,3)
        for serial in proj_center:
            img = imgs[serial]
            
            h, w = img.shape[:2]
            bbox = cv2.boundingRect(np.array(proj_center[serial]).astype(int))
            predictor.set_image(img)
            
            masks, scores, logits = predictor.predict(
                point_coords=np.array(proj_center[serial]).reshape(-1, 2),
                point_labels=np.array([1]*len(proj_center[serial])),
                box=np.array([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]),
                multimask_output=False,
            )
            
            mask = (masks[0] * 255).astype("uint8")
            output_path = os.path.join(outdir, serial, f"frame_{frame_idx}.png")
            os.makedirs(os.path.join(outdir, serial), exist_ok=True)
            cv2.imwrite(output_path, mask)
            
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            output_img_path = os.path.join(outdir_img, serial, f"frame_{frame_idx}.jpg")
            os.makedirs(os.path.join(outdir_img, serial), exist_ok=True)
            cv2.imwrite(output_img_path, masked_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            debug_img = img.copy()
            
            # plot bbox
            debug_img = cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)

            cv2.imshow("mask", cv2.resize(masked_img, (w//2, h//2)))
            cv2.waitKey(0)
        
root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
for obj_name in os.listdir(root_dir)[:1]:
    obj_path = os.path.join(root_dir, obj_name)
    for index in os.listdir(obj_path):
        demo_path = os.path.join("capture/object_turntable", obj_name, index)
        load_mask(os.path.join(home_path, "paradex_download", demo_path), predictor)
        

# root_dir = os.path.join(shared_dir, "capture/object_turntable")
# obj_list = ['pepper_tuna']

# for obj_name in obj_list:
#     index_list = os.listdir(os.path.join(root_dir, obj_name))
#     for index in index_list:
#         demo_path = os.path.join("capture/object_turntable", obj_name, index)
#         image_dir = os.path.join(home_path, "paradex_download", demo_path, "images")
#         if not os.path.exists(image_dir):
#             continue
#         output_dir = os.path.join(home_path, "paradex_download", demo_path, "masks")
#         os.makedirs(output_dir, exist_ok=True)
        
#         serial_list = os.listdir(image_dir)
#         for serial in serial_list:
#             serial_image_dir = os.path.join(image_dir, serial)
#             serial_output_dir = os.path.join(output_dir, serial)
#             os.makedirs(serial_output_dir, exist_ok=True)
            
#             for img_name in os.listdir(serial_image_dir):
#                 img_path =  os.path.join(serial_image_dir, img_name)
#                 img = cv2.imread(img_path)
#                 img = predictor.set_image(img)
                
#                 masks, scores, logits = predictor.predict(
#                     img,
#                     multimask_output=False,
#                     box=None,
#                     point_coords=None,
#                     point_labels=None,
#                     mask_threshold=0.0,
#                     iou_threshold=0.0,
#                     stability_score_threshold=0.0,
#                 )
                
#                 mask = (masks[0].cpu().numpy() * 255).astype("uint8")
#                 output_path = os.path.join(serial_output_dir, img_name)
#                 predictor.save_mask(output_path, mask)