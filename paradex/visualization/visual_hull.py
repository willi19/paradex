import numpy as np

def get_visual_hull(mask_dict, proj_mtx, voxel, thres):
    vote_grid = np.zeros((voxel.shape[1]), dtype=np.float32)
    count_grid = np.zeros((voxel.shape[1]), dtype=np.float32) + 0.01

    for serial_num, mask in mask_dict.items():
        combined_mask = np.any(mask, axis=0)  # (H, W)

        os.makedirs("mask", exist_ok=True)
        cv2.imwrite(f"mask/{serial_num}.png", combined_mask.astype(np.uint8) * 255)
                
        w, h = combined_mask.shape[1], combined_mask.shape[0]

        proj = proj_mtx[serial_num]

        image_points_homo = proj @ voxel

        valid_depth = image_points_homo[2, :] > 0
        image_points = image_points_homo[:2] / image_points_homo[2:3]
        image_points[:, valid_depth] = (image_points_homo[:2, valid_depth] / 
                                       image_points_homo[2:3, valid_depth])

        u_coords = image_points[0, :]
        v_coords = image_points[1, :]

        valid_u = (u_coords >= 0) & (u_coords < w)
        valid_v = (v_coords >= 0) & (v_coords < h)
        valid_projection = valid_depth & valid_u & valid_v

        u_int = np.clip(np.round(u_coords[valid_projection]).astype(int), 0, w - 1)
        v_int = np.clip(np.round(v_coords[valid_projection]).astype(int), 0, h - 1)
        
        mask_values = combined_mask[v_int, u_int]

        vote_grid[valid_projection] += mask_values.astype(np.int32)
        count_grid[valid_projection] += 1
    visual_hull = voxel[:3, ((vote_grid / count_grid) >= thres) & (count_grid > 4)].T
    return visual_hull
