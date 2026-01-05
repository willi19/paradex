import pickle
import os
import trimesh
import numpy as np
import cv2

from paradex.utils.path import shared_dir
from paradex.image.image_dict import ImageDict

# trimesh로 박스 생성 (크기를 지정)
# mesh.export(os.path.join(rsc_path, "object", "book", "book.obj"))
obj_name = "pringles"
root_dir = os.path.join(shared_dir, "obj_register", obj_name)

for index in os.listdir(root_dir):
    mesh = trimesh.load(os.path.join(shared_dir, f"RSS2026_Mingi/object/paradex/{obj_name}/raw_mesh/pringles.obj"))

    img_dict = ImageDict.from_path(os.path.join(root_dir, index))
    if not os.path.exists(os.path.join(root_dir, index, "image")):
        img_dict.undistort()
        img_dict = img_dict.from_path(os.path.join(root_dir, index))
    _, obj_marker_3d = img_dict.triangulate_markers()
    _, template_marker_3d = img_dict.triangulate_markers("4X4_50")
    
    
    merged_obj_marker = []
    for mid in obj_marker_3d:
        merged_obj_marker.append(obj_marker_3d[mid])
    merged_obj_marker = np.concatenate(merged_obj_marker, axis=0)
    marker_obj_2d = img_dict.project_pointcloud(merged_obj_marker)

    merged_template_marker = []
    for mid in template_marker_3d:
        if template_marker_3d[mid] is not None:
            merged_template_marker.append(template_marker_3d[mid])
    
    merged_template_marker = np.concatenate(merged_template_marker, axis=0)
    marker_template_2d = img_dict.project_pointcloud(merged_template_marker)


    debug_img_dict = img_dict.draw_keypoint(marker_obj_2d)
    debug_img_dict = debug_img_dict.draw_keypoint(marker_template_2d, color=(0,0,255))

    # debug_img_dict.save(os.path.join(root_dir, index, "debug"))


    p1, p2, p3, p4 = np.mean(template_marker_3d[0], axis=0), np.mean(template_marker_3d[1], axis=0), np.mean(template_marker_3d[2], axis=0), np.mean(template_marker_3d[3], axis=0)

    x_dir = (p1 + p2) - (p3 + p4)
    y_dir  = (p2 + p4) - (p1 + p3)

    x_dir /= np.linalg.norm(x_dir)
    y_dir /= np.linalg.norm(y_dir)
    z_dir = np.cross(x_dir, y_dir)
    z_dir /= np.linalg.norm(z_dir)

    middle = (p1 + p2 + p3 + p4) / 4
    obj_T = np.eye(4)

    obj_T[:3, 0] = x_dir
    obj_T[:3, 1] = y_dir
    obj_T[:3, 2] = z_dir
    obj_T[:3, 3] = middle - 0.096 * z_dir

    obj_cor = np.linalg.inv(obj_T)
    marker_offset = {}
    for mid in obj_marker_3d:
        print(mid, index)
        marker_offset[mid] = ((obj_cor[:3,:3] @ obj_marker_3d[mid].T) + obj_cor[:3,3:]).T
    
    os.makedirs(os.path.join(shared_dir, f"RSS2026_Mingi/marker_offset/{obj_name}/"), exist_ok=True)
    np.save(os.path.join(shared_dir, f"RSS2026_Mingi/marker_offset/{obj_name}/{index}.npy"), marker_offset)

    mesh.apply_transform(obj_T)
    debug_img_dict = debug_img_dict.project_mesh(mesh)
    debug_img_dict.save(os.path.join(root_dir, index, "debug"))
    # print(marker_3d.keys())


# img_dir = os.path.join(shared_dir, f"object/marker_offset/book/{index}/image")
# img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}
# serial_list = list(img_dict.keys())
# serial_list.sort()

# intrinsic, extrinsic = load_current_camparam()
# cammtx = get_cammtx(intrinsic, extrinsic)

# c2r = load_latest_C2R()
# r2c = np.linalg.inv(c2r)

# floor_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic,'4X4_50')
# obj_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)
# marker_index = [650, 651, 652, 653]
# marker_offset = np.load(f"{shared_dir}/object/marker_offset/book/{0}/marker_offset.npy",allow_pickle=True).item()

# A = []
# for marker_id in marker_index:
#     A.append(marker_offset[marker_id])
# A = np.concatenate(A)

# B = []
# for marker_id in marker_index:
#     B.append(obj_cor_3d[marker_id])
# B = np.concatenate(B)

# obj_T = rigid_transform_3D(A, B)
# mesh.apply_transform(obj_T)

# intrinsic_list = []
# extrinsic_list = []
# for serial_name in serial_list:
#     extmat = extrinsic[serial_name]
#     extrinsic_list.append(extmat)    
#     intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
    
# renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')
# frame, mask = project_mesh_nvdiff(mesh, renderer)
# mask = mask.detach().cpu().numpy()[:,:,:,0].astype(np.bool_)

# os.makedirs(os.path.join(shared_dir, f"object/marker_offset/book/{index}/debug"), exist_ok=True)

# for i, serial_num in enumerate(serial_list):
#     img = undistort_img(img_dict[serial_num], intrinsic[serial_num])
    
#     robot_2d = []
    
#     for id, cor in obj_cor_3d.items():
#         if cor is not None:
#             robot_2d.append(project(cammtx[serial_num], cor))
#     draw_aruco(img, robot_2d, list(obj_cor_3d.keys()))
    
#     robot_2d = []
    
#     for id, cor in floor_cor_3d.items():
#         if cor is not None:
#             robot_2d.append(project(cammtx[serial_num], cor))
#     draw_aruco(img, robot_2d, list(floor_cor_3d.keys()))
    
#     overlay_mask(img, mask[i], np.array((255,0, 0)), 0.7)
    
#     cv2.imwrite(os.path.join(shared_dir, f"object/marker_offset/book/{index}/debug/{serial_num}.png"), img)