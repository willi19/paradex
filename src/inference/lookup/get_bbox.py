import numpy as np
import cv2
import json
import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

from paradex.utils.file_io import shared_dir, load_current_camparam, load_latest_C2R
from paradex.model.yolo_world_module import YOLO_MODULE
from paradex.image.undistort import undistort_img
from paradex.image.merge import merge_image
from paradex.image.projection import get_cammtx, project_point

from paradex.geometry.polygon import get_minimum_rotated_rectangle_from_mask
from paradex.visualization.visual_hull import plot_3d_points, plot_multiple_3d_points
from paradex.visualization.shape import draw_mask, draw_rectangle

import numpy as np
from scipy.spatial import ConvexHull
import cv2

def visualize_all_masks_and_rectangles(image, masks, rectangle_data):
    """
    모든 마스크와 최소 회전 사각형을 함께 시각화하는 함수
    
    Args:
        image: 원본 이미지
        masks: 모든 마스크들 (H, W) 형태의 배열들
        rectangle_data: 각 마스크별 회전 사각형 정보 리스트
        save_path: 저장 경로 (선택사항)
    
    Returns:
        vis_img: 시각화된 이미지
    """
    overlay = image.copy()
    
    colors = [
        (255, 0, 0),      # 순수 빨간색
        (0, 255, 0),      # 순수 초록색  
        (0, 0, 255),      # 순수 파란색
        (255, 255, 0),    # 순수 노란색 (cyan -> yellow)
        (255, 0, 255),    # 순수 마젠타
        (0, 255, 255),    # 순수 시안
        (255, 128, 0),    # 밝은 주황색
        (128, 0, 255),    # 밝은 보라색
        (0, 255, 128),    # 밝은 청록색
        (255, 0, 128),    # 밝은 분홍색
        (128, 255, 0),    # 라임색
        (0, 128, 255),    # 밝은 하늘색
    ]
    
    for idx, (mask, rect_data) in enumerate(zip(masks, rectangle_data)):
        if rect_data is None:
            continue
            
        color = colors[idx % len(colors)]
        
        if mask is not None:
            overlay = draw_mask(overlay, mask, color)
            
        rect_corners = np.array(rect_data['corners'], dtype=np.int32)
        overlay = draw_rectangle(overlay, rect_corners, color, 4)  # thickness 3->4
        
        center = rect_data['center']
        cv2.circle(overlay, (int(center[0]), int(center[1])), 18, color, -1)  # 15->18
        cv2.circle(overlay, (int(center[0]), int(center[1])), 18, (255, 255, 255), 3)  # 2->3
        cv2.putText(overlay, str(idx), 
                   (int(center[0]) - 8, int(center[1]) + 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)  # 0.6->0.8, 2->3
        
    return overlay

def undistort_image_dir(root_path, intrinsic):
    if not os.path.exists(os.path.join(root_path, "undistorted_images")):
        img_list = os.listdir(os.path.join(root_path, "image"))
        os.makedirs(os.path.join(root_path, "undistorted_images"), exist_ok=True)
        for img_name in img_list:
            serial_num = img_name.split('.')[0]
            img = cv2.imread(os.path.join(root_path, "image", f"{serial_num}.png"))
            undistorted_img = undistort_img(img, intrinsic[serial_num])
            cv2.imwrite(os.path.join(root_path, "undistorted_images", f"{serial_num}.png"), undistorted_img)

def detect_object():
    if not os.path.exists(os.path.join(root_path, "detections.npy")):    
        detection_results = {}
        rectangle_info = {}

        for img_name in tqdm.tqdm(img_list):
            serial_num = img_name.split('.')[0]
            rgb_img = cv2.imread(os.path.join(root_path, "undistorted_images", img_name))
            detections = yolo_module.process_img(rgb_img, with_segmentation=True, top_1=False)
            
            if detections.xyxy.size > 0:
                detections.bbox_center = detections.xyxy[:, :2] + (detections.xyxy[:, 2:] - detections.xyxy[:, :2]) / 2
                
                if hasattr(detections, 'mask') and detections.mask is not None:
                    num_detections = detections.mask.shape[0]
                    print(f"Camera {serial_num}: Found {num_detections} detections")
                    
                    rectangle_info[serial_num] = []
                    
                    for detection_idx in range(num_detections):
                        mask = detections.mask[detection_idx]  # shape: (H, W)
                        
                        rect_corners, rect_info, axis_aligned_bbox, area_reduction = get_minimum_rotated_rectangle_from_mask(mask)
                        
                        if rect_corners is not None:
                            center, (width, height), angle = rect_info
                            rect_data = {
                                'detection_idx': detection_idx,
                                'corners': rect_corners.tolist(),
                                'center': center,
                                'size': (width, height),
                                'angle': angle,
                                'area_reduction': area_reduction,
                                'original_bbox': detections.xyxy[detection_idx].tolist() if detection_idx < len(detections.xyxy) else None
                            }
                            rectangle_info[serial_num].append(rect_data)
                        else:
                            print(f"  Detection {detection_idx}: No valid mask found")
                            rectangle_info[serial_num].append(None)
                    
                    if not any(rectangle_info[serial_num]):
                        del rectangle_info[serial_num]
            
            detection_results[serial_num] = detections
            import time

            if serial_num in rectangle_info and hasattr(detections, 'mask') and detections.mask is not None:
                valid_masks = []
                valid_rect_data = []
                start_time = time.time()    
                for detection_idx, rect_data in enumerate(rectangle_info[serial_num]):
                    if rect_data is not None:
                        valid_masks.append(detections.mask[detection_idx])
                        valid_rect_data.append(rect_data)
                print(f"Processing time for {serial_num}: {time.time() - start_time:.2f} seconds")

                if valid_masks:
                    # 모든 detection을 한 이미지에 표시
                    combined_vis = visualize_all_masks_and_rectangles(
                        rgb_img, 
                        valid_masks, 
                        valid_rect_data
                    )
                    print(f"Visualizing {len(valid_masks)} detections for {serial_num}", time.time() - start_time)
                    # 한 곳에만 저장
                    os.makedirs(os.path.join("mask_visualization", str(index)), exist_ok=True)
                    save_path = os.path.join("mask_visualization", str(index), f"{serial_num}.png")
                    cv2.imwrite(save_path, combined_vis)
                    print(f"Saved visualization for {serial_num} to {save_path}", time.time() - start_time)
                
        np.save(os.path.join(root_path, "detections.npy"), detection_results)
    
    else:
        detection_results = np.load(os.path.join(root_path, "detections.npy"), allow_pickle=True).item()

    return detection_results

def get_visual_hull(mask_dict, proj_mtx, voxel, thres):
    # thres = minimum length of cube to be seen

    vote_grid = np.zeros((voxel.shape[1]), dtype=np.float32)
    count_grid = np.zeros((voxel.shape[1]), dtype=np.float32) + 0.01
    for serial_num, mask in mask_dict.items():
        combined_mask = np.any(mask, axis=0)  # (H, W)
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
    
    selected_index = (((vote_grid / count_grid) > thres) & (count_grid > 16))
    visual_hull = voxel[:3, selected_index].T
    return visual_hull, vote_grid[selected_index]

def refine_visual_hull(mask_dict, proj_mtx, voxel, thres, thres2):
    voxel_h = np.concatenate([voxel, np.ones((voxel.shape[0], 1))], axis=1).T
    cube = np.array([[1,0,0, 0],[-1,0,0,0],[0,1,0,0],[0,-1,0,0],[0,0,1,0],[0,0,-1,0]]) * (thres / 2)
    v_cube = [c.reshape(4, 1) + voxel_h for c in cube]

    vote_grid = np.zeros((voxel.shape[0]), dtype=np.float32)
    count_grid = np.zeros((voxel.shape[0]), dtype=np.float32) + 0.01

    mask_mapping = -np.ones((len(mask_dict), voxel.shape[0]), dtype=np.int32)

    serial_list = list(mask_dict.keys())
    serial_list.sort()

    for ind, serial_num in enumerate(serial_list):
        mask = mask_dict[serial_num]
        w, h = mask.shape[2], mask.shape[1]
        proj = proj_mtx[serial_num]

        image_points_homo = [proj @ vc for vc in v_cube]
        image_points_homo = np.array(image_points_homo)

        valid_depth = np.any(image_points_homo[:, 2, :], axis=0)
        image_points = image_points_homo[:, :2, :] / image_points_homo[:, 2:, :]

        u_coords = image_points[:, 0, :]
        v_coords = image_points[:, 1, :]

        valid_u = (u_coords >= 0) & (u_coords < w)
        valid_v = (v_coords >= 0) & (v_coords < h)
        
        valid_projection = np.all(valid_depth & valid_u & valid_v, axis=0)
        u_int = np.clip(np.round(u_coords[:, valid_projection]).astype(int), 0, w - 1)
        v_int = np.clip(np.round(v_coords[:, valid_projection]).astype(int), 0, h - 1)

        mask_values = mask[:, v_int, u_int]
        mask_ind = np.all(mask_values, axis=1)
        true_indices, index = np.where(mask_ind)

        true_indices_orig = -np.ones((np.sum(valid_projection)))
        true_indices_orig[index] = true_indices
        mask_mapping[ind, valid_projection] = true_indices_orig

        in_mask = np.any(mask_ind, axis=0)
        vote_grid[valid_projection] += in_mask.astype(np.int32)
        count_grid[valid_projection] += 1

    selected_index = (((vote_grid / count_grid) > thres) & (count_grid > 16))
    visual_hull = voxel[selected_index]
    mask_mapping = mask_mapping[:, selected_index]
    return visual_hull, vote_grid[selected_index], mask_mapping

def visualize_by_vote_count(voxel, vote_grid, count_grid):
    """
    Vote count별로 점들을 분류해서 시각화
    """
    point_groups = []
    labels = []
    
    # Vote count별 분류 (0~4+ votes)
    for vote_count in range(0, 24):
        mask = vote_grid == vote_count# & (count_grid > 4)
        label = f"{vote_count} votes"
        
        if np.sum(mask) > 0:
            points = voxel[:3, mask].T
            point_groups.append(points)
            labels.append(label)
            print(f"{label}: {len(points)} points")
    
    return point_groups, labels

from collections import defaultdict
def group_mask_patterns(mask_mapping):
    """
    Group points by their mask detection patterns across cameras.
    
    Args:
        mask_mapping: numpy array of shape (23, 3808)
                     [camera_idx, point_idx] = mask_id or -1
    
    Returns:
        pattern_groups: dict {pattern_tuple: [point_indices]}
    """
    pattern_groups = defaultdict(list)
    
    for point_idx in range(mask_mapping.shape[1]):
        pattern = tuple(mask_mapping[:, point_idx])  # 모든 카메라에서 이 point의 mask 할당
        pattern_groups[pattern].append(point_idx)
    
    return dict(pattern_groups)

def find_connected_components(edge, serial_list, mask_mapping):
    """
    Find connected components from edge list.
    """
    # 모든 노드 수집
    nodes = set()
    for (i, m1), (j, m2) in edge:
        nodes.add((i, m1))
        nodes.add((j, m2))
    
    # 인접 리스트 생성
    graph = {node: [] for node in nodes}
    for (i, m1), (j, m2) in edge:
        graph[(i, m1)].append((j, m2))
        graph[(j, m2)].append((i, m1))
    
    # DFS로 연결 요소 찾기
    visited = set()
    connected_components = []
    
    def dfs(node, component):
        if node in visited:
            return
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, component)
    
    for node in nodes:
        if node not in visited:
            component = []
            dfs(node, component)
            connected_components.append(component)
    
    return connected_components


# 메인 코드
C2R = load_latest_C2R()
obj_name = "pringles_case"
yolo_module = YOLO_MODULE(categories=obj_name)

intrinsic, extrinsic = load_current_camparam("20250812_162600")
proj_mtx = get_cammtx(intrinsic, extrinsic)

x = np.linspace(0, 1.1, 110)
y = np.linspace(-1.1, 1.1, 220)
z = np.linspace(-0.06, 0.4, 40)

X, Y, Z = np.meshgrid(x, y, z)
# robot_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
robot_points_homo = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), np.ones(X.ravel().shape)])
world_points_homo = (C2R @ robot_points_homo)

for index in os.listdir(f"{shared_dir}/multi_pringles")[:1]:
    root_path = os.path.join(f"{shared_dir}/multi_pringles/{index}")
    
    undistort_image_dir(root_path, intrinsic)
    img_list = os.listdir(os.path.join(root_path, "undistorted_images"))
    
    device = torch.device("cuda:0")
    detection_results = detect_object()
    
    mask_dict = {}
    for serial_num, detections in detection_results.items():
        if hasattr(detections, 'mask') and detections.mask is not None:
            mask_dict[serial_num] = detections.mask
    
    # plot_3d_points(world_points_homo, subsample=1000)
    camera_points = []
    for serial_num, ext in extrinsic.items():
        camera_center = - np.linalg.inv(ext[:3, :3]) @ ext[:3, 3]
        camera_points.append(camera_center)
    camera_points = np.array(camera_points)

    # plot_3d_points(camera_points)

    
    visual_hull, vote_grid = get_visual_hull(mask_dict, proj_mtx, world_points_homo, thres=0.8)
    # import pdb; pdb.set_trace()
    # point_group, labels = visualize_by_vote_count(visual_hull.T, vote_grid, count_grid)
    # point_group.append(camera_points)
    # labels.append("camera")
    # plot_multiple_3d_points(point_group, labels)
    # visual_hull_h = np.concatenate([visual_hull, np.ones((visual_hull.shape[0], 1))], axis=1)
    # visual_hull_robot = (np.linalg.inv(C2R) @ visual_hull_h.T ).T[:, :3]
    # plot_3d_points(visual_hull_robot)

    img_dict = {}
    for serial_num, proj in proj_mtx.items():
        img = cv2.imread(os.path.join("mask_visualization", str(index), f"{serial_num}.png"))# np.zeros((1536, 2048, 3), dtype=np.uint8)
        # proj_pts = project_point(visual_hull.copy(), proj, img, color=(0,255,0))
        img_dict[serial_num] = img.copy()


    # plot_3d_points(visual_hull)
    color_list = [(128, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
    
    refined_visual_hull, vote_grid, mask_mapping = refine_visual_hull(mask_dict, proj_mtx, visual_hull, thres=0.01, thres2=0.75)
   #  plot_3d_points(refined_visual_hull)
    # groups = group_mask_patterns(mask_mapping)
    
    serial_list = list(proj_mtx.keys())
    serial_list.sort()
    for cind, serial_num in enumerate(serial_list):
        proj = proj_mtx[serial_num]    
        img = img_dict[serial_num]# np.zeros((1536, 2048, 3), dtype=np.uint8)
        # for mi in range(-1, int(np.max(mask_mapping[cind])+1)):
        #     mi_visual_hull = refined_visual_hull[mask_mapping[cind]==mi]
        #     img = project_point(mi_visual_hull, proj, img, color=color_list[mi+1])
        for vc in range(12, 18):
            img = project_point(refined_visual_hull[vote_grid==vc], proj, img, color=color_list[vc-12])
        
        os.makedirs(f"project/{index}", exist_ok=True)
        cv2.imwrite(f'project/{index}/{serial_num}.png', img)
        img_dict[serial_num] = img.copy()
    cv2.imwrite(f"project/{index}/merged.png", merge_image(img_dict))

    mask_group = []
    mask_idx = {}
    edge = []

    for i, sn in enumerate(serial_list):
        for ind in range(np.max(mask_mapping[i])):
            mask_idx[(sn, ind)] = -1

    for i in range(len(serial_list)):
        for j in range(i+1, len(serial_list)):
            for m1 in range(np.max(np.max(mask_mapping[i]))):
                c1 = (mask_mapping[i]==m1)
                for m2 in range(np.max(np.max(mask_mapping[j]))):
                    c2 = (mask_mapping[j]==m2)

                    num1 = np.sum(c1)
                    num2 = np.sum(c2)

                    num = np.sum(c1 & c2)
                    edge.append((((i, m1), (j, m2)), num / max(num1, num2)))

                    
    # 사용법
    connected_components = find_connected_components(edge, serial_list, mask_mapping)

    print(f"Found {len(connected_components)} connected components:")
    for i, component in enumerate(connected_components):
        os.makedirs(f"detection/{index}",exist_ok=True)
        print(f"Component {i+1}: {len(component)} masks")
        tmp_img_dict = {}
        for cam_idx, mask_id in component:
            serial_num = serial_list[cam_idx]
            point_count = np.sum(mask_mapping[cam_idx] == mask_id)
            print(f"  Camera {serial_num} mask_{mask_id}: {point_count} points")

            img = cv2.imread(os.path.join("mask_visualization", str(index), f"{serial_num}.png"))
            tmp_img_dict[f"{serial_list[cam_idx]}_{mask_id}"] = project_point(refined_visual_hull[mask_mapping[cam_idx]==mask_id], proj_mtx[serial_num], img)
        
        print(i)
        cv2.imwrite(f"detection/{index}/{i}.png", merge_image(tmp_img_dict))