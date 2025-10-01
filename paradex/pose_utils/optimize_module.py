import os
import time
import numpy as np
import cv2
from numpy import concatenate as ncat
import json
import pickle
from collections import deque
from pathlib import Path
import copy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
PROJECT_DIR = Path(__file__).absolute().parent.parent
print(f'PROJECT_DIR {PROJECT_DIR}')
sys.path.insert(0, str(PROJECT_DIR))

from paradex.pose_utils.io import makevideo, get_binary_mask
from paradex.pose_utils.vis_utils import make_grid_image, make_grid_image_np, putText, crop_and_resize_by_mask, pick_corners
from paradex.pose_utils.geometry import rotation_6d_to_matrix, matrix_to_rotation_6d, rigid_transform_3D, triangulate_ransac_batched
from paradex.pose_utils.scene import Scene
from paradex.model.yolo_world_module import check_mask
from paradex.object_detection.object_optim_config import hide_list
 

def silhouette_iou_loss(pred_silhouette, target_silhouette):
    intersection = (pred_silhouette * target_silhouette).sum()
    union = (pred_silhouette + target_silhouette).clamp(0, 1).sum()
    return 1 - (intersection / union)

def silhouette_tg_loss(pred_silhouette, target_silhouette):
    intersection = (pred_silhouette * target_silhouette).sum()
    return 1 - (intersection / target_silhouette.sum())


def get_transformed_obj(obj_dict, optim_scale, optim_R, optim_t, R_type='6d'):
    '''
        Dimension check
    '''
    if R_type == '6d':
        transformed_verts = torch.einsum('mn, bjn -> bjm',optim_scale*rotation_6d_to_matrix(optim_R),(obj_dict['verts']))+optim_t[:3]
    else:
        transformed_verts = torch.einsum('mn, bjn -> bjm',optim_scale*optim_R,(obj_dict['verts']))+optim_t

    transformed_obj = copy.deepcopy(obj_dict)
    transformed_obj['verts'] = transformed_verts

    return transformed_obj


def optimize_with_initial_sRt(input_data):

    start_time = time.time()
    model_inputs, rotation_id, scaled, default_scale, rotation_candidate, initial_translate, learning_args, use_rgb, object_output_dir = input_data
    scene, obj_dict, batch_num, obj_mask_stackted_t, obj_masked_rgb_stacked_t, gt_combined_mask, object_final_output_dir = model_inputs
    learning_rate, iter_numb, early_stopping_max, early_stopping_unit, iou_weight, loss_weight, average_iou_loss_filter, initial_useiou_flag = learning_args

    device = rotation_candidate.device

    optim_R = matrix_to_rotation_6d(rotation_candidate.clone().detach()).float().to(device).requires_grad_(True)
    optim_t = torch.tensor(initial_translate).float().squeeze().to(device).requires_grad_(True)
    optim_scale = torch.tensor([default_scale]).squeeze().to(device).requires_grad_(not scaled)

    if scaled:
        optimizer = optim.Adam([optim_R, optim_t], lr=learning_rate)  # scale later
    else:
        optimizer = optim.Adam([optim_R, optim_t, optim_scale], lr=learning_rate)  # scale later

    debug = True
    if debug: # visualize first
        transformed_obj_dict = get_transformed_obj(obj_dict, optim_scale, optim_R, optim_t)
        # transformed_verts = torch.einsum('mn, bjn -> bjm',optim_scale*pytransform.rotation_6d_to_matrix(optim_R),(p3d_verts))+optim_t
        # transformed_mesh = Meshes(verts=transformed_verts, faces=p3d_faces, textures=p3d_textures)

        rendered_rgb_batched, rendered_silhouette_batched = scene.batch_render(transformed_obj_dict)
        grid_rgb_img = (make_grid_image(rendered_rgb_batched, 1, batch_num).detach().cpu().numpy()*255).astype(np.uint8)
        cv2.imwrite('debug_batch_rgb.png', grid_rgb_img)
        grid_sil_img = (make_grid_image(rendered_silhouette_batched, 1, batch_num).detach().cpu().numpy()*255).astype(np.uint8)
        cv2.imwrite('debug_batch_sil.png', grid_sil_img)

    # prepare final loss for saving minimum loss parameter
    fnl_R = None
    fnl_t = None
    fnl_scale = None
    
    min_loss = None
    early_stopping_count = 0

    img_paths = [] # for make video
    rgb_img_paths = []

    criterion =  nn.MSELoss(reduction='none')
    # Initially filter human mask

    if use_rgb:
        gt_rgb_grid_image = make_grid_image(obj_masked_rgb_stacked_t, 1, batch_num).detach().cpu().numpy()*255

    for nit in range(iter_numb):
        optimizer.zero_grad()

        loss = 0.
        # Set diff value for logging
        sil_mse_diff, sil_iou_diff, rgb_mse_diff = 0., 0., 0.

        transformed_obj_dict = get_transformed_obj(obj_dict, optim_scale, optim_R, optim_t)
        rendered_rgb_batched, rendered_silhouette_batched = scene.batch_render(transformed_obj_dict, render_rgb = use_rgb)

        sil_mse = criterion(obj_mask_stackted_t, rendered_silhouette_batched)
        sil_mse_loss = torch.mean(sil_mse)
        loss+=sil_mse_loss
        sil_mse_diff =  sil_mse_loss.item()
        
        if initial_useiou_flag:
            sil_iou_loss = silhouette_iou_loss(obj_mask_stackted_t, rendered_silhouette_batched) # F.mse_loss(rendered_silhouette, obj_mask_t_dict[cam_id], reduction='mean')
            loss+=sil_iou_loss*iou_weight
            sil_iou_diff=sil_iou_loss.item()
        if use_rgb:
            rgb_mse = criterion(obj_masked_rgb_stacked_t,  rendered_rgb_batched)
            rgb_mse_each = torch.mean(rgb_mse, axis=(1,2,3))
            rgb_mse_loss = torch.mean(rgb_mse)
            loss+=rgb_mse_loss
            rgb_mse_diff = rgb_mse_loss.item()

        if initial_useiou_flag:
            if sil_iou_diff >= 0.98:
                print("early stopping")
                return None, None, None
            if sil_iou_diff<=average_iou_loss_filter:
                initial_useiou_flag = False
                print("IoU filter turned off")
    
        if nit%10==0:
            print(f'{rotation_id}: {nit}: sil_mse_diff: {sil_mse_diff}, sil_iou_diff: {sil_iou_diff}, rgb_mse_diff: {rgb_mse_diff}, scale: {optim_scale.item()}')

        if nit % 4 == 0:
            # save to image.
            if use_rgb:
                rendered_rgb = (make_grid_image(rendered_rgb_batched, 1, batch_num).detach().cpu().numpy()*255).astype(np.uint8)
                cv2.imwrite(str(object_output_dir/('rotation_%s_rgb_%05d.png'%(rotation_id, nit))), \
                        np.vstack([gt_rgb_grid_image, rendered_rgb]))
                rgb_img_paths.append(str(object_output_dir/('rotation_%s_rgb_%05d.png'%(rotation_id, nit))))
            renderd_combined_mask = make_grid_image(rendered_silhouette_batched, 1, batch_num).detach().cpu().numpy()
            commbined_sil = np.zeros((renderd_combined_mask.shape[0], renderd_combined_mask.shape[1],3))
            commbined_sil[gt_combined_mask[...,0]>0] += [1,0,0]
            commbined_sil[renderd_combined_mask[...,0]>0] += [0,1,0]
            cv2.imwrite(str(object_output_dir/('rotation_%s_sil_%05d.png'%(rotation_id, nit))),commbined_sil*255)
            img_paths.append(str(object_output_dir/('rotation_%s_sil_%05d.png'%(rotation_id, nit))))

        loss*=loss_weight
        loss.backward()
        optimizer.step()

        if min_loss is None or min_loss-loss.item()>early_stopping_unit:
            early_stopping_count=0
        else:
            early_stopping_count+=1
            if early_stopping_count>early_stopping_max:
                print('Early Stopping...')
                break

        if min_loss is None or loss.item()<min_loss:
            min_loss = loss.item()
            fnl_R = optim_R.clone().detach()
            fnl_t = optim_t.clone().detach()
            fnl_scale = optim_scale.clone().detach()
            if nit%10 == 0: # save pickle file.
                fnl_R_rotmat = rotation_6d_to_matrix(fnl_R)
                result_dict = {'R':fnl_R_rotmat, 't':fnl_t, 'scale':fnl_scale}
                pickle.dump(result_dict, open(object_final_output_dir/f'obj_output_after_optim_total_{rotation_id}.pickle','wb'))

    # make video
    output_video_path = str(object_output_dir/('rotation_%s.mp4'%(rotation_id)))
    makevideo(img_paths, output_video_path)

    if use_rgb:
        output_video_path = str(object_output_dir/('rotation_rgb_%s.mp4'%(rotation_id)))
        makevideo(rgb_img_paths, output_video_path)

    fnl_R_rotmat = rotation_6d_to_matrix(fnl_R)
    result_dict = {'R':fnl_R_rotmat, 't':fnl_t, 'scale':fnl_scale}
    pickle.dump(result_dict, open(object_final_output_dir/f'obj_output_after_optim_total_{rotation_id}.pickle','wb'))
    
    # NOTE: final visualization.
    transformed_obj_dict = get_transformed_obj(obj_dict, fnl_scale, fnl_R, fnl_t)
    rendered_rgb_batched, rendered_silhouette_batched = scene.batch_render(transformed_obj_dict) # should always render RGB

    sil_mse = nn.functional.mse_loss(obj_mask_stackted_t, rendered_silhouette_batched).item()
    rgb_mse = nn.functional.mse_loss(obj_masked_rgb_stacked_t, rendered_rgb_batched).item()

    # Final Visualization
    cv2.imwrite(str(object_final_output_dir/('rotation_%s_rgb_final.png'%(rotation_id))), np.vstack([make_grid_image(obj_masked_rgb_stacked_t, 1, batch_num).detach().cpu().numpy()*255,
                                                                                                        make_grid_image(rendered_rgb_batched, 1, batch_num).detach().cpu().numpy()*255]))
    renderd_combined_mask = make_grid_image(rendered_silhouette_batched, 1, batch_num).detach().cpu().numpy()
    commbined_sil = np.zeros((renderd_combined_mask.shape[0], renderd_combined_mask.shape[1],3))
    commbined_sil[gt_combined_mask[...,0]>0] += [1,0,0]
    commbined_sil[renderd_combined_mask[...,0]>0] += [0,1,0]
    cv2.imwrite(str(object_final_output_dir/('rotation_%s_sil_final.png'%(rotation_id))),commbined_sil*255)
    print(f'{rotation_id}: rgb_loss {rgb_mse} sil_loss {sil_mse}')

    json.dump({'rgb_loss':rgb_mse, 'sil_loss':sil_mse}, open(object_final_output_dir/f'obj_final_loss_{rotation_id}.json','w'))

    end_time = time.time()
    print(f'Optimizatino for one candidate ended in {end_time-start_time}')
    return result_dict, rgb_mse, sil_mse



def get_initial_T_by_picking(scene, tg_cam_ids, o3d_mesh):
    # No need for mask 
    import open3d as o3d
    colors = [
    (0, 0, 255),    # 빨강
    (0, 165, 255),  # 주황
    (0, 255, 255),  # 노랑
    (0, 255, 0),    # 초록
    (255, 0, 0),    # 파랑
    (128, 0, 0),    # 남색
    (128, 0, 128)   # 보라
    ]

    num_corners = 4
    gathered, projections = dict(), dict()
    for cam_id in tg_cam_ids:
        img_np = scene.get_image(cam_id, 0)
        proj = scene.proj_matrix[cam_id]
        det_keypoints, labels = pick_corners(img_np, num_corners)

        for idx in det_keypoints:
            if idx in gathered:
                gathered[idx].append(det_keypoints[idx])
                projections[idx].append(proj)
            else:
                gathered[idx] = [det_keypoints[idx]]
                projections[idx] = [proj]
        
    # triangulate points
    kp3d = dict()
    for idx in gathered:
        pt = np.vstack(gathered[idx])
        tmp_kp3d, tmp_valid = triangulate_ransac_batched({0:pt}, projections[idx], 0.9, 100, 3)
        if tmp_valid[0]:
            kp3d[str(idx)] = tmp_kp3d[0]

    # For Debug: Reproject image
    for cam_id in tg_cam_ids:
        canvas = scene.get_image(cam_id, 0)
        proj = scene.proj_matrix[cam_id]

        for kpt_idx in kp3d:
            kpt = np.concatenate((kp3d[kpt_idx], np.ones((1))))
            projected = proj@kpt
            projected[:2] = projected[:2]/projected[2]

            cv2.circle(canvas, (int(projected[0]), int(projected[1])), radius=1, color=colors[int(kpt_idx)], thickness=4)
        cv2.imwrite('debug.png', canvas)
    cv2.destroyAllWindows()

    picked_indexs = []
    vis1 = o3d.visualization.VisualizerWithVertexSelection()
    vis1.create_window(window_name='Patch select')
    vis1.add_geometry(o3d_mesh)

    while True:
        vis1.update_geometry(o3d_mesh)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    picked_point = vis1.get_picked_points()
    meshcoord = []
    camcoord = []
    for pidx, point in enumerate(picked_point[::-1]):
        meshcoord.append(point.coord)
        picked_indexs.append(point.index)
        camcoord.append(kp3d[str(pidx)])
    print(picked_indexs)
    vis1.destroy_window()

    R, t, T, scale = rigid_transform_3D(np.stack(meshcoord), np.stack(camcoord), with_scale=True)
    mesh_test = copy.deepcopy(o3d_mesh)
    mesh_test.transform(T)

    return mesh_test, scale, R, t, T


def get_gt_data(scene, mask_root, cam_ids, device, computed_mask:dict=None, fidx=0, image_dir=None):
    batch_num = len(cam_ids)

    if computed_mask is not None:
        # for cam_id in cam_ids:
        obj_mask = computed_mask
    else:
        assert os.path.exists(mask_root) and len(os.listdir(mask_root))>0, 'cannot find Obejct Masks'

        if os.path.exists(mask_root) and len(os.listdir(mask_root))>0:
            # Read All Masks
            obj_mask = {}
            for cam_id in cam_ids:
                resizes_mask = scene.get_mask(cam_id, fidx=fidx)
                resized_mask = resizes_mask[...,0]
                resized_mask[resized_mask>0]=1
                obj_mask[cam_id] = resized_mask

    obj_mask_t_dict = {}
    obj_mask_stackted_t = []
    obj_masked_rgb_t_dict = {}
    obj_masked_rgb_stacked_t = []
    for cam_id in cam_ids:
        obj_mask_t = torch.FloatTensor(obj_mask[cam_id]).unsqueeze(0).to(device).requires_grad_(False) # 0: background 1: mask
        obj_mask_t_dict[cam_id] = obj_mask_t # 1XheightXwidth
        obj_mask_stackted_t.append(obj_mask_t)
        
        rgb_img_np = cv2.cvtColor(scene.get_image(cam_id, fidx=fidx, image_dir = image_dir), cv2.COLOR_BGR2RGB)
        rgb_img_t = torch.FloatTensor([rgb_img_np]).to(device).requires_grad_(False) # 1XhXwX3

        obj_masked_rgb_t = rgb_img_t.clone().detach()
        obj_masked_rgb_t[obj_mask_t==0] = 255
        obj_masked_rgb_t_dict[cam_id] = obj_masked_rgb_t/255
        obj_masked_rgb_stacked_t.append(obj_masked_rgb_t/255)

    obj_mask_stackted_t = torch.vstack(obj_mask_stackted_t).unsqueeze(-1)
    gt_combined_mask = obj_mask_stackted_t.transpose(1,0).reshape(scene.height, batch_num*scene.width).detach().cpu().numpy()[...,None]
    obj_masked_rgb_stacked_t = torch.vstack(obj_masked_rgb_stacked_t)
    gt_combined_maksed_rgb = obj_masked_rgb_stacked_t.transpose(1,0).reshape(scene.height, batch_num*scene.width,3).detach().cpu().numpy()

    return obj_mask_t_dict, obj_masked_rgb_t_dict, obj_mask_stackted_t, gt_combined_mask, obj_masked_rgb_stacked_t, gt_combined_maksed_rgb




# using yolo or predefined. 
def load_obj_masks(org_scene:Scene, reload_mask = False, debug=False, confidence=0.002, image_dir=None):
    if reload_mask:
        assert org_scene.mask_detector is not None, 'Scene should have mask detector.'

    if reload_mask:
        obj_name = org_scene.obj_nm

        yolo_module = org_scene.mask_detector
        mask_generation_st = time.time()
        detection_results = {}
        results_img = []
        for cam_id in org_scene.cam_ids:

            bgr_img = org_scene.get_image(cam_id, 0, image_dir = image_dir)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2BGR)
            detections = yolo_module.process_img(rgb_img)
            mask = detections.mask
            
            if len(mask)>0 and not check_mask(mask[0]):
                detections.confidence *= 0.0

            detection_results[cam_id] = detections

            if debug:
                canvas = yolo_module.annotate_image(bgr_img, detections, categories=yolo_module.categories, with_confidence=True)
                canvas = putText(canvas, cam_id, color=(0,0, 255))
                results_img.append(canvas)
                cv2.imwrite('test.png', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        if debug:
            cv2.imwrite(f'debug_grid_{obj_name}.png', make_grid_image_np(np.array(results_img), 4,6))

        confidence_dict = {cam_id: detection_results[cam_id].confidence.item() \
                           for cam_id in detection_results if detection_results[cam_id].confidence and detection_results[cam_id].confidence > confidence and cam_id not in hide_list}
        cam_N = 10
        top_n_cams2confidence = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)[:cam_N]
        top_n_cams = [cam_id for cam_id, confidence in top_n_cams2confidence]
        top_n_confidence = [confidence for cam_id, confidence in top_n_cams2confidence]

        mask_dict_org = {cam_id: get_binary_mask(detection_results[cam_id].mask[0]) for cam_id in top_n_cams}

        mask_generation_ed = time.time()
        print(f"GT mask generation {mask_generation_ed-mask_generation_st} sec")
    else:
        mask_dict_org = {}
        for cam_id in org_scene.cam_ids:
            mask = get_binary_mask(org_scene.get_mask(cam_id, fidx=0, image_dir = image_dir)[...,0])

            if check_mask(mask):
                mask_dict_org[cam_id] = mask
            

        top_n_cams = [key for key in mask_dict_org]
        top_n_confidence = [1 for key in mask_dict_org]

    
    return mask_dict_org, top_n_cams, top_n_confidence


def parse_and_crop(obj_mask_t_dict, obj_masked_rgb_t_dict):
        # Parse GT mask
    tg_rgbs, tg_sils = [], []
    tg_rgbs_full, tg_sils_full = [], []
    for cam_id in obj_masked_rgb_t_dict:
        # Get Maksed Image and silhouette 
        tg_rgbs_full.append(obj_masked_rgb_t_dict[cam_id][0].detach().cpu().numpy()*255)
        tg_sils_full.append(obj_mask_t_dict[cam_id][0].detach().cpu().numpy()*255)
        tg_rgb = crop_and_resize_by_mask(obj_masked_rgb_t_dict[cam_id][0].detach().cpu().numpy()*255,  \
                                        obj_mask_t_dict[cam_id][0].detach().cpu().numpy()*255, 124)
        tg_rgbs.append(tg_rgb)
        tg_sil = crop_and_resize_by_mask(obj_mask_t_dict[cam_id][0].unsqueeze(-1).expand(-1,-1,3).detach().cpu().numpy()*255,  \
                                    obj_mask_t_dict[cam_id][0].detach().cpu().numpy()*255, 124)
        tg_sils.append(tg_sil)
    tg_rgbs = np.stack(tg_rgbs)
    tg_sils = np.stack(tg_sils)
    tg_rgbs_full = np.stack(tg_rgbs_full)
    tg_sils_full = np.stack(tg_sils_full)

    return tg_rgbs, tg_sils, tg_rgbs_full, tg_sils_full



def render_and_compute_loss_all_cams(obj_dict, scene, rot_6d_t, initial_translate, default_scale,\
                                    tg_rgbs, tg_sils, tg_rgbs_full, tg_sils_full, \
                                    turn_on_filter=False, device='cuda'):
        
    trasnformed_obj_dict = get_transformed_obj(obj_dict, default_scale, rot_6d_t,\
                                                torch.tensor(initial_translate).to(device).float(), )
    rendered_rgb_batched, rendered_silhouette_batched = scene.batch_render(trasnformed_obj_dict)

    # Parse Rendered Image
    rendered_rgbs, rendered_sils = [], []
    for rendered_idx in range(rendered_rgb_batched.shape[0]):
        rendered_rgb = crop_and_resize_by_mask(rendered_rgb_batched[rendered_idx].detach().cpu().numpy()*255,  \
                                    rendered_silhouette_batched[rendered_idx,...,-1].detach().cpu().numpy()*255, \
                                    124)
        rendered_rgbs.append(rendered_rgb)
        rendered_sil = crop_and_resize_by_mask(rendered_silhouette_batched[rendered_idx,...,-1].unsqueeze(-1).expand(-1,-1,3).detach().cpu().numpy()*255,  \
                                    rendered_silhouette_batched[rendered_idx,...,-1].detach().cpu().numpy()*255, \
                                    124)
        rendered_sils.append(rendered_sil)
        
    rendered_rgbs = np.stack(rendered_rgbs)
    rendered_sils = np.stack(rendered_sils)
    rendered_rgbs_full = rendered_rgb_batched.detach().cpu().numpy()*255
    rendered_sils_full = rendered_silhouette_batched.squeeze().detach().cpu().numpy()*255

    rgb_diff_full = np.linalg.norm(rendered_rgbs_full.reshape(rendered_rgbs_full.shape[0],-1)\
                                    -tg_rgbs_full.reshape(tg_rgbs_full.shape[0],-1), axis=1)
    sil_diff_full = np.linalg.norm(rendered_sils_full.reshape(rendered_sils_full.shape[0],-1)\
                                    -tg_sils_full.reshape(tg_sils_full.shape[0],-1), axis=1)
    rgb_diff = np.linalg.norm(rendered_rgbs.reshape(rendered_rgbs.shape[0],-1)\
                                    -tg_rgbs.reshape(tg_rgbs.shape[0],-1), axis=1)
    sil_diff = np.linalg.norm(rendered_sils.reshape(rendered_sils.shape[0],-1)\
                                    -tg_sils.reshape(tg_sils.shape[0],-1), axis=1)
    # Compare Rendered Amount to Filter out wrong mask
    if turn_on_filter:
        in_mask_rendered = np.sum(rendered_sils_full.reshape(rendered_sils_full.shape[0],-1)/255, axis=1)
        in_mask_tg = np.sum(tg_sils_full.reshape(tg_sils_full.shape[0],-1)/255, axis=1)
        mask_percentage = np.minimum(in_mask_rendered, in_mask_tg)/np.maximum(in_mask_rendered, in_mask_tg)
        mask_filter = mask_percentage>0.5
        # Check RGB and Sillhouette Loss
        rgb_filter = np.logical_and(mask_filter, rgb_diff_full<=np.percentile(rgb_diff_full, 80))
        sil_filter = np.logical_and(mask_filter, sil_diff_full<=np.percentile(sil_diff_full, 80))

        if len(rgb_filter) <= 2 or len(sil_filter) <= 2: # If Candidate Fails
            rgb_loss = scene.height*scene.width*255
            sil_loss = scene.height*scene.width*255
        else:
            def compute_filtered_mean(diff, filter):
                thres = np.percentile(diff[filter],80)
                new_filter = np.logical_and(filter,diff<=thres)
                loss = (diff[new_filter]).mean()
                return loss, new_filter
            rgb_loss, rgb_filter = compute_filtered_mean(rgb_diff, rgb_filter) # Filter with Cropped Area Again
            sil_loss, sil_filter = compute_filtered_mean(sil_diff, sil_filter) # Filter with Cropped Area Again
            # rgb_loss = (rgb_diff[rgb_diff<np.percentile(rgb_diff, 40)]*top_n_confidence[rgb_diff<np.percentile(rgb_diff, 40)]).mean()
            # sil_loss = (sil_diff[sil_diff<np.percentile(sil_diff, 40)]*top_n_confidence[sil_diff<np.percentile(sil_diff, 40)]).mean()
    else:
        rgb_filter = np.ones(rgb_diff.shape[0], dtype=np.bool_)
        sil_filter = np.ones(sil_diff.shape[0], dtype=np.bool_)
        rgb_loss = np.mean(rgb_diff)
        sil_loss = np.mean(sil_diff)

    return rendered_rgbs, rendered_rgbs_full, rendered_sils, rendered_sils_full, \
            rgb_diff_full, sil_diff_full, rgb_diff, sil_diff, \
            rgb_filter, rgb_loss, sil_filter, sil_loss