import sys, os
from pathlib import Path
PROJECT_PATH = Path(__file__).parent.parent
sys.path.append(PROJECT_PATH)
import numpy as np
import pickle
import cv2
import time
import shutil
import torch
from paradex.object_detection.obj_utils.scene import Scene
from paradex.object_detection.obj_utils.vis_utils import  make_grid_image_np
from paradex.object_detection.obj_utils.geometry import project_3d_to_2d
from paradex.object_detection.multiview_utils.template import Template
from paradex.object_detection.multiview_utils.matcher import MatcherTo3D
from paradex.object_detection.multiview_utils.optimizer import combined_visualizer
from paradex.object_detection.multiview_utils.matchingset import MatchItem, MatchingSet, group_optimization
from paradex.object_detection.multiview_utils.img_processing import draw_inliers, \
                            rendersil_obj2allview, SRC_COLOR, draw_text
from paradex.object_detection.obj_utils.geometry import rotation_6d_to_matrix_np, \
                        matrix_to_rotation_6d
from paradex.object_detection.multiview_utils.multiview_parsing import combine_multiview_matching, parsing_inlier2flag
import matplotlib.pyplot as plt
cmap = plt.get_cmap("jet") # jet, viridis, plasma, coolwarm
device = 'cuda'

import json
default_template_path = PROJECT_PATH/'multiview_template2tracking'/'default_template.json'
default_template = json.load(open(default_template_path, 'r'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--scene_path', type=str, required=True)
parser.add_argument('--template_path', type=str, default=None)
parser.add_argument('--obj_name', type=str, default=None)
parser.add_argument('--vis', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--img_L', type=int, default=256)
parser.add_argument('--batch', type=int, default=8)
args = parser.parse_args()

if 'multi' in args.scene_path:
    image_dir_nm = 'undistorted_images'
else:
    image_dir_nm = None

torch.manual_seed(0)

if __name__ == '__main__':
    obj_name = args.scene_path.split("/")[-2]
    dino_obj_name = args.obj_name if args.obj_name is not None else obj_name
    if args.template_path is None:
        args.template_path = default_template[args.scene_path.split("/")[-2]]
    
    template = Template(args.template_path, obj_name=obj_name) # template has mask and rgb image. 
    tg_scene = Scene(scene_path=Path(args.scene_path), rescale_factor=0.5, \
                    mask_dir_nm=f'mask_hq/{obj_name}', image_dir_nm=image_dir_nm,\
                    mask_module='yolo', use_sam=True, dino_obj_nm=dino_obj_name)
    tg_scene.get_batched_renderer(tg_scene.cam_ids)
    matcherto3d = MatcherTo3D(device, img_L=args.img_L)
    paircount_threshold = 30
    inliers_threshold = 30
    
    org_scaled_verts = template.obj_dict['verts'][0].clone().detach()
    sampled_indexes = torch.randperm(org_scaled_verts.shape[0])[:100]
    sampled_obj_verts = org_scaled_verts[sampled_indexes]

    if tg_scene.ttl_frame_length == 0: # Image
        tg_scene.ttl_frame_length = 1

    objoutput_path = os.path.join(args.scene_path,'obj_output')
        
    # prepare rendering directory
    oneview_dir = Path(objoutput_path+'/one_view')
    os.makedirs(oneview_dir, exist_ok=True)
    optimization_dir = Path(objoutput_path+'/optim')
    os.makedirs(optimization_dir, exist_ok=True)
    set_root_dir = Path(objoutput_path+'/set')
    os.makedirs(set_root_dir, exist_ok=True)


    for tg_frame in range(0,tg_scene.ttl_frame_length, 10): # frame wise
        img_bucket = {}
        detection_bucket = {}
        
        det_st_time = time.time()
        # Get Image and Mask
        for cam_id in tg_scene.cam_ids:
            detections = tg_scene.get_multidetection(cam_id, tg_frame)
            rgb_image = cv2.cvtColor(tg_scene.get_image(cam_id=cam_id, fidx=tg_frame), cv2.COLOR_BGR2RGB)
            # pringles mask
            # cv2.imwrite(f'test{cam_id}.png',tg_scene.mask_detector.annotate_image(np.copy(rgb_image), detections=detections, categories=['pringles']))
            img_bucket[cam_id] = rgb_image
            detection_bucket[cam_id] = detections
        det_ed_time = time.time()
        print(f'detection takes {det_ed_time-det_st_time}')

        if args.debug: # View All Images
            img_list = []
            for cam_id in template.img_template:
                img_list.append(template.img_template[cam_id])
            cv2.imwrite(objoutput_path+'/template.jpeg', make_grid_image_np(np.array(img_list),4,6))

        matching_bucket = {}
        init_st_time = time.time()

        ttl_mask_numb = 0 # total mask number which at least one correspondence
        if os.path.exists(objoutput_path+'/matching_buckets.pkl'):
            matching_bucket = pickle.load(open(objoutput_path+'/matching_buckets.pkl','rb'))
        else:
            # Get macthing based on detections
            for tg_cam_id in tg_scene.cam_ids: # tg_scene.cam_ids is already sorted
                # Projection matrix
                tg_img = img_bucket[tg_cam_id]
                tg_detections = detection_bucket[tg_cam_id]
                matching_bucket[tg_cam_id] = {}
                # for each dection
                for midx, tg_mask in enumerate(tg_detections.mask):
                    st_time = time.time()
                    src_3d_dict, tg_2d_dict, org_2d_dict = \
                        matcherto3d.match_img2template(tg_cam_id, tg_img, np.repeat(tg_mask[..., None], 3, axis=2).astype(np.int64)*255.0, \
                                                    template, paircount_threshold, batch_size=args.batch, \
                                                    draw=(args.vis and args.debug), use_crop=True, image_name = f'./tmp/{tg_cam_id}_{midx}.jpeg')
                    # check how many points from one image connected to 3d points. more better.
                    pair_count = 0
                    for cam_id in src_3d_dict:
                        if len(src_3d_dict[cam_id])>0:
                            pair_count+=len(src_3d_dict[cam_id])
                    ed_time = time.time()
                    print(f"Get matching {ed_time-st_time} sec")
                    matching_bucket[tg_cam_id][midx] = {'count':pair_count, 'src_3d':src_3d_dict, 'tg_2d':tg_2d_dict, 'src_2d':org_2d_dict}
                    if pair_count>0:
                        ttl_mask_numb+=1
            ed_time = time.time()
            print(f"Finding pairs ended in {ed_time-init_st_time}")

            pickle.dump(matching_bucket, open(objoutput_path+'/matching_buckets.pkl','wb'))
        
        tg_cam_id_list = [cam_id for cam_id in matching_bucket]
        initial_3d_bucket = {}
        
        st_time = time.time() 
        
        matching_bucket = combine_multiview_matching(matching_bucket)
        
        # Get Target View PnP - One DB to get 3D
        for cidx, tg_cam_id in enumerate(matching_bucket): 
            tg_cam_extr_4X4 = np.eye(4)
            tg_cam_extr_4X4[:3] = tg_scene.cam2extr[tg_cam_id]
            proj_matrix = tg_scene.cam2intr[tg_cam_id]@tg_scene.cam2extr[tg_cam_id]
            
            initial_3d_bucket[tg_cam_id] = {}
            tg_img = img_bucket[tg_cam_id]

            for midx in matching_bucket[tg_cam_id]:
                tmp_matching = matching_bucket[tg_cam_id][midx]
                if tmp_matching['count'] > 0:
                    # 1. Combined
                    # SOLVEPNP_SQPNP 
                    '''
                        bool	useExtrinsicGuess = false,
                        int	iterationsCount = 100,
                        float	reprojectionError = 8.0,
                        double	confidence = 0.99,
                        OutputArray	inliers = noArray(),
                        int	flags = SOLVEPNP_ITERATIVE )
                    '''
                    combined_src_3d, combined_tg_2d = tmp_matching['combined_src_3d'], tmp_matching['combined_tg_2d']
                    src_cam_ids = tmp_matching['src_arr_cam_ids']
                    
                    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                        combined_src_3d,
                        combined_tg_2d,
                        tg_scene.cam2intr[tg_cam_id], distCoeffs=None,
                        reprojectionError=8,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    
                    if ret:
                        obj2img_matrix = np.eye(4)
                        obj2img_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
                        obj2img_matrix[:3, 3]  = tvec[:, 0]
                        obj_tg_T = torch.tensor(np.linalg.inv(tg_cam_extr_4X4)@obj2img_matrix, device=device).float()
                        initial_3d_bucket[tg_cam_id][midx] = (obj_tg_T, inliers)

                        inliers_mask = np.zeros((combined_src_3d.shape[0]))
                        inliers_mask[inliers]=1
                        inlier_flags = parsing_inlier2flag(inliers, src_cam_ids) # 
                        inlier_percentage = inliers.shape[0]/combined_src_3d.shape[0]
                        inlier_number = sum([1 if inlier_flags[cam_id].any() else 0 for cam_id in inlier_flags]) # inlier number mean inlier DB image number

                        tmp_matching['inliers'] = inliers_mask
                        tmp_matching['inliers_count'] = len(inliers)

                        if args.vis and args.debug:
                            #  Render To All View
                            transformed_verts = np.einsum('mn, jn -> jm', obj_tg_T[:3,:3].detach().cpu().numpy(), combined_src_3d)+ obj_tg_T[:3,3].detach().cpu().numpy()
                            projected_2d = project_3d_to_2d(transformed_verts, proj_matrix[None]).squeeze().astype(np.int64)
                            mean_distance_inlier = np.sum(inliers_mask*np.linalg.norm((projected_2d-combined_tg_2d),axis=1))/np.sum(inliers_mask)
                            rendered_sil, _ = rendersil_obj2allview(tg_scene, template.obj_dict, obj_tg_T, img_bucket, \
                                                            highlight={tg_cam_id:SRC_COLOR})
                            cv2.imwrite(str(oneview_dir/f'{tg_cam_id}_{midx}_using_combined_{inliers.shape[0]}_{mean_distance_inlier}_{inlier_number}.jpeg') ,\
                                rendered_sil)

                            inlier_compare_list = []
                            for src_cam_id in tmp_matching['src_3d']:
                                match_img, match_img_inliers = draw_inliers(template.img_template[src_cam_id], tg_img, \
                                                                            tmp_matching['src_2d'][src_cam_id], tmp_matching['tg_2d'][src_cam_id], \
                                                                            inlier_flags[src_cam_id])
                                inlier_compare_list.append(np.hstack((match_img, match_img_inliers)))
                            cv2.imwrite(str(oneview_dir/f'{tg_cam_id}_{midx}_using_combined_inliers_{inlier_percentage}.jpeg'), np.vstack(inlier_compare_list))
                        
                        # draw_inlier outlier projection loss

                            
        ed_time = time.time()
        print(f"Get All PnP and Combine DB for each view {ed_time-st_time} sec")

        # # visualize to 3D
        # import open3d as o3d
        # o3d_mesh, scaled = get_initial_mesh(args.obj_name.split(" ")[0], post_processing=True)
        # vis_o3dmesh_list = []
        # for tg_cam_id in initial_3d_bucket:
        #     for midx in initial_3d_bucket[tg_cam_id]:
        #         vis_o3dmesh_list.append(deepcopy(o3d_mesh).transform(initial_3d_bucket[tg_cam_id][midx].detach().cpu().numpy()))
        

        matchingitem_dict = {}
        st_time = time.time()
        # make matching item
        for cidx, tg_cam_id in enumerate(matching_bucket):
            proj_matrix = tg_scene.cam2intr[tg_cam_id]@tg_scene.cam2extr[tg_cam_id]
            # For each mask
            for midx in matching_bucket[tg_cam_id]:
                tmp_matching = matching_bucket[tg_cam_id][midx]
                if tmp_matching['count']>0 and tmp_matching['inliers_count'] > inliers_threshold:
                    transformed_verts = torch.einsum('mn, jn -> jm', initial_3d_bucket[tg_cam_id][midx][0][:3,:3], \
                                                sampled_obj_verts)+ initial_3d_bucket[tg_cam_id][midx][0][:3,3]

                    new_item = MatchItem(cam_id=tg_cam_id, midx=midx, matching_item=tmp_matching,\
                        detection=detection_bucket[tg_cam_id][midx], \
                        initial_T=initial_3d_bucket[tg_cam_id][midx][0], \
                        transformed_verts=transformed_verts.detach().cpu().numpy(),
                        proj_matrix=proj_matrix)

                    matchingitem_dict[f'{tg_cam_id}_{midx}'] = new_item
                    # translated_T = np.copy(new_item.initial_T)
                    
                    
        # Two View Test (All Possible Pairs)
        twoview_optim = False
        if twoview_optim:
            matching_keys = list(matchingitem_dict.keys())
            for idx1 in range(len(matching_keys)):
                for idx2 in range(idx1+1, len(matching_keys)):
                    key1, key2 = matching_keys[idx1], matching_keys[idx2]
                    if key1.split("_")[0] != key2.split("_")[0]:
                        tg_list = [matchingitem_dict[key1], matchingitem_dict[key2]]
                        T = (tg_list[0].initial_T + tg_list[1].initial_T)/2
                        min_loss, optim_output = group_optimization(tg_list, T, \
                                                                tg_scene, img_bucket, template.obj_dict, \
                                                                loop_numb=30, stepsize=2, vis=False, use_ceres=True)
                        
                        # optim_output = {'initial_T':initial_T, 'min_6d':r6d_opt, 'min_t':t_opt}
                        T_opt = np.eye(4)
                        T_opt[:3,:3] = rotation_6d_to_matrix_np(optim_output['min_6d'])
                        T_opt[:3,3] = optim_output['min_t']
                        highlights = {}
                        for matchingitem in tg_list:
                            highlights[matchingitem.cam_id] = SRC_COLOR
                        rendered_on_overlaid = combined_visualizer(T_opt, tg_scene, template.obj_dict, tg_list, \
                                                                    img_bucket, highlights, device)
                        rendered_on_overlaid = draw_text(rendered_on_overlaid, f'loss: {min_loss}')
                        cv2.imwrite(f'./tmp/optim_using_{key1}_{key2}.jpeg', rendered_on_overlaid)
        
        # Manual 3 View Optim
        
        # inlier counting 수대로 sort
                

        # Make Set using Optimization
        matchingset_list = []
        keys_sorted = sorted(matchingitem_dict.keys(), key=lambda k: matchingitem_dict[k].inlier_count, reverse=True)

        for key in keys_sorted:
            print(f"** {key} Optimization")
            tg_cam_id, midx = key.split("_")
            new_item = matchingitem_dict[key]
            
            min_validset_idx = None
            min_valid_distance = None
            min_optim_output = None

            valid = False
            for sidx, exising_set in enumerate(matchingset_list):
                # check with optim
                # NOTE: translation thres: I increased translation threshold because few pnp output result in bad translation (no depth input)
                # validate compatibility: check center distance + run group optimization and return and filtering result
                loss, distance, optim_output = exising_set.validate_compatibility(new_item, obj_dict=template.obj_dict, \
                                                                        translation_thres=0.3, loss_thres=10, \
                                                                        loop_numb=30, vis=(args.vis and args.debug), ceres=True)
                print(f'loss:{loss} distance:{distance} optim_ouptput:{optim_output}')

                # move rendered image target information: loss, set number
                if loss is not None: # optimization run
                    if optim_output is not None: # Optimization Run
                        valid = True 
                        if min_validset_idx is None or min_valid_distance>distance:
                            min_validset_idx = sidx
                            min_valid_distance = distance
                            min_optim_output = optim_output
                    if (args.vis and args.debug):
                        shutil.move('./tmp/optim/rendered_pairs_optim.mp4', str(optimization_dir/f'rendered_pairs_optim_set{exising_set.idx}_{tg_cam_id}_{midx}.mp4'))
                        target_path = str(optimization_dir/f'set{exising_set.idx}_{new_item.cam_id}_{new_item.midx}_{(loss is not None and optim_output is not None)}_loss_{loss}_match.jpeg')
                        shutil.copy('tmp_pairs.jpeg', target_path)
                #     else: # Not Converged (no less than 10)
                #         print("Not Converged")
                # else: # translation filtering
                #     print("Not Converged")

            if len(matchingset_list)>0 and valid:
                matchingset_list[min_validset_idx].add(new_item)
                matchingset_list[min_validset_idx].update_T(min_optim_output)
            else:
                matchingset_list.append(MatchingSet(len(matchingset_list), new_item, tg_scene=tg_scene, img_bucket=img_bucket))

        ed_time = time.time()
        print(f"Finding Match Ended in  {ed_time-st_time}")
        print(f"Overall in  {ed_time-init_st_time}")


        # # Matching Debugging
        # for mid in matchingitem_dict: 
        #     min_loss, optim_output = group_optimization([matchingitem_dict[mid]], matchingitem_dict[mid].initial_T, \
        #                                             tg_scene, img_bucket, template.obj_dict, vis=args.vis)
        #     print(f"Optim_{mid}")
        
        # NOTE: FINAL output visualization. 
        output_dict = {}
        output_idx = 0
        reoptim = False
        for matchingset in matchingset_list: 
            if len(matchingset.set) >= 2: # TODO: check thres
                    if args.vis:
                        if reoptim:
                            firstitem = list(matchingset.set)[0]

                            min_loss, optim_output = group_optimization(list(matchingset.set), matchingset.optim_T, \
                                                                    tg_scene, img_bucket, template.obj_dict, loop_numb=30, stepsize=2, vis=True, use_ceres=True)
                            print(f'loss:{loss} optim_ouptput:{optim_output}')
                            
                            if min_loss is not None and optim_output is not None:
                                shutil.move('./tmp/optim/rendered_pairs_optim.mp4', str(set_root_dir/f'rendered_pairs_optim_set{matchingset.idx}_{tg_cam_id}_{midx}.mp4'))
                                valid = True if optim_output is not None else False
                                target_path = os.path.join(set_root_dir,f'set{matchingset.idx}_{valid}_loss_{min_loss}_match.jpeg')
                                shutil.copy('tmp_pairs.jpeg', target_path)

                    if not reoptim:
                        target_path = os.path.join(set_root_dir,f'set{matchingset.idx}_match.jpeg')
                        highlights = {}
                        for matchingitem in list(matchingset.set):
                            highlights[matchingitem.cam_id] = SRC_COLOR

                        rendered_on_overlaid = combined_visualizer(matchingset.optim_T, tg_scene, template.obj_dict, list(matchingset.set), \
                                            img_bucket, highlights, device)
                        cv2.imwrite(target_path, rendered_on_overlaid)                        
                    output_dict[output_idx] = matchingset.optim_T
                    output_idx+=1

        pickle.dump(output_dict,open(os.path.join(objoutput_path,'obj_T.pkl'),'wb'))



