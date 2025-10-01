import sys, os
import numpy as np 
import torch 
import torch.optim as optim
import cv2
import time
from copy import deepcopy
import shutil
from pathlib import Path
PROJECT_PATH = Path(__file__).parent.parent
sys.path.append(str(PROJECT_PATH))
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from paradex.object_detection.optimize_module import EarlyStopping
from paradex.object_detection.obj_utils.geometry import rotation_6d_to_matrix, rotation_6d_to_matrix_np, \
                        matrix_to_rotation_6d, \
                        project_3d_to_2d, project_3d_to_2d_tensor
from paradex.object_detection.multiview_utils.img_processing import draw_pairs_wtext, draw_inliers, \
                            rendersil_obj2allview, SRC_COLOR, SET_COLOR, draw_match
from paradex.object_detection.obj_utils.vis_utils import make_grid_image_np
from paradex.object_detection.obj_utils.io import makevideo



def residual_func(params, src_3d, tg_2d, proj_mats, vis, tg_scene, obj_dict, item_list, \
                                                img_bucket, highlights, device):
    '''
        the minimization proceeds with respect to its first argument
    '''
    tmp_dir = './tmp_ceres'
    r6d = params[:6]
    t   = params[6:9]
    R = rotation_6d_to_matrix_np(r6d)
    
    transformed = src_3d @ R.T + t  # (N,3)
    projected = project_3d_to_2d(transformed, proj_mats, same_dim=True)
    residuals = (projected - tg_2d).reshape(-1)  # flatten
    
    if vis:
        T_opt = np.eye(4)
        T_opt[:3,:3] = R
        T_opt[:3,3] = t

        file_numb = len(os.listdir(tmp_dir))    
        rendered_on_overlaid = combined_visualizer(T_opt, tg_scene, obj_dict, item_list, \
                                                    img_bucket, highlights, device)
        cv2.imwrite(os.path.join(tmp_dir,'%03d.jpeg'%(file_numb)), rendered_on_overlaid)

    return residuals


def optimize_ceres(src_3d_arr:np.ndarray, tg_2d_arr:np.ndarray, proj_matrixes:np.ndarray, initial_T:np.ndarray, \
                        loop_numb=50, lr=1e-4, sampling_numb=None, device='cuda', \
                        vis=False, vis_tgs=()):
    '''
        src_3d_arr: NX3
        tg_2d_arr: NX2
        proj_matrixes: NX3X4
        intial_T : 3X4
    '''
    # x0: initial guess
    R0 = initial_T[:3,:3]
    t0 = initial_T[:3,3]
    # → 6D rotation representation
    # matrix_to_rotation_6d
    # u, s, vh = np.linalg.svd(R0)
    # R0 = u @ vh  # ensure orthogonal
    # flatten params
    r6d_0 = R0[:2].flatten()
    params0 = np.concatenate([r6d_0, t0])

    if vis:
        assert len(vis_tgs)==5, 'Should specify vis target if you want to visualie it'                     
        tg_scene, img_bucket, obj_dict, item_list, stepsize = vis_tgs
        
        tmp_dir = './tmp_ceres'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        
        optim_render_path = './tmp/optim'
        # if os.path.exists(optim_render_path):
        #     shutil.rmtree(optim_render_path)
        os.makedirs('./tmp', exist_ok=True)
        os.makedirs(optim_render_path, exist_ok=True)
        
        highlights = {}
        for matchingitem in item_list:
            highlights[matchingitem.cam_id] = SRC_COLOR
    else:
        tg_scene, img_bucket, obj_dict, item_list, stepsize, highlights = None,None,None,None,None,{}

    # run scipy optimization (Ceres-style)
    res = least_squares(
        residual_func,
        params0,
        args=(src_3d_arr, tg_2d_arr, proj_matrixes, vis, tg_scene, obj_dict, item_list, \
                                                img_bucket, highlights, device),
        method="trf",  # Trust Region Reflective
        max_nfev=loop_numb,
        verbose=0,
        ftol=1e-3
    )
    
    if vis:
        file_list = sorted(os.listdir(tmp_dir))
        file_list = [os.path.join(tmp_dir, file_nm) for file_nm in file_list]
        makevideo(file_list, os.path.join(optim_render_path,'rendered_pairs_optim.mp4'), fps=10, delete_imgs=True)

    # get optimized R, t
    r6d_opt = res.x[:6]
    t_opt = res.x[6:9]
    R_opt = rotation_6d_to_matrix_np(r6d_opt)
    T_opt = np.eye(4)
    T_opt[:3,:3] = R_opt
    T_opt[:3,3] = t_opt
    
    
    residual_output = residual_func(res.x, src_3d_arr, tg_2d_arr, proj_matrixes,\
                                vis, tg_scene, obj_dict, item_list, \
                                img_bucket, highlights, device)
    min_loss = np.mean(np.linalg.norm(residual_output.reshape((-1,2)), axis=1))


    
    # visualize output
    if vis:
        rendered_on_overlaid = combined_visualizer(T_opt, tg_scene, obj_dict, item_list, \
                                                    img_bucket, highlights, device)
        cv2.imwrite(f'tmp_pairs.jpeg', rendered_on_overlaid)
            # TODO make to video.

    
    optim_output = {'initial_T':initial_T, 'min_6d':r6d_opt, 'min_t':t_opt}
        

    return min_loss, optim_output


def optimize(src_3d_arr:np.ndarray, tg_2d_arr:np.ndarray, proj_matrixes:np.ndarray, initial_T:np.ndarray, \
                                        loop_numb=50, lr=1e-4, sampling_numb=None, device='cuda', \
                                        vis=False, vis_tgs=()):

        optim_st_time = time.time()

        obj_initial_T =  torch.tensor(initial_T, device=device).float()
        src_3d_tensor = torch.tensor(src_3d_arr, device=device).float()  
        tg_2d_tensor = torch.tensor(tg_2d_arr, device=device).float()  
        proj_matrixes_tensor = torch.tensor(proj_matrixes, device=device).float()  

        if vis:
            assert len(vis_tgs)==5, 'Should specify vis target if you want to visualie it'                     
            tg_scene, img_bucket, obj_dict, item_list, stepsize = vis_tgs

            highlights = {}
            for matchingitem in item_list:
                highlights[matchingitem.cam_id] = SRC_COLOR

            optim_render_path = './tmp/optim'
            # if os.path.exists(optim_render_path):
            #     shutil.rmtree(optim_render_path)
            os.makedirs('./tmp', exist_ok=True)
            os.makedirs(optim_render_path, exist_ok=True)

            rendered_pairs_paths = []

        min_loss = np.inf
        min_value = None
        
        # sampling if needed
        if sampling_numb is not None and src_3d_tensor.shape[0]>sampling_numb:
            idxes = torch.randperm(src_3d_tensor.shape[0])[:sampling_numb] 
            src_3d_tensor = src_3d_tensor[idxes]
            tg_2d_tensor = tg_2d_tensor[idxes]
            proj_matrixes_tensor = proj_matrixes_tensor[idxes]

            print(f"Optimization Pool {src_3d_tensor.shape[0]} and sampling using only {sampling_numb}")  

        optim_R = matrix_to_rotation_6d(obj_initial_T[:3,:3].detach().clone()).float().to(device).requires_grad_(True)
        optim_t = obj_initial_T[:3,3].detach().clone().float().to(device).requires_grad_(True)
        
        optimizer = optim.sgd([optim_R, optim_t], lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=loop_numb,
            eta_min=lr/2 
        )
        

        early_stopping = EarlyStopping(patience=int(loop_numb*0.2))

        loss_list = []
        for iter in range(loop_numb):
            transformed_verts = torch.einsum('mn, jn -> jm', rotation_6d_to_matrix(optim_R), \
                                            src_3d_tensor)+ optim_t
            # Dimension > important
            projected_joint = project_3d_to_2d_tensor(transformed_verts, proj_matrixes_tensor, same_dim=True)   
            
            loss = torch.mean(torch.linalg.norm(projected_joint-tg_2d_tensor, axis=1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            if min_loss>loss.item():
                min_value = (optim_R.clone().detach(),optim_t.clone().detach())
                min_loss = loss.item() 
                
            if iter%5==0:
                if vis:
                    optimized_T = torch.eye(4, device=device).float()
                    optimized_T[:3,:3] =  rotation_6d_to_matrix(optim_R.clone().detach())
                    optimized_T[:3,3] = optim_t.clone().detach()
                    
                    rendered_on_overlaid = combined_visualizer(optimized_T.detach().cpu().numpy(), \
                                                            tg_scene, obj_dict, item_list, \
                                                            img_bucket, highlights, device)
                    cv2.imwrite(os.path.join(optim_render_path, 'distance_%03d.jpeg'%(iter)), rendered_on_overlaid)
                    rendered_pairs_paths.append(os.path.join(optim_render_path, 'distance_%03d.jpeg'%(iter)))
                    
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter {iter:3d} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
            loss_list.append(min_loss)
    
            if early_stopping(loss.item()):
                print("Early Stopping triggered")
                break

        # After optimization
        if vis:
            makevideo(rendered_pairs_paths, os.path.join(optim_render_path,'rendered_pairs_optim.mp4'), True)

            optimized_T = torch.eye(4, device=device).float()
            optimized_T[:3,:3] =  rotation_6d_to_matrix(min_value[0])
            optimized_T[:3,3] = min_value[1]

            for tg_vis, T in {'before_optim': obj_initial_T,'after_optim': optimized_T}.items():
                rendered_on_overlaid = combined_visualizer(T, tg_scene, obj_dict, item_list, img_bucket, highlights, device)
                cv2.imwrite(f'tmp_pairs_{tg_vis}.jpeg', rendered_on_overlaid)
                # TODO make to video.

        optim_ed_time = time.time()
        print(f"Optimization takes {optim_ed_time-optim_st_time} sec")
        print(f"Loss Before {loss_list[0]} After:{loss_list[-1]}")
        
        optim_output = {'initial_T':obj_initial_T, 'min_6d':min_value[0], 'min_t':min_value[1]}

        return min_loss, optim_output


def draw_pairpoints_distance(matching_item_list, img_bucket, rotmat, trans):
    canvas_list = []
    canvas_dict = {}
    for matching_item in matching_item_list:
        combined_tg_2d = matching_item.combined_tg_2d[matching_item.inliers]
        combined_src_3d = matching_item.combined_src_3d[matching_item.inliers]
    
        transformed_verts = np.einsum('mn, jn -> jm', rotmat, combined_src_3d)+ trans
        projected_2d = project_3d_to_2d(transformed_verts, matching_item.proj_matrix[None]).squeeze().astype(np.int64)

        norm = np.mean(np.linalg.norm(projected_2d-combined_tg_2d, axis=1))
        canvas = deepcopy(img_bucket[matching_item.cam_id])
        lengths = np.linalg.norm(projected_2d-combined_tg_2d, axis=1)
        lengths/=max(lengths)
        text_input = f"{norm:.1f}"

        canvas = draw_pairs_wtext(canvas, projected_2d, combined_tg_2d.astype(np.int64), lengths, text_input)
        canvas_list.append(canvas)
        canvas_dict[matching_item.cam_id] = canvas
    
    img_list = []
    for cam_id in img_bucket:
        if cam_id in canvas_dict:
            img_list.append(canvas_dict[cam_id])
        else:
            img_list.append(img_bucket[cam_id])

    return  cv2.cvtColor(make_grid_image_np(np.stack(canvas_list), 4, 6),cv2.COLOR_BGR2RGB), \
            cv2.cvtColor(make_grid_image_np(np.stack(img_list), 4, 6),cv2.COLOR_BGR2RGB)


# def draw_img_result(matchingset, optim_R, optim_t, img_bucket, highlight_cams=[]):
#     canvas_list = []
#     for matching_item in matchingset.set:
#         combined_tg_2d = matching_item.combined_tg_2d
#         combined_src_3d = matching_item.combined_src_3d

#         # sampled_idxes = np.random.choice(combined_tg_2d.shape[0], int(combined_tg_2d.shape[0]*0.3), replace=False)
#         combined_src_3d = combined_src_3d
#         combined_tg_2d = combined_tg_2d
    
#         transformed_verts = np.einsum('mn, jn -> jm', rotation_6d_to_matrix(optim_R).detach().cpu().numpy(), \
#                                     combined_src_3d)+ optim_t.detach().cpu().numpy()
#         projected_2d = project_3d_to_2d(transformed_verts, matching_item.proj_matrix[None]).squeeze().astype(np.int64)

#         norm = np.mean(np.linalg.norm(projected_2d-combined_tg_2d, axis=1))
#         canvas = deepcopy(img_bucket[matching_item.cam_id])
#         lengths = np.linalg.norm(projected_2d-combined_tg_2d, axis=1)
#         lengths/=max(lengths)
#         text_input = f"{norm:.1f}"

#         canvas = draw_pairs_wtext(canvas, projected_2d, combined_tg_2d, lengths, text_input)

#         if matching_item.cam_id in highlight_cams:
#             h,w = canvas.shape[:2]
#             cv2.rectangle(canvas, (0, 0), (w-1, h-1), (0, 255, 255), 10)
#         canvas_list.append(canvas)

def combined_visualizer(T_arr, tg_scene, obj_dict, item_list, img_bucket, highlights, device):
    T = torch.tensor(T_arr, device=device).float() 
    _, rendered_dict = rendersil_obj2allview(tg_scene, obj_dict, T, img_bucket, highlights)
    _, rendered_on_overlaid = draw_pairpoints_distance(item_list, rendered_dict, \
                                                T_arr[:3,:3], T_arr[:3,3])
    return rendered_on_overlaid