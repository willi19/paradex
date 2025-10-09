import sys, os
from pathlib import Path
PROJECT_PATH = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_PATH)
import numpy as np
import cv2
from copy import deepcopy

import torch
from PIL import Image, ImageDraw
from paradex.object_detection.obj_utils.vis_utils import make_grid_image_np
from paradex.object_detection.multiview_utils.template import Template
from paradex.object_detection.multiview_utils.img_processing import crop_with_mask
from paradex.object_detection.default_config import ELOFTR_CKPT_PATH
# from utils.geometry import project_3d_to_2d_tensor, project_3d_to_2d
# TODO should check lofter
from thirdparty.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter
# Initialize the matcher with default settings

# TODO: add bundle adjustment

# device = 'cuda'

class MatcherTo3D:
    def __init__(self, device, img_L=256, ckpt_path = ELOFTR_CKPT_PATH):
        self.device=device
        _default_cfg = deepcopy(full_default_cfg)
        matcher = LoFTR(config=_default_cfg)

        # Load pretrained weights
        matcher.load_state_dict(torch.load(ckpt_path, weights_only=False)['state_dict'])
        matcher = reparameter(matcher)  # Essential for good performance
        matcher = matcher.eval().to(self.device)
        self.matcher = matcher
        self.img_L = img_L

    def process_img(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray_raw = cv2.resize(img_gray, (img_gray.shape[1]//32*32, img_gray.shape[0]//32*32))
        img_tensor = torch.from_numpy(img_gray_raw)[None][None].to(self.device) / 255.
        return img_tensor
    
    def process_imgs(self, imgs:list):
        processed_imgs = []
        for img in imgs:
            processed_imgs.append(self.process_img(img)[0])
        return torch.stack(processed_imgs)
    
    def process_mask(self, mask):
        # mask original shape HXWX3
        mask = cv2.resize(mask, (mask.shape[1]//32*32, mask.shape[0]//32*32))
        mask = torch.tensor(mask, device=self.device)
        parsed_mask = mask[None, ...,0]
        parsed_mask[parsed_mask>0]=1
        return parsed_mask
    
    def process_masks(self, masks:list):
        processed_masks = []
        for mask in masks:
            processed_masks.append(self.process_mask(mask)[0])
        return torch.stack(processed_masks)

    def get_color(self, score):
        '''
            Map score to RGB color
            score:0~1
        '''
        r = int(255 * (1 - score))
        g = int(255 * score)
        b = 0
        # return (1-score, score, 0)
        return (r,g,b)


    def match_pair(self, img0, img1, mask0=None, mask1=None):
        # Make Resized RGB Image
        img0_resized = cv2.resize(img0, (img0.shape[1]//32*32, img0.shape[0]//32*32))
        img1_resized = cv2.resize(img1, (img1.shape[1]//32*32, img1.shape[0]//32*32))

        batch = {'image0': self.process_img(img0), # resize and parser to tensor
                 'image1': self.process_img(img1)} # resize and parser to tensor
        
        # if mask0 is not None and mask1 is not None:
        #     batch['mask0'] = self.process_mask(mask0)
        #     batch['mask1'] = self.process_mask(mask1)

        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].detach().cpu().numpy()  # Matched keypoints in image0 xy
            mkpts1 = batch['mkpts1_f'].detach().cpu().numpy()  # Matched keypoints in image1
            mconf = batch['mconf'].detach().cpu().numpy()      # Matching confidence scores
            # mask check
            flag0 = mask0[mkpts0[:,1].astype(np.int64), mkpts0[:,0].astype(np.int64)][...,0]>0
            flag1 = mask1[mkpts1[:,1].astype(np.int64), mkpts1[:,0].astype(np.int64)][...,0]>0

            pair_output = {'keypoints0':mkpts0[flag0&flag1], 'keypoints1':mkpts1[flag0&flag1], 'matching_scores':mconf[flag0&flag1]}

        return img0_resized, img1_resized, batch, pair_output
    

    def match_pair_batch(self, img0_list, img1_list, mask0_list=None, mask1_list=None, conf_thres=0.5):
        # Make Resized RGB Image

        if mask0_list is not None and mask1_list is not None:
            mask0_tensor = self.process_masks(mask0_list)
            mask1_tensor = self.process_masks(mask1_list)
            mask0_flag = torch.sum(mask0_tensor, axis=(1,2)) > 0
            mask1_flag = torch.sum(mask1_tensor, axis=(1,2)) > 0
            mask_flag = mask0_flag&mask1_flag
            mask0_tensor = mask0_tensor[mask_flag].detach().cpu().numpy()
            mask1_tensor = mask1_tensor[mask_flag].detach().cpu().numpy()
            print(mask_flag)
        else:
            mask_flag = None
        
        # masked 
        img_idxes = [idx for idx, flag in enumerate(mask_flag) if flag]
        img0_resized = [cv2.resize(img, (img.shape[1]//32*32, img.shape[0]//32*32)) \
                                    for img_idx, img in enumerate(img0_list) if (mask_flag is None or mask_flag[img_idx])]
        img1_resized = [cv2.resize(img, (img.shape[1]//32*32, img.shape[0]//32*32)) \
                                    for img_idx, img in enumerate(img1_list) if (mask_flag is None or mask_flag[img_idx])]
        batch = {'image0': self.process_imgs(img0_list), # resize and parser to tensor
                 'image1': self.process_imgs(img1_list)} # resize and parser to tensor
        

        pair_outputs = [None for src_img in img0_list]

        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f']  # Matched keypoints in image0 xy
            mkpts1 = batch['mkpts1_f']  # Matched keypoints in image1
            mconf = batch['mconf']      # Matching confidence scores
            conf_flag = mconf>conf_thres
            b_ids = batch['b_ids']
            
            for bidx, img_idx in enumerate(img_idxes):
                keypoints0 = mkpts0[(b_ids==bidx)&conf_flag].detach().cpu().numpy().astype(np.int64)
                keypoints1 = mkpts1[(b_ids==bidx)&conf_flag].detach().cpu().numpy().astype(np.int64)
                
                if mask0_list is not None and mask1_list is not None:
                    in_mask0 = mask0_tensor[bidx][keypoints0[:,1], keypoints0[:,0]]>0
                    in_mask1 = mask1_tensor[bidx][keypoints1[:,1], keypoints1[:,0]]>0
                    
                    pair_outputs[img_idx] = {'keypoints0':keypoints0[in_mask0&in_mask1],
                                            'keypoints1':keypoints1[in_mask0&in_mask1],
                                            'matching_scores':mconf[(b_ids==bidx)&conf_flag].detach().cpu().numpy()[in_mask0&in_mask1]
                                            }
                else:
                    pair_outputs[img_idx] = {'keypoints0':keypoints0[in_mask0&in_mask1],
                        'keypoints1':keypoints1[in_mask0&in_mask1],
                        'matching_scores':mconf[(b_ids==bidx)&conf_flag].detach().cpu().numpy()[in_mask0&in_mask1]
                        }
            # pair_output = {'keypoints0':mkpts0[flag0&flag1], 'keypoints1':mkpts1[flag0&flag1], 'matching_scores':mconf[flag0&flag1]}

        return img0_resized, img1_resized, batch, pair_outputs
    

    

    def draw_match(self, img0_resized, img1_resized, pair_output):

        height0, width0 = img0_resized.shape[:2]
        height1, width1 = img1_resized.shape[:2]
        plot_image = np.zeros((max(height0, height1), width0 + width1, 3), dtype=np.uint8)
        plot_image[:height0, :width0] = img0_resized
        plot_image[:height1, width0:] = img1_resized

        plot_image_pil = Image.fromarray(plot_image)
        draw = ImageDraw.Draw(plot_image_pil)

        keypoints0_x, keypoints0_y = pair_output["keypoints0"].astype(np.int64).T
        keypoints1_x, keypoints1_y = pair_output["keypoints1"].astype(np.int64).T

        for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
            keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, pair_output["matching_scores"]
        ):
            if matching_score>0.5:
                color = self.get_color(matching_score)
                draw.line(
                    (keypoint0_x, keypoint0_y, keypoint1_x + width0, keypoint1_y),
                    fill=color,
                    width=3,
                )
                draw.ellipse((keypoint0_x - 2, keypoint0_y - 2, keypoint0_x + 2, keypoint0_y + 2), fill="black")
                draw.ellipse(
                    (keypoint1_x + width0 - 2, keypoint1_y - 2, keypoint1_x + width0 + 2, keypoint1_y + 2),
                    fill="black",
                )

        return cv2.cvtColor(np.array(plot_image_pil), cv2.COLOR_BGR2RGB)
    

    def match_img2template(self, tg_img, tg_mask, template:Template, threshold=40, batch_size = 6, draw=False, use_crop=False, image_name='test.jpeg'):
        src_3d_dict = {}
        tg_2d_dict= {}
        org_2d_dict = {}

        src_img_list = []
        src_mask_list = []
        tg_img_list = []
        tg_mask_list = []

        # use_crop
        src_similarityT_list = []
        tg_similarityT = None
        org_src_img_list = []
        org_tg_img_list = []

        for cam_id in template.img_template:
            src_img = template.img_template[cam_id]
            src_mask = template.mask_template[cam_id]
            if tg_mask is None:
                tg_mask = np.ones_like(src_mask)
            if use_crop:
                org_src_img_list.append(src_img)
                org_tg_img_list.append(tg_img)

                src_img_cropped, src_mask_cropped, src_similarityT = crop_with_mask(src_img, src_mask, output_L=self.img_L)
                src_similarityT_list.append(src_similarityT)
                tg_img_cropped, tg_mask_cropped, tg_similarityT = crop_with_mask(tg_img, tg_mask, output_L=self.img_L)

                src_img_list.append(src_img_cropped)
                src_mask_list.append(src_mask_cropped)
                tg_img_list.append(tg_img_cropped)
                tg_mask_list.append(tg_mask_cropped)
            else:
                src_img_list.append(src_img)
                src_mask_list.append(src_mask)
                tg_img_list.append(tg_img)
                tg_mask_list.append(tg_mask)
        template_cam_ids = [cam_id for cam_id in template.img_template] 
           
        matches = []
        for b_numb in range(int(np.ceil(len(tg_img_list)/batch_size))):
            src_resized, tg_resized, batch, pair_outputs = self.match_pair_batch(src_img_list[b_numb*batch_size:b_numb*batch_size+batch_size], \
                                                    tg_img_list[b_numb*batch_size:b_numb*batch_size+batch_size], 
                                                    src_mask_list[b_numb*batch_size:b_numb*batch_size+batch_size], 
                                                    tg_mask_list[b_numb*batch_size:b_numb*batch_size+batch_size])
            
        
            if use_crop: # Retransfer to Original Image Space
                new_pairoutputs = []
                tg_similarityT_inv = np.linalg.inv(tg_similarityT)
                for img_idx, pair_output in enumerate(pair_outputs):
                    if pair_output is not None:
                        src_similarityT = src_similarityT_list[b_numb*batch_size+img_idx]
                        src_similarity_T_inv = np.linalg.inv(src_similarityT)
                        pair_output['keypoints0'] = \
                            (src_similarity_T_inv@\
                             np.concatenate((pair_output['keypoints0'], np.ones((pair_output['keypoints0'].shape[0],1))),axis=1).T).T[:,:2].astype(np.int64)
                        pair_output['keypoints1'] = \
                            (tg_similarityT_inv@\
                             np.concatenate((pair_output['keypoints1'], np.ones((pair_output['keypoints1'].shape[0],1))),axis=1).T).T[:,:2].astype(np.int64)
                        
                        h,w = org_src_img_list[-1].shape[:2]
                        pair_output['keypoints0'][:, 0] = np.where(pair_output['keypoints0'][:, 0] < 0, 0, pair_output['keypoints0'][:, 0])
                        pair_output['keypoints0'][:, 0] = np.where(pair_output['keypoints0'][:, 0] >=w, w-1, pair_output['keypoints0'][:, 0])
                        pair_output['keypoints0'][:, 1] = np.where(pair_output['keypoints0'][:, 1] < 0, 0, pair_output['keypoints0'][:, 1])
                        pair_output['keypoints0'][:, 1] = np.where(pair_output['keypoints0'][:, 1] >=h, h-1, pair_output['keypoints0'][:, 1])
                        pair_output['keypoints1'][:, 0] = np.where(pair_output['keypoints1'][:, 0] < 0, 0, pair_output['keypoints1'][:, 0])
                        pair_output['keypoints1'][:, 0] = np.where(pair_output['keypoints1'][:, 0] >=w, w-1, pair_output['keypoints1'][:, 0])
                        pair_output['keypoints1'][:, 1] = np.where(pair_output['keypoints1'][:, 1] < 0, 0, pair_output['keypoints1'][:, 1])
                        pair_output['keypoints1'][:, 1] = np.where(pair_output['keypoints1'][:, 1] >=h, h-1, pair_output['keypoints1'][:, 1])
                        
                        new_pairoutputs.append(pair_output)
                    else:
                        new_pairoutputs.append(None)
                        
                bidx=0
                for img_idx, pair_output in enumerate(new_pairoutputs):
                    if pair_output is not None:
                        if draw:
                            match_img = self.draw_match(org_src_img_list[b_numb*batch_size+img_idx], org_tg_img_list[bidx], pair_output)
                            matches.append(match_img)
                        
                        cam_id = template_cam_ids[b_numb*batch_size+img_idx]
                        print(f"{cam_id}: {len(pair_output['keypoints0'])}")
                        if len(pair_output['keypoints0']) > threshold:
                            xs, ys = pair_output['keypoints0'].T
                            src_3d = template.img2point3d[cam_id][ys,xs]
                            in_3d_flag = (src_3d.sum(axis=1)!=0)
                            src_3d_dict[cam_id], tg_2d_dict[cam_id], org_2d_dict[cam_id] = \
                                    src_3d[in_3d_flag], pair_output['keypoints1'][in_3d_flag], pair_output['keypoints0'][in_3d_flag]
                        bidx+=1
            else:
                bidx=0
                for img_idx, pair_output in enumerate(pair_outputs):
                    if pair_output is not None:
                        if draw:
                            match_img = self.draw_match(src_resized[bidx], tg_resized[bidx], pair_output)
                            matches.append(match_img)
                        
                        cam_id = template_cam_ids[b_numb*batch_size+img_idx]
                        print(f"{cam_id}: {len(pair_output['keypoints0'])}")
                        if len(pair_output['keypoints0']) > threshold:
                            xs, ys = pair_output['keypoints0'].T
                            src_3d = template.img2point3d[cam_id][ys,xs]
                            in_3d_flag = (src_3d.sum(axis=1)!=0)
                            src_3d_dict[cam_id], tg_2d_dict[cam_id], org_2d_dict[cam_id] = \
                                    src_3d[in_3d_flag], pair_output['keypoints1'][in_3d_flag], pair_output['keypoints0'][in_3d_flag]
                        bidx+=1
                                                        
        print("after finding matches")
        if draw and len(matches)>0:
            print("Saved Img to ", image_name)
            cv2.imwrite(image_name, make_grid_image_np(np.array(matches), 4, np.ceil(len(matches)/4).astype(int)))

        return src_3d_dict, tg_2d_dict, org_2d_dict
    


    def match_img2template_at_once(self, tg_img, template:Template, threshold=40):
        '''
            matching image and image one by one.
        '''
        src_3d_dict = {}
        tg_2d_dict= {}
        org_2d_dict = {}

        tg_mask = np.ones_like(src_mask)
    
        # make 3d to 2d pairs
        matches = []
        for cam_id in template.img_template:
            # src from template
            src_img = template.img_template[cam_id]
            src_mask = template.mask_template[cam_id]

            if len(src_mask[src_mask>0])>0:
                src_resized, tg_resized, batch, pair_output = \
                                            self.match_pair(src_img, tg_img, src_mask, tg_mask)
                match_img = self.draw_match(src_resized, tg_resized, pair_output)
                print(len(pair_output['matching_scores']>0.5))
                if len(pair_output['matching_scores']>0.5) > threshold:
                    src_3d_dict[cam_id], tg_2d_dict[cam_id], org_2d_dict[cam_id] = [], [], []

                    for src_kpt, tg_kpt, score in zip(pair_output['keypoints0'], pair_output['keypoints1'], pair_output['matching_scores']):
                        if score > 0.5:
                            src_3d = template.img2point3d[cam_id][int(src_kpt[1]),int(src_kpt[0])] # src_kpt : x,y
                            if src_3d.sum()!=0:
                                tg_2d = tg_kpt.astype(np.int64)
                                src_3d_dict[cam_id].append(src_3d)
                                tg_2d_dict[cam_id].append(tg_2d)
                                org_2d_dict[cam_id].append(src_kpt.astype(np.int64))

                matches.append(match_img)                        

        print("after finding matches")
        cv2.imwrite('test.png', np.vstack(matches))

        return src_3d_dict, tg_2d_dict, org_2d_dict

if __name__ == "__main__":
    matcherto3d = MatcherTo3D(device='cuda', img_L=256)