import os, sys
import torch
import logging
import os, sys
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import cv2

import sys
from pathlib import Path
PROJECT_DIR = str(Path(__file__).absolute().parent.parent)
sys.path.append(PROJECT_DIR)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import logging

from paradex.object_detection.obj_utils.cnos_utils import BatchedData, CustomResizeLongestSide


device = 'cuda'

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

class CustomDINOv2:
    def __init__(
        self,
        model_name='dinov2_vitl14',
        token_name='x_norm_clstoken',
        image_size=224,
        chunk_size=16,
        patch_size=14,
        device = device
    ):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load(repo_or_dir='facebookresearch/dinov2', model=model_name).to(device)
        self.model_name = model_name
        self.token_name = token_name
        self.proposal_size = image_size
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.descriptor_width_size = descriptor_size[model_name]
        self.device = device
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.rgb_normalize_t = T.Compose(
            [
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        # use for global feature
        self.rgb_resize = CustomResizeLongestSide(
            self.descriptor_width_size, dividable_size=self.patch_size
        )
        logging.info(
            f"Init CustomDINOv2 with full size={self.descriptor_width_size} and proposal size={self.proposal_size} done!"
        )
        
    def resize_img(self, imgs):
        '''
            imgs:tensor, NX3XHXW, float(). normalized
        '''
        img_h, img_w = imgs.shape[2:]
        img_L = min(img_h, img_w)
        max(0,int((img_h-img_L)/2)),min(img_h,img_h-int((img_h-img_L)/2))
        max(0,int((img_w-img_L)/2)),min(img_w,img_w-int((img_w-img_L)/2))
        scale_factor = self.proposal_size / img_L
        
        # Crop If not 
        cropped_imgs = imgs[:,:,
            max(0,int((img_h-img_L)/2)):min(img_h,img_h-int((img_h-img_L)/2)),
            max(0,int((img_w-img_L)/2)):min(img_w,img_w-int((img_w-img_L)/2))]
        
        resized_imgs = F.interpolate(
            cropped_imgs, scale_factor=scale_factor
        )

        return resized_imgs
    
    def extract_feature(self, image):
        """
        1. Normalize image with DINOv2 transfom HXWX3
            if image is tensor > should be 0~1 float
        2. Resize RGB
        3. features
        """
        if torch.is_tensor(image):
            if image.ndim == 4:
                image = image.permute(0,3,1,2)
            if image.ndim == 3:
                image = image.permute(2,0,1)
            rgb = self.rgb_normalize_t(image).to(self.device).float()
        else:
            rgb = self.rgb_normalize(image).to(self.device).float()

        if rgb.ndim==3:
            rgbs = rgb.unsqueeze(0)
        else:
            rgbs = rgb

        resized_rgbs = self.resize_img(rgbs)

        # features = self.forward_by_chunk(resized_rgbs)
        rets = self.model.forward_features(resized_rgbs)

        # processed_masked_rgbs = self.rgb_proposal_processor(
        #     rgbs, boxes
        # )  # [N, 3, target_size, target_size]
        return rets


    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                features = self.forward_by_chunk(images)
            else:
                features = self.model(images)
        else:  # get both features
            raise NotImplementedError
        return features


    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_features(
                batch_rgbs[idx_batch], token_name="x_norm_clstoken"
            )
            features.cat(feats)
        return features.data
    

if __name__ == "__main__":
    image_path = '/home/jisoo/teserract_nas/mesh_template/attached_container/000000.png'
    img_np = cv2.imread(image_path)
    img_np_cropped = img_np[:,80:-80]
    dino_model = CustomDINOv2(device=device)
    dino_model.extract_feature(img_np)