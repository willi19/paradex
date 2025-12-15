import torch
from PIL import Image
import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from paradex.utils.path import home_path, shared_dir
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, merge_charuco_detection
from paradex.image.image_dict import ImageDict
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.image.aruco import get_board_cor

import torch

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
model.eval()
torch.set_grad_enabled(False)

processor = Sam3Processor(model, confidence_threshold=0.5, device=str("cuda:0"))

def load_mask(root_dir, predictor):
    obj_name = os.path.basename(os.path.dirname(root_dir))
    print(f"Processing {obj_name} in {root_dir}")
    outdir = os.path.join(root_dir, "masks")
    outdir_img = os.path.join(root_dir, "masked_images")
    
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_img, exist_ok=True)

    image_dir = os.path.join(root_dir, "images")
    for serial_num in os.listdir(image_dir):
        out_dir = os.path.join(outdir, serial_num)
        os.makedirs(out_dir, exist_ok=True)
        out_dir_img =  os.path.join(outdir_img, serial_num)
        os.makedirs(out_dir_img, exist_ok=True)
        
        for img_file in os.listdir(os.path.join(image_dir, serial_num)):
            image = cv2.imread(os.path.join(image_dir, serial_num, img_file))
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            output = processor.set_text_prompt(state=inference_state, prompt=f"{obj_name} on the board")
            debug = plot_results(image, inference_state)
            plt.close("all")
            
            masks = output["masks"].cpu().numpy()
            
            import pdb; 
            
            
            
            
        
root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
for obj_name in ["yellow_clock"]:# os.listdir(root_dir):
    obj_path = os.path.join(root_dir, obj_name)
    for index in os.listdir(obj_path):
        demo_path = os.path.join("capture/object_turntable", obj_name, index)
        load_mask(os.path.join(home_path, "paradex_download", demo_path), processor)
        
