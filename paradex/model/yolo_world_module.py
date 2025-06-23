from typing import List

import os
import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from inference.models.yolo_world import YOLOWorld

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent))
# from efficient_sam_module import load, inference_with_boxes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from video_module import (
#     generate_file_name,
#     calculate_end_frame_index,
#     create_directory,
#     remove_files_older_than
# )

class YOLO_MODULE:
    def __init__(self, model_id='yolo_world/v2-l', categories:str=None, device=DEVICE):
        # self.EFFICIENT_SAM_MODEL = load(device=DEVICE)
        self.YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

        # parse annotators
        self.BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
        self.MASK_ANNOTATOR = sv.MaskAnnotator()
        self.LABEL_ANNOTATOR = sv.LabelAnnotator()

        categories = self.process_categories(categories)
        self.categories = categories
        self.init_model(categories)
        

    def process_categories(self, categories: str) -> List[str]:
        return [category.strip() for category in categories.split(',')]

    def init_model(self, categories):
        self.YOLO_WORLD_MODEL.set_classes(categories)


    def process_img(self, input_image:np.ndarray,
            iou_threshold: float = 0.5,
            with_segmentation: bool = True,
            with_confidence: bool = False,
            with_class_agnostic_nms: bool = False,
            confidence: float = 0.001,
            top_1: bool = True
            ):
        
        results = self.YOLO_WORLD_MODEL.infer(input_image, confidence=confidence)
        detections = sv.Detections.from_inference(results)
        # remain only largest confidence items

        if top_1:
            detections = self.parse_detection(detections)

        detections = detections.with_nms(
            class_agnostic=with_class_agnostic_nms,
            threshold=iou_threshold
        )
        if with_segmentation:
            detections.mask = inference_with_boxes(
                image=input_image,
                xyxy=detections.xyxy,
                model=self.EFFICIENT_SAM_MODEL,
                device=DEVICE
            )
        # output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        # output_image = self.annotate_image(
        #     input_image=output_image,
        #     detections=detections,
        #     categories=self.categories,
        #     with_confidence=with_confidence
        # )
        # return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        return detections
    
    def parse_detection(self, detections):
        if len(detections)<=0:
            return detections
        class_names = detections.data['class_name']

        final_mask = None

        for class_name in class_names:
            class_mask = class_names==class_name
            top_confidence = max(detections.confidence[class_mask])

            if final_mask is None:
                final_mask = detections.confidence==top_confidence
            else:
                final_mask = final_mask|(detections.confidence==top_confidence)
                
        return detections[final_mask]
    

    def annotate_image(self,
        input_image: np.ndarray,
        detections: sv.Detections,
        categories: List[str],
        with_confidence: bool = False,
    ) -> np.ndarray:
        labels = [
            (
                f"{categories[class_id]}: {confidence:.3f}"
                if with_confidence
                else f"{categories[class_id]}"
            )
            for class_id, confidence in
            zip(detections.class_id, detections.confidence)
        ]
        output_image = self.MASK_ANNOTATOR.annotate(input_image, detections)
        output_image = self.BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
        output_image = self.LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
        return output_image




# def process_video(
#     input_video: str,
#     categories: str,
#     confidence_threshold: float = 0.3,
#     iou_threshold: float = 0.5,
#     with_segmentation: bool = True,
#     with_confidence: bool = False,
#     with_class_agnostic_nms: bool = False
# ) -> str:
#     # cleanup of old video files
#     remove_files_older_than(RESULTS, 30)

#     categories = process_categories(categories)
#     YOLO_WORLD_MODEL.set_classes(categories)
#     video_info = sv.VideoInfo.from_video_path(input_video)
#     total = calculate_end_frame_index(input_video)
#     frame_generator = sv.get_video_frames_generator(
#         source_path=input_video,
#         end=total
#     )
#     result_file_name = generate_file_name(extension="mp4")
#     result_file_path = os.path.join(RESULTS, result_file_name)
#     with sv.VideoSink(result_file_path, video_info=video_info) as sink:
#         for _ in tqdm(range(total), desc="Processing video..."):
#             frame = next(frame_generator)
#             results = YOLO_WORLD_MODEL.infer(frame, confidence=confidence_threshold)
#             detections = sv.Detections.from_inference(results)
#             detections = detections.with_nms(
#                 class_agnostic=with_class_agnostic_nms,
#                 threshold=iou_threshold
#             )
#             if with_segmentation:
#                 detections.mask = inference_with_boxes(
#                     image=frame,
#                     xyxy=detections.xyxy,
#                     model=EFFICIENT_SAM_MODEL,
#                     device=DEVICE
#             )
#             frame = annotate_image(
#                 input_image=frame,
#                 detections=detections,
#                 categories=categories,
#                 with_confidence=with_confidence
#             )
#             sink.write_frame(frame)
#     return result_file_path



if __name__ == '__main__':

    confidence_threshold=0.001
    yolo = YOLO_MODULE(categories='pringles')
    print("yolo module initialized")
    img = cv2.imread('/home/jisoo/teserract_nas/processed/toothbrush_holder/0/rgb_extracted/22684755/00000.jpeg')
    detections = yolo.process_img(img)

    output_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res_vis = yolo.annotate_image(output_image, detections, categories=yolo.categories, with_confidence=True)
    cv2.imwrite('test.png',res_vis)

    print("here")
