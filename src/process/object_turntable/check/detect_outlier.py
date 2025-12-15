import os
import cv2
import numpy as np
import tqdm
import multiprocessing as mp
import tqdm

from paradex.image.undistort import undistort_img
from paradex.utils.path import shared_dir, home_path
from paradex.calibration.utils import load_camparam
from paradex.image.aruco import find_common_indices, merge_charuco_detection, detect_charuco
from paradex.transforms.conversion import SOLVE_XA_B


if __name__ == "__main__":
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
    for obj_name in sorted(os.listdir(root_dir)):
        obj_path = os.path.join(root_dir, obj_name)
        for index in os.listdir(obj_path):
            image_path = os.path.join(root_dir, obj_name, index, "selected")
            masked_image_path = os.path.join(root_dir, obj_name, index, "masked_images")
            mask_path = os.path.join(root_dir, obj_name, index, "masks")
            
            intrinsic, _ = load_camparam(os.path.join(root_dir, obj_name, index))
            dropped_frame = np.zeros((1536, 2048, 3), dtype=np.uint8)
            dropped_frame[::2, ::2] = 255
            
            if not os.path.exists(image_path):
                continue
            for serial_num in tqdm.tqdm(os.listdir(image_path)):
                framedrop_frame = undistort_img(dropped_frame.copy(), intrinsic[serial_num])
                serial_path = os.path.join(image_path, serial_num)
                img_files = os.listdir(serial_path)
                for img_file in img_files:
                    idx = int(img_file.split("frame_")[-1].split(".jpg")[0])
                    if idx % 40 != 1 or idx < 160:
                        if os.path.exists(os.path.join(serial_path, img_file)):
                            # print(f"Removing invalid frame: {os.path.join(serial_path, img_file)}")
                            os.remove(os.path.join(serial_path, img_file))
                        if os.path.exists(os.path.join(masked_image_path, serial_num, img_file)):
                            # print(f"Removing invalid masked image: {os.path.join(masked_image_path, serial_num, img_file)}")
                            os.remove(os.path.join(masked_image_path, serial_num, img_file))
                        if os.path.exists(os.path.join(mask_path, serial_num, img_file.replace(".jpg", ".png"))):
                            # print(f"Removing invalid mask image: {os.path.join(mask_path, serial_num, img_file.replace('.jpg', '.png'))}")
                            os.remove(os.path.join(mask_path, serial_num, img_file.replace(".jpg", ".png")))
                    else:
                        frame = cv2.imread(os.path.join(serial_path, img_file))
                        diff = cv2.absdiff(frame, framedrop_frame)
                        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                        non_zero_count = cv2.countNonZero(thresh)
                        if non_zero_count < 5000:
                            if os.path.exists(os.path.join(serial_path, img_file)):
                                print(f"Removing dropped frame: {os.path.join(serial_path, img_file)}")
                                os.remove(os.path.join(serial_path, img_file))
                            if os.path.exists(os.path.join(masked_image_path, serial_num, img_file)):
                                print(f"Removing dropped masked image: {os.path.join(masked_image_path, serial_num, img_file)}")
                                os.remove(os.path.join(masked_image_path, serial_num, img_file))
                            if os.path.exists(os.path.join(mask_path, serial_num, img_file.replace(".jpg", ".png"))):
                                print(f"Removing dropped mask image: {os.path.join(mask_path, serial_num, img_file.replace('.jpg', '.png'))}")
                                os.remove(os.path.join(mask_path, serial_num, img_file.replace(".jpg", ".png")))
