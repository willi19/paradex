import os
import cv2

from paradex.utils.file_io import handeye_calib_path

name_list = os.listdir(handeye_calib_path)
name_list.sort()
name = name_list[-1]

index_list = os.listdir(os.path.join(handeye_calib_path, name))
for index in index_list:
    img = cv2.imread(os.path.join(handeye_calib_path, name, index, "overlay", "22645029.png"))
    cv2.imwrite(f"debug/{index}.png", img)