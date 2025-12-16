import cv2
from cv2 import aruco

squares_x, squares_y = 4, 5
square_length = 0.05   # meters 또는 “임의 단위”(일관성만 있으면 됨)
marker_length = 0.04

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

img = board.generateImage((2000, 2500))  # 픽셀 크기 (충분히 크게)
cv2.imwrite("charuco.png", img)
