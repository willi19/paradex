import numpy as np
import cv2

def spin2cv(pImg, h, w):
    """
    Convert Spinnaker image to OpenCV format.

    Args:
        pImg: Spinnaker image object
        h (int): Image height
        w (int): Image width

    Returns:
        cvImg (np.ndarray): Converted OpenCV image
    """
    image_data = pImg.GetData()
    cvImg = np.array(image_data, dtype=np.uint8).reshape((h, w))
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BayerRG2RGB)
    return cvImg