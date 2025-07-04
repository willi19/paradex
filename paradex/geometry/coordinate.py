import numpy as np

# T ^ {wrist}_{Xsens}
XSENS2WRIST_Left = np.array([[1, 0, 0, 0], 
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

XSENS2WRIST_Right = np.array([[1, 0, 0, 0], 
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])

XSENS2GLOBAL = np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

VIVE2WRIST = np.array([[1, 0, 0, 0], 
                        [0, 1, 0, 0], 
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1]])

