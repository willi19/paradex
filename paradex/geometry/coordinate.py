import numpy as np

DEVICE2WRIST = {
    "xarm":np.array([[],[],[],[]]),
    "franka":np.array([[],[],[],[]]),
    "allegro":np.array([[],[],[],[]]),
    "inspire":np.array([[],[],[],[]]),
    "xsens_left":np.array([[1, 0, 0, 0], 
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]]),
    "xsens_right":np.array([[1, 0, 0, 0], 
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]]),
    "occulus":np.array([[],[],[],[]])
}

DEVICE2GLOBAL = {
    "xsens":np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]),
    "occulus":pass,
    "xarm":pass,
    "franka":pass
}