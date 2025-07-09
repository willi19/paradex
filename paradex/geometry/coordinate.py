import numpy as np

# translation is from connector
DEVICE2WRIST = {
    "xarm":np.array([[1, 0 ,0 ,0],
                     [0, 0, 1, -0.01],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]]),
    
    "franka":np.array([[1, 0 ,0 ,0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]),
    
    "allegro":np.array([[0, 1 ,0 ,0],
                        [0, 0, 1, 0.035], #0.095 0.035
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]]), # fix this to connector latter, This is where connector is from global wrist
    
    "inspire":np.array([[1, 0 ,0 ,0],
                        [0, 0, -1, 0.025],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]]),
    
    "xsens_left":np.array([[1, 0, 0, 0], 
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]]),
    
    "xsens_right":np.array([[1, 0, 0, 0], 
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]]),
    
    "occulus":np.array([[1, 0 ,0 ,0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
}

DEVICE2GLOBAL = {
    "xsens":np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]),
    "occulus":np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]),
    "xarm":np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]),
    "franka":np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
}