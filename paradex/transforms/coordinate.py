import numpy as np

# see https://www.notion.so/Coordinate-system-21fee7e093e98037b92dfc69146158cc)}

# T ^ {Wrist : predefined wrist coordinate} _ {Device Wrist coordinate}
DEVICE2WRIST = {
    "xarm":np.array([[1, 0 ,0 ,0],
                     [0, 0, 1, -0.01],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]]),
    # "xarm":np.array([[1/np.sqrt(2), 0, -1/np.sqrt(2),0],[0, 1, 0, 0],[1/np.sqrt(2), 0, 1/np.sqrt(2), 0],[0, 0, 0, 1]]) @ np.array([[1, 0 ,0 ,0],
    #                  [0, 0, 1, -0.01],
    #                  [0, -1, 0, 0],
    #                  [0, 0, 0, 1]]),
    
    "franka":np.array([[1, 0 ,0 ,0],
                       [0, 0, 1, 0],
                       [0, -1, 1, 0],
                       [0, 0, 0, 1]]),
    
    "allegro":np.array([[0, 1 ,0 ,0],
                        [0, 0, 1, 0.04], #0.095 0.035
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]]), # fix this to connector latter, This is where connector is from global wrist
    
    # Hand is mounted reversed on this rig, so rotate it 180 deg about the wrist
    # (arm) axis. Original (before flip):
    #   [[1,0,0,0],[0,0,-1,0.025],[0,1,0,0],[0,0,0,1]]
    # Left-multiplied by Rz(180) = diag(-1,-1,1) — flip about the wrist z (arm) axis.
    "inspire":np.array([[-1, 0 , 0, 0],
                        [ 0, 0, 1, -0.025],
                        [ 0, 1, 0, 0],
                        [ 0, 0, 0, 1]]),
    
    "xsens_left":np.array([[1, 0, 0, 0], 
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]]),
    
    "xsens_right":np.array([[1, 0, 0, 0], 
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]]),
}

# T ^ {Global : predefined global coordinate} _ {Device coordinate}
DEVICE2GLOBAL = {
    "xarm":np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]),
    "franka":np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
}