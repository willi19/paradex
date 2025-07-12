Todo
1. Refine robot representation
    Arm : add eef frame and connector frame to the urdf
    Hand : add wrist frame and connector frame to the urdf

    Reason : we use wrist frame and eef frame frequently and this will cause problem when arm or hand chages

    Then fix paradex/robot/RobotWrapper and Unimanual teleop code to retarget with this notation.

    