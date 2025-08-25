import chime
import time

def home_robot(arm, pose):
    arm.home_robot(pose.copy())  
    
    while not arm.is_ready():
        time.sleep(0.1)

    chime.info()