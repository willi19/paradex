import cv2
import time

from paradex.io.camera_system.pyspin import get_serial_list, autoforce_ip, load_camera

try:
    serial_list = get_serial_list()
    print(serial_list)
except:
    print("Could not get serial list from pyspin.")

try:
    autoforce_ip()
    print("autoforce ip")
except:
    print("Could not autoforce ip")

cam_dict = {}
for serial_num in serial_list:
    try:
        cam_dict[serial_num] = load_camera(serial_num)
        print("camera loaded : ", serial_num)
    except:
        print("camera load failed : ", serial_num)
for serial_num in serial_list:
    try:
        cam_dict[serial_num].start("single", False)
        img, _ = cam_dict[serial_num].get_image()
        cv2.imwrite("tmp.png", img)
        cam_dict[serial_num].stop()
        print(cam_dict[serial_num].serial_num, "capturing image")
    except:
        print(cam_dict[serial_num].serial_num, "failed to capture image")
        
    try:
        cam_dict[serial_num].start("continuous", False, 10)
        start_time = time.time()
        for i in range(10):
            img, _ = cam_dict[serial_num].get_image()
            # cv2.imwrite(f"tmp_{i}.png", img)
            print(serial_num, "capture", i, "th image", time.time()-start_time)
        cam_dict[serial_num].stop()
    except:
        print(cam_dict[serial_num].serial_num, "failed to capture image")
        
    try:
        cam_dict[serial_num].start("continuous", False, 1)
        start_time = time.time()
        for i in range(10):
            img, _ = cam_dict[serial_num].get_image()
            # cv2.imwrite(f"tmp_{i}.png", img)
            print(serial_num, "capture", i, "th image", time.time()-start_time)
        cam_dict[serial_num].stop()
    except:
        print(cam_dict[serial_num].serial_num, "failed to capture image")
    
    try:
        cam_dict[serial_num].release()
        print("release", serial_num)
    except:
        print("release failed", serial_num)

