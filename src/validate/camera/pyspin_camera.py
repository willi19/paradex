from paradex.io.camera.camera import Camera
from paradex.io.camera.camera_loader import CameraManager
from paradex.io.camera.pyspin import get_serial_list

try:
    serial_list = get_serial_list()
except:
    print("Could not get serial list from pyspin.")
    
try:
    for serial_name in serial_list:
        cam = Camera("pyspin", serial_name)
except:
    print("Could not initialize pyspin camera with serial:", serial_name)


try:
    cameraloader1 = CameraManager()
    cameraloader2 = CameraManager()