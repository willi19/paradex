from paradex.io.camera_system.camera_loader import CameraLoader
import time

camera = CameraLoader()

for i in range(2):
    camera.start("full", False, "test_camloader",fps=30)
    start_time = time.time()    
    while True:
        error = camera.get_all_errors()
        if len(error) > 0:
            print("Errors detected:")
            for cam_name, (err, tb) in error.items():
                print(f"Camera {cam_name} error: {err}")
                print(tb)
            camera.stop()
            break
                
        if time.time() - start_time > 5:
            camera.stop()
            break
    
    camera.start("image", False, "test_camloader")
    camera.stop()
    
camera.end()
