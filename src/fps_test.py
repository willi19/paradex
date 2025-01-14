import time
import threading
from .camera import camera
import PySpin as ps
import json
# Global variables
key_running = True
stop_flag = False

def enable_logging():
    pass

def monitor_camera(camera_instance):
    """
    Monitors camera frames, calculates FPS, and logs frame drops in real-time.
    """
    global stop_flag

    print(f"Starting Camera: {camera_instance.serialnum}")

    while camera_instance.is_capturing:
        frame, valid = camera_instance.get_capture()

        if valid:
            fps = camera_instance.get_fps()
            # print(f"[Camera {camera_instance.serialnum}] FPS: {fps:.2f}, Frame Count: {camera_instance.curframe}")
        else:
            print(f"[Camera {camera_instance.serialnum}] Frame Dropped")

        # Check for stop flag
        if stop_flag:
            break

    camera_instance.stop_camera()
    print(f"Camera {camera_instance.serialnum} stopped.")

def listen_for_stop():
    """
    Listens for the 'q' key to stop the program.
    """
    global stop_flag, key_running

    print("Press 'q' to stop the program.")
    while key_running:
        if input().strip().lower() == 'q':
            stop_flag = True
            key_running = False
            break

def main():
    global key_running

    # Initialize the Camera
    system = ps.System.GetInstance()
    # interface_list = system.GetInterfaces()
    # for pInterface in interface_list:
    #     camera_list = pInterface.GetCameras()

    camPtr = system.GetCameras().GetByIndex(0)
    processor = None  # Your image processor instance
    cameraLens = json.load(open("config/lens.json","r"))
    lensinfo = json.load(open("config/camera.json","r"))
    root = "/home/capture16/captures1"  # Replace with the root path
    port = 1  # Port number
    enable_logging()
    cam = camera.Camera(camPtr, processor, cameraLens, lensinfo, root, port)

    # Start camera monitoring
    camera_thread = threading.Thread(target=monitor_camera, args=(cam,))
    key_thread = threading.Thread(target=listen_for_stop)

    camera_thread.start()
    key_thread.start()

    camera_thread.join()
    key_thread.join()

    key_running = False
    print("Program exited.")

if __name__ == "__main__":
    main()

