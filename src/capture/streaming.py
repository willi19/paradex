import socket
import threading
import PySpin
import json
import time
from ..camera import camera
import os

def configure_camera(cam):
    """
    Configures the camera for video acquisition.
    """
    nodemap = cam.GetNodeMap()
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
    node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())

    print(f"Camera {cam.GetUniqueID()} configured to continuous acquisition mode.")

def send_to_main_pc(data, server_ip, server_port):
    """
    Sends data to the main PC via TCP.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip, server_port))
            s.sendall(json.dumps(data).encode('utf-8'))
    except Exception as e:
        print(f"Error sending data to main PC: {e}")

def capture_video(camera_index, duration, server_ip, server_port):
    """
    Captures video using Spinnaker's SpinVideo and sends metadata to the main PC.
    """
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    
    if cam_list.GetSize() <= camera_index:
        print(f"Camera index {camera_index} is out of range.")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    camPtr = cam_list.GetByIndex(camera_index)
    cameraLens = json.load(open("config/lens.json", "r"))
    lensinfo = json.load(open("config/camera.json", "r"))
    
    root = "/home/capture16/captures1"
    port = camera_index
    cam = camera.Camera(camPtr, None, cameraLens, lensinfo, root, port)
    
    try:
        cnt = 0
        start_time = time.time()
        while time.time() - start_time < duration:
            frame = cam.get_capture()
            cnt += 1
            # Send metadata or status to the main PC
            data = {
                "camera_index": camera_index,
                "frame_count": cnt,
                "timestamp": time.time(),
                "status": "Capturing"
            }
            send_to_main_pc(data, server_ip, server_port)
    finally:
        cam.stop_camera()
        del camPtr
        cam_list.Clear()
        system.ReleaseInstance()

def main():
    duration = 180  # Duration in seconds
    server_ip = "192.168.1.100"  # Replace with the IP address of the main PC
    server_port = 5000  # Replace with the listening port on the main PC

    threads = [
        threading.Thread(target=capture_video, args=(0, duration, server_ip, server_port)),
        threading.Thread(target=capture_video, args=(1, duration, server_ip, server_port)),
        threading.Thread(target=capture_video, args=(2, duration, server_ip, server_port)),
        threading.Thread(target=capture_video, args=(3, duration, server_ip, server_port)),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("All camera threads completed.")

if __name__ == "__main__":
    main()
