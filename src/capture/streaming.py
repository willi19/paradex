from multiprocessing import Process
import threading
import PySpin
import time
import json
from ..camera import camera
import os

def configure_camera(cam):
    """
    Configures the camera for video acquisition.
    """
    nodemap = cam.GetNodeMap()

    # Set acquisition mode to continuous
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
    node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())

    print(f"Camera {cam.GetUniqueID()} configured to continuous acquisition mode.")

def capture_video(camera_index, video_filename, duration):
    """
    Captures video using Spinnaker's SpinVideo and saves it to a file.
    :param camera_index: Index of the camera in the system.
    :param video_filename: Output video file name.
    :param duration: Duration to record (in seconds).
    """
    system = PySpin.System.GetInstance()

    # Get the list of cameras
    cam_list = system.GetCameras()
    if cam_list.GetSize() <= camera_index:
        print(f"Camera index {camera_index} is out of range.")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    camPtr = cam_list.GetByIndex(camera_index)#[camera_index]
    # cam_list.Clear()
    # for i in range(len(cam_list)):
    #     if i != camera_index:
    #         cam_list[i] = None

    cameraLens = json.load(open("config/lens.json","r"))
    lensinfo = json.load(open("config/camera.json","r"))
    
    root = "/home/capture16/captures1"  # Replace with the root path
    port = camera_index  # Port number
    cam = camera.Camera(camPtr, None, cameraLens, lensinfo, root, port)
    try:
        cam.set_record()
        cnt = 0
        while cnt < duration*30:
            cam.get_capture()
            cnt += 1
        cam.set_record()
    finally:
        cam.stop_camera()
        del camPtr 
        cam_list.Clear()
        system.ReleaseInstance()

def main():
    # Video recording duration in seconds
    duration = 180

    # Create processes for multiple cameras
    processes = [
        threading.Thread(target=capture_video, args=(0, "output_camera_0", duration)),
        threading.Thread(target=capture_video, args=(1, "output_camera_1", duration)),
        threading.Thread(target=capture_video, args=(2, "output_camera_2", duration)),
        threading.Thread(target=capture_video, args=(3, "output_camera_3", duration)),
    ]
    # capture_video(0,"asdf",duration)
    # Start all processes
    for p in processes:
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All camera processes completed.")

if __name__ == "__main__":
    main()