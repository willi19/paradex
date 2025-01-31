import socket
import threading
import PySpin
import json
import time
from multiprocessing import Process, Queue, Event
import signal
import sys
from ..camera import camera
from ..utils.image_util import spin2cv
from ..process.keypoint_detector import HandKeypointDetector
import cv2

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


def capture_video(camera_index, duration, frame_queue, stop_event):
    """
    Captures video using Spinnaker's SpinVideo and sends frames to a processing queue.
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
    cam = camera.Camera(camPtr, None, cameraLens, lensinfo, root, port, syncMode=True)

    try:
        start_time = time.time()
        while time.time() - start_time < duration and not stop_event.is_set():
            frame, ret = cam.get_capture()
            img = spin2cv(frame, 1536, 2048)
            # Send frame to the processing queue
            if ret:
                frame_queue.put((camera_index, img))
    finally:
        cam.stop_camera()
        del camPtr
        cam_list.Clear()
        system.ReleaseInstance()


def process_frames(frame_queue, num_cam, stop_event):
    """
    Processes frames for hand detection and sends results to the main PC.
    """
    detector = HandKeypointDetector()
    while not stop_event.is_set():
        try:
            frame_data = frame_queue.get(timeout=1)  # Timeout to check for stop_event
            if frame_data is None:
                # Stop signal received
                break

            camera_index, frame = frame_data
            img = spin2cv(frame)

            # Perform heavy hand detection (placeholder logic)
            hand_keypoints = detector.detect_keypoints(frame)  # This should return hand landmarks

            # Build the result to send to the server
            if hand_keypoints:
                # Format detected keypoints as a list of dictionaries
                landmarks = []
                for hand in hand_keypoints:
                    landmarks.append([
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in hand.landmark
                    ])

                # Send structured data to the main PC
                data = {
                    "camera_index": camera_index,
                    "hands_detected": len(hand_keypoints),  # Number of hands detected
                    "keypoints": landmarks,  # List of keypoints for each hand
                    "timestamp": time.time(),  # Timestamp of detection
                }
            else:
                # If no hands detected, send an empty response
                data = {
                    "camera_index": camera_index,
                    "hands_detected": 0,
                    "keypoints": [],
                    "timestamp": time.time(),
                }

            send_to_main_pc(data, server_ip, server_port)

        except Exception as e:
            print(f"Error in processing: {e}")


def signal_handler(stop_event, threads, processes):
    """
    Handles SIGINT to terminate all threads and processes gracefully.
    """
    print("\nSIGINT received. Terminating all threads and processes...")
    stop_event.set()

    # Terminate threads
    for t in threads:
        if t.is_alive():
            print(f"Joining thread {t.name}")
            t.join(timeout=2)

    # Terminate processes
    for p in processes:
        if p.is_alive():
            print(f"Terminating process {p.name}")
            p.terminate()

    sys.exit(0)


def main():
    duration = 180  # Duration in seconds
    
    # Queue to hold frames for processing
    frame_queue = Queue()
    stop_event = Event()

    # Processes for capturing video from multiple cameras
    capture_processes = [
        Process(target=capture_video, args=(i, duration, frame_queue, stop_event))
        for i in range(4)
    ]

    # Thread for heavy processing
    processing_thread = threading.Thread(
        target=process_frames, args=(frame_queue, num_cam=4, stop_event), daemon=True
    )

    # Signal handler for Ctrl-C
    def sigint_handler(sig, frame):
        signal_handler(stop_event, [processing_thread], capture_processes)

    signal.signal(signal.SIGINT, sigint_handler)  # Register signal handler

    # Start all capture processes
    for p in capture_processes:
        p.start()

    # Start the processing thread
    processing_thread.start()

    try:
        # Wait for all capture processes to finish
        for p in capture_processes:
            p.join()

        # Send stop signal to the processing thread
        frame_queue.put(None)

        # Wait for processing thread to finish
        processing_thread.join()
    except KeyboardInterrupt:
        sigint_handler(None, None)

    print("All processes and threads completed.")


if __name__ == "__main__":
    main()
