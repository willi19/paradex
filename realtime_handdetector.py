from ..viewer.multiimage import MultiImage
from ..camera.camera_loader import CameraLoader
from ..process.keypoint_detector import KeypointDetector

shared_memories = {}
update_flags = {}

if __name__ == "__main__":
    from ..camera.camera_loader import CameraManager
    shared_memories = {}
    update_flags = {}

    manager = CameraManager(num_cameras=4, duration=30, is_streaming=True, shared_memories=shared_memories, update_flags=update_flags)
    manager.start()

    # Initialize and start the viewer
    viewer = MultiStreamViewer(shared_memories, update_flags, 640, 480, 4)
    viewer.start()

    print("Main program is running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\nTerminating program...")
        viewer.stop()
        manager.signal_handler()
