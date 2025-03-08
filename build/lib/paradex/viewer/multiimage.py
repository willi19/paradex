import cv2
import numpy as np
from multiprocessing import Process
import time

from multiprocessing import shared_memory


class MultiStreamViewer:
    def __init__(self, w=640, h=480, is_streaming=False, shared_memories={}, update_flags={}, num_cameras=4, flag_value=1):
        """
        MultiStreamViewer for displaying multiple camera streams in a single window.

        :param queue: Multiprocessing Queue to receive images.
        :param w: Width of each camera image.
        :param h: Height of each camera image.
        :param num_cameras: Number of cameras (1–16).
        """
        self.w = w
        self.h = h
        self.num_cameras = num_cameras
        self.cam_index = {}
        self.serial_map = {}
        self.shared_memories = shared_memories
        self.update_flags = update_flags
        self.process = []
        self.flag_value = flag_value

    def stream_start(self):
        """
        Start the viewer process.
        """
        self.process = Process(target=self.stream_run, daemon=True)
        self.process.start()


    def stream_run(self):
        """
        Main loop for displaying images.
        """
        grid_size = self.calculate_grid_size(self.num_cameras)
        display_size = (grid_size[1] * self.w, grid_size[0] * self.h)
        canvas = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
        while True:
            for cam_id in range(self.num_cameras):
                update_flag = self.update_flags[cam_id]

                if update_flag.value == self.flag_value:  # New frame available
                    shm_info = self.shared_memories[cam_id]
                    with shm_info["lock"]:
                        frame = np.copy(shm_info["array"])
                        update_flag.value = 0  # Reset update flag

                    # Resize and overlay text
                    img = cv2.resize(frame, (self.w, self.h))
                    img = self.overlay_text(img, f"Camera {cam_id}")

                    # Add the image to the grid
                    row, col = divmod(cam_id, grid_size[1])
                    x_start, y_start = col * self.w, row * self.h
                    canvas[y_start:y_start + self.h, x_start:x_start + self.w] = img

            # Display the canvas
            cv2.imshow("MultiStream Viewer", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def calculate_grid_size(self, num_cameras):
        """
        Calculate the grid size (rows, cols) based on the number of cameras.
        """
        cols = int(np.ceil(np.sqrt(num_cameras)))
        rows = int(np.ceil(num_cameras / cols))
        return rows, cols

    def overlay_text(self, img, text):
        """
        Overlay text on the image.
        """
        overlay = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)  # Green
        position = (10, 20)  # Top-left corner
        cv2.putText(overlay, text, position, font, font_scale, color, thickness)
        return overlay

    def create_display_canvas(self, grid_size):
        """
        Create a blank canvas for displaying the grid.
        """
        return np.zeros((grid_size[0] * self.h, grid_size[1] * self.w, 3), dtype=np.uint8)

    def stream_stop(self):
        """
        Stop the viewer process.
        """
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    from ..camera.camera_loader import CameraManager
    shared_memories = {}
    update_flags = {}

    manager = CameraManager(num_cameras=4, duration=30, is_streaming=True, shared_memories=shared_memories, update_flags=update_flags)
    manager.start()

    # Initialize and start the viewer
    viewer = MultiStreamViewer(640, 480, True, shared_memories, update_flags, 4)
    viewer.stream_start()

    print("Main program is running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\nTerminating program...")
        viewer.stream_stop()
        manager.signal_handler()
