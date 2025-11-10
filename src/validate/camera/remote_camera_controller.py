import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

if __name__ == "__main__":
    rcc = remote_camera_controller("test")

    rcc.start("video", False, "test1_1110", fps=30)
    n = input("Press Enter to stop...")
    rcc.stop()
