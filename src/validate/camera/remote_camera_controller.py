import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

if __name__ == "__main__":
    rcc = remote_camera_controller("test")

    rcc.start("image", False, "test1_1030")
    rcc.stop()

    rcc.start("full", False, "test1_1030", fps=30)
    time.sleep(2)
    rcc.stop()

    rcc.start("image", False, "test2_1030", fps=30)
    rcc.stop()

    rcc.start("video", False, "test2_1030", fps=5)
    time.sleep(2)
    rcc.stop()
    rcc.end()