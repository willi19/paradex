import time

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

if __name__ == "__main__":
    rcc = remote_camera_controller("test")

    rcc.arm(syncMode=True, fps=30)
    rcc.set_record("test2_1110", on=True)
    n = input("Press Enter to stop...")
    rcc.stop()
    rcc.end()
