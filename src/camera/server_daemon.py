import time

from paradex.io.camera_system.camera_server_daemon import camera_server_daemon

if __name__ == "__main__":
    server = camera_server_daemon()
    while True:
        time.sleep(1)