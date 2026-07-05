import signal
import sys
import time

from paradex.io.camera_system.camera_server_daemon import camera_server_daemon

if __name__ == "__main__":
    server = camera_server_daemon()

    def _shutdown(signum, frame):
        # Clean release on a normal kill / Ctrl-C (SIGKILL can't be caught).
        print(f"[Info] signal {signum} received.")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        time.sleep(1)
