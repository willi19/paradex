import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.io.camera_system.camera_server_daemon import camera_server_daemon

if __name__ == "__main__":
    server = camera_server_daemon()
    while True:
        time.sleep(1)
