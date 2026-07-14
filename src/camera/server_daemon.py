import argparse
import os
import signal
import sys
import time
from pathlib import Path

# Allow the documented command to run directly from a source checkout as well
# as from an editable install on a capture PC.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.io.camera_system.camera_server_daemon import camera_server_daemon

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paradex capture-PC camera agent")
    parser.add_argument(
        "--backend",
        choices=("pyspin", "aravis-gstreamer"),
        default=os.getenv("PARADEX_CAMERA_BACKEND", "pyspin"),
        help="Local camera backend. Use aravis-gstreamer after capture-PC setup.",
    )
    args = parser.parse_args()
    server = None

    def handle_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        server = camera_server_daemon(backend=args.backend)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Info] Shutdown requested.")
    finally:
        if server is not None:
            server.close()
