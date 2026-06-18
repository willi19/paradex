import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.io.camera_system.monitor_daemon import CameraMonitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-remote",
        action="store_true",
        help="kill stale capture-PC camera daemons/stream clients before startup",
    )
    args = parser.parse_args()
    monitor = CameraMonitor(web_port=1234, clean_remote=args.clean_remote)
