import os
import signal
import sys
import time
import socket
import logging
from logging.handlers import RotatingFileHandler

from paradex.io.camera_system.camera_server_daemon import camera_server_daemon


def _setup_logging():
    """Persist all daemon output to a rotating log file (in addition to console).

    ``run_script`` launches the daemon with stdout -> /dev/null, so without this
    the ``[ERROR]``/traceback prints would vanish. This tees stdout/stderr into a
    rotating log file so errors survive on the capture PC.

    Directory: ``PARADEX_LOG_DIR`` (default ``~/shared_data/logs``, so the main PC
    can read every capture PC's log in one place; falls back to ``~/.paradex/logs``
    if that path is not writable). File: ``camera_daemon_<hostname>.log``.
    """
    log_path = None
    for d in (os.path.expanduser(os.environ.get("PARADEX_LOG_DIR", "~/shared_data/logs")),
              os.path.expanduser("~/.paradex/logs")):
        try:
            os.makedirs(d, exist_ok=True)
            log_path = os.path.join(d, f"camera_daemon_{socket.gethostname()}.log")
            handler = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
            break
        except Exception:
            log_path = None
    if log_path is None:
        return None

    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger = logging.getLogger("camera_daemon")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    class _Tee:
        """Write to the original stream and mirror each line into the log file."""
        def __init__(self, stream):
            self.stream = stream

        def write(self, msg):
            if self.stream is not None:
                self.stream.write(msg)
                self.stream.flush()
            line = msg.rstrip()
            if line:
                logger.info(line)

        def flush(self):
            if self.stream is not None:
                self.stream.flush()

    sys.stdout = _Tee(sys.__stdout__)
    sys.stderr = _Tee(sys.__stderr__)
    print(f"[Info] camera daemon logging to {log_path}")
    return log_path


if __name__ == "__main__":
    _setup_logging()
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
