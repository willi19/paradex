"""Shared logging for paradex.

One helper, ``get_logger(name)``, returns a logger that writes to the console
*and* to the NAS log folder so a run's output survives on shared storage.

Layout (PC- and date-namespaced so the 6 capture PCs never clobber each other
when they all write to the same NAS)::

    ~/shared_data/log/<pc_name>/<YYYYMMDD>/<name>_<HHMMSS>.log

The file is created lazily on the first ``get_logger`` call for a given name in
a process, so one process run == one file. If the NAS is unreachable the file
handler is skipped and logging degrades to console-only rather than crashing the
caller.
"""

import logging
import os
from datetime import datetime

from paradex.utils.path import shared_dir, pc_name

log_root = os.path.join(shared_dir, "log")

_FORMAT = "%(asctime)s [%(name)s] %(levelname)-5s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _file_path(name):
    now = datetime.now()
    day_dir = os.path.join(log_root, pc_name, now.strftime("%Y%m%d"))
    os.makedirs(day_dir, exist_ok=True)
    return os.path.join(day_dir, f"{name}_{now.strftime('%H%M%S')}.log")


def get_logger(name):
    """Return a console+NAS logger for ``name`` (idempotent per process).

    ``name`` is used both as the logger name and the log-file stem, so keep it
    short and filesystem-safe (e.g. ``"xarm"``, ``"camera"``, ``"rcc"``).
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured this process — reuse same file
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        file_handler = logging.FileHandler(_file_path(name))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as e:  # NAS down / not mounted — keep console logging alive
        logger.warning("NAS log file unavailable (%s); logging to console only", e)

    return logger
