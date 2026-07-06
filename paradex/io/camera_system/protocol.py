"""Shared wire protocol constants for the camera command channel (5482).

Kept in its own tiny module so both the main-PC controller
(``remote_camera_controller``) and the capture-PC daemon
(``camera_server_daemon``) import the same values without dragging in each
other's heavy dependencies (PySpin, OpenCV, ...).
"""
import os

# Bumped when the command/reply shape changes incompatibly. The controller sends
# it on ``register``; the daemon compares and warns on a mismatch, so a capture PC
# left on stale code after a partial ``git_pull`` is caught instead of silently
# misbehaving.
PROTOCOL_VERSION = 1


def get_auth_token():
    """Return the shared command token, or ``None`` for no authentication.

    Read from ``PARADEX_CAMERA_TOKEN``. When unset (the default) the daemon
    accepts commands from any peer — fine on a closed lab LAN. Set the *same*
    value on every capture PC's daemon and on the controller to require it, so a
    stray process on the network can't drive the cameras.
    """
    return os.environ.get("PARADEX_CAMERA_TOKEN") or None
