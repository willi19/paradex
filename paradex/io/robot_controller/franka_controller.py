"""FR3 controller for hand-eye — now a thin shim over the UNIFIED controller.

The full implementation lives in the realtime_planning workspace so a single
class serves both hand-eye (blocking move / get_data) and the real-time policy
follower (streaming). This module keeps the historical import path and the exact
public surface hand-eye uses (FrankaController, FRANKA_HOME_QPOS) working.

  from paradex.io.robot_controller.franka_controller import (
      FrankaController, FRANKA_HOME_QPOS)
  c = FrankaController(step_size=0.15, step_time=0.5)   # unchanged call
  c.move(qpos, is_servo=False); c.get_data(); c.end()

If the realtime_planning module is not importable, the original implementation is
preserved verbatim in `franka_controller.py.orig_bak`.
"""

import os
import sys

# Base is realtime_planning/franka_controller.py (the merged controller).
_RT_PLAN = os.environ.get("RT_PLAN", os.path.expanduser("~/data2/realtime_planning"))
if _RT_PLAN not in sys.path:
    sys.path.insert(0, _RT_PLAN)

from franka_controller import (              # noqa: E402  (path set above)
    FrankaController as _Unified,
    Config,
    FRANKA_HOME_QPOS as _HOME,
)

# Hand-eye historically passed FRANKA_HOME_QPOS as a plain list; keep that type so
# nothing downstream that assumes a list breaks. (np.array(...) still works too.)
FRANKA_HOME_QPOS = [float(x) for x in _HOME]

# action DoF, kept for callers that imported it.
action_dof = 7


class FrankaController(_Unified):
    """Hand-eye default: effort JTC, background spin (fresh TF/state between moves),
    auto-recover a prior reflex on construction — matching the original behavior."""

    def __init__(self, step_size=0.15, step_time=0.5, min_steps=3, max_steps=40,
                 connect_timeout=10.0, auto_ready=True):
        super().__init__(
            Config(mode="jtc_effort"),
            step_size=step_size, step_time=step_time,
            min_steps=min_steps, max_steps=max_steps,
            connect_timeout=connect_timeout, auto_ready=auto_ready,
            spin_background=True,
        )
