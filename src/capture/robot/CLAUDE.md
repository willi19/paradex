# CLAUDE.md — src/capture/robot

## Purpose
Robot-side data collection: teleop demonstration recording (`teleop_real.py`) and
manual XArm waypoint teaching (`xarm_teaching.py`).

## Files
- `teleop_real.py` — main/robot PC. Builds `CaptureSession(camera=False, arm, hand,
  teleop=device)`. Loop: `cs.teleop()` (pre-record) → if not `"exit"`, `cs.start(<save_path>/<ts>)`
  → `cs.teleop()` (record) → `cs.stop()`. Breaks on `"exit"`. `--device {xsens,occulus}`,
  `--arm`, `--hand`, `--save_path`.
- `xarm_teaching.py` — PC wired to XArm. Direct `XArmAPI` (no paradex robot wrapper).
  Enables manual mode (`set_mode(2)`), then on `c` keypress saves `get_joint_states()[0][:6]`
  and `aa2mtx(get_position_aa())`. `q` quits and restores `set_mode(0)`, disables motion.

## paradex modules used
- `paradex.dataset_acqusition.capture.CaptureSession` (teleop_real)
- `paradex.transforms.conversion.aa2mtx` (xarm_teaching)
- `paradex.utils.keyboard_listener.listen_keyboard`
- `paradex.utils.system.network_info` (xarm IP)
- `xarm.wrapper.XArmAPI` (third-party, direct)

## Data flow & IO
- `teleop_real.py`: `CaptureSession` records arm/hand/state to `<save_path>/<timestamp>/`.
  Session boundaries are controlled by teleop gestures, not keyboard — `CaptureSession.teleop()`
  returns `"stop"` (state==2 held ~90 ticks) or `"exit"` (state==3 held ~90 ticks).
- `xarm_teaching.py`: writes `<idx>_qpos.npy` (6 joint radians) and `<idx>_aa.npy`
  (4x4 matrix) per `c` press into `--save_path`.

## When working here
- `teleop_real.py` is gesture-driven; `xarm_teaching.py` is keyboard-driven (`c`/`q`).
- `CaptureSession.teleop()` is called twice per session: first to "prepare", then to
  "record" while `start()` is active.

## Gotchas
- The `--device` choice is spelled `occulus` (sic), not `oculus`.
- `xarm_teaching.py` talks to the XArm SDK directly (not via `paradex.io.robot_controller`);
  it leaves manual mode and disables motion on exit — don't skip that cleanup.
- `--save_path` is optional in `xarm_teaching.py`; without it, poses are read but not saved.
- `teleop_real.py` sets `camera=False` — it records robot data only, no images.
