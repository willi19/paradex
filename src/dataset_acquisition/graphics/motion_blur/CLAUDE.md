# CLAUDE.md — src/dataset_acquisition/graphics/motion_blur

## Purpose
Synced multi-camera video capture of an xarm replaying a trajectory, sweeping (exposure,gain) pairs × speed. Motion-blur dataset.

## Files
- `capture.py` — main. Loads `q_deg` from `--traj`, writes `meta.json`, builds `CaptureSession(camera=True, arm="xarm", hand=None, teleop=None)`. Per cell: `_capture_trial` resets to `q_deg[0]`, sleeps, `cs.start(base, mode="video", fps, exposure_time, gain, stage)`, `replay_q_deg`, `cs.stop()`. `stage = exp{e}_gain{g}_spd{s}`.
- `replay.py` — `reset_to(controller, q_deg, speed_rad_s=0.35)` blocking `move(is_servo=False)`; `replay_q_deg(controller, q_deg_traj, speed_deg_s)` interpolates each segment at SAMPLE_HZ=100, sleeps to maintain dt, `move(q_cmd_rad, is_servo=True)`. Pre-warms with 20 servo commands to q[0].
- `__init__.py` — empty package marker.

## paradex modules used
- `paradex.dataset_acqusition.capture.CaptureSession`
- `paradex.utils.path.shared_dir`
- (via replay) `XArmController.move` servo interface; numpy only otherwise.

## Data flow & IO
- `cs.start(base, ...)` writes to `shared_dir/<base>/raw/<stage>/{videos,arm,timestamps}`; camera started sync=True; UTGE900 fires at `fps`.
- `base = capture/graphics/motion_blur/<name>/<ts>` (relative; CaptureSession prepends `shared_dir`).
- `meta.json` written at `base_abs` top level.

## When working here
- `capture.py` does `sys.path.insert(0, dirname(__file__))` then `from replay import ...` — replay is imported as a top-level module, NOT a package-relative import. Keep that or imports break.
- `--exposures`/`--gains` paired 1:1; `--speeds` is an independent outer-product dimension.

## Gotchas
- `replay_q_deg` is open-loop servo streaming at 100 Hz; speed is enforced via interpolated step count, not the controller.
- Trajectory units are degrees (`q_deg`); replay converts to rad internally.
- Sibling sharp-image scripts in [`../`](..) deliberately bypass CaptureSession; this one uses it (so sync + camparam/C2R saving happen here).
