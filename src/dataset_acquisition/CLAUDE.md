# CLAUDE.md — src/dataset_acquisition

## Purpose
Top-level group of per-dataset capture pipelines. Each subdir records raw multi-camera sessions (optionally with arm/hand/teleop) to `~/shared_data/capture/<dataset>/<name>/<timestamp>/`.

## Subdirs
- `graphics/` — exposure×gain image grids (`image_capture.py` static pose, `image_traj.py` per-waypoint). Free-run, no sync.
- `graphics/motion_blur/` — synced videos sweeping exposure/gain/speed over a joint trajectory (`capture.py` + `replay.py`).
- `hri/` — camera-only, keyboard-triggered HRI recordings.
- `miyungpa/` — XSens-teleoperated camera+arm+hand demos.
- `object_turntable/` — camera-only turntable captures (keyboard-triggered).

## Shared abstraction: CaptureSession
`paradex/dataset_acqusition/capture.py`. Constructor: `CaptureSession(camera=False, arm=None, hand=None, teleop=None, hand_ip=False)`.
- `camera=True` → builds `remote_camera_controller` + `UTGE900` sync generator; if arm or hand also set, adds a `TimestampMonitor`.
- `start(save_path, mode="video", fps=30, exposure_time=None, gain=None, stage=None)` — `save_path` is RELATIVE to `shared_dir`. Layout: `<save_path>/raw/` (or `raw/<stage>/`) with `videos|images/`, `arm/`, `hand/`, `teleop/`, `timestamps/`. Camera started with sync=True.
- `stop()` saves cam params + C2R via `save_current_camparam`/`save_current_C2R`, dumps teleop `state/` arrays.
- `teleop()` runs the XSens retarget loop, returns `"stop"` or `"exit"`.
- `end()` releases all devices.

## Gotchas
- Module is spelled `dataset_acqusition` (missing 'i') — do not "fix".
- `graphics/image_capture.py` and `image_traj.py` do NOT use CaptureSession — they call `remote_camera_controller` directly with `syncMode=False`.
- Trajectory default for graphics/motion_blur: `~/mcc_minimal/traj/dynamic/xarm/seed42_fwd100.npz` (key `q_deg`, degrees).

## When working here
Read the per-subdir CLAUDE.md for exact flow. Do not edit `.py` here without checking the `CaptureSession` contract above.
