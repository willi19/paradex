# CLAUDE.md — src/util/camera_tuning

## Purpose
Interactive per-camera image tuning. `live_tuner.py` shows all cameras live and
lets you push exposure/gain changes to the selected camera in real time, then save
them to `camera.json`.

## Files
- `live_tuner.py` — owns the cameras directly (single thread): `autoforce_ip()` →
  `load_camera(serial)` per camera → `PyspinCamera.start("continuous", False, frame_rate=fps)`.
  Main loop grabs `get_image()` from each, merges via `merge_image`, overlays
  `exp`/`gain` and highlights the selected camera. OpenCV trackbars (`camera`,
  `exp_us`, `gain_db`) + keys adjust the selected camera; `s` writes back to
  `camera.json`, `q` quits. Applies changes with `PyspinCamera._configureExposure()`
  / `_configureGain()` (ExposureTime/Gain are writable during acquisition).

## paradex modules used
- `paradex.io.camera_system.pyspin` — `get_serial_list`, `autoforce_ip`, `load_camera`, `PyspinCamera`
- `paradex.image.merge.merge_image`
- `paradex.utils.system.config_dir` (→ `system/current/camera.json`)

## Data flow & IO
`load_camera` applies the camera.json baseline; the tuner reads it back
(`cam.exposure_time`/`cam.gain`) as the starting slider values, pushes live edits to
the driver, and on `s` writes `{serial: {exposure, gain}}` back to
`system/current/camera.json` (key names must stay `exposure`/`gain`).

## When working here
- Run standalone on a capture PC — it opens the cameras directly, so a capture
  daemon must NOT be holding them.
- Reaching into `PyspinCamera._config*` is intentional; there is no public
  set-exposure-while-live API yet (see design/camera-recording-redesign.md P3, which
  proposes `set_exposure(serial, val)`).

## Gotchas
- `get_image()` returns `None` on a grab timeout (P4 fix) — the loop draws a black
  tile instead of hanging, so a flaky camera won't freeze the tuner.
- Single-threaded sequential grab: fine for a low-fps viewer, not a capture path.
- camera.json key is `exposure` (not `exposure_time`).
