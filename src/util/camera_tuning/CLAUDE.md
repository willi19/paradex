# CLAUDE.md — src/util/camera_tuning

## Purpose
Interactive per-camera image tuning. `live_tuner.py` shows all cameras live and
lets you push exposure/gain changes to the selected camera in real time, then save
them to `camera.json`.

## Files
- `remote_tuner.py` — **Main PC**, whole-rig. Tunes every camera across all capture
  PCs live **through the running daemon** (no daemon-off, no per-PC): `rcc.arm()` +
  `rcc.set_stream(True)`, SSH-launches `stream_client.py` for the preview
  (`DataCollector` receives JPEGs), merges + overlays exp/gain, and pushes edits with
  `rcc.set_param(gain/exposure={serial: val})`. `s` saves to the main-PC
  `camera.json`. Same keys as `live_tuner`. **Prefer this** for normal rig tuning.
- `live_tuner.py` — **Capture PC**, single-PC, daemon-off. Owns cameras directly:
  `autoforce_ip()` → `load_camera(serial)` → `PyspinCamera.start("continuous", ...)`.
  Grabs `get_image()`, merges, overlays, trackbars/keys adjust the selected camera,
  `s` saves to `camera.json`. Applies changes via the public `PyspinCamera.set_gain()`
  / `set_exposure()` (ExposureTime/Gain are writable during acquisition).

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
- `remote_tuner.py` (main PC) is the normal path — it goes through the daemon, so the
  daemon SHOULD be up and it does not touch hardware directly.
- `live_tuner.py` runs standalone on a capture PC — it opens the cameras directly, so a
  capture daemon must NOT be holding them.
- Live apply is now a first-class API: `PyspinCamera.set_gain/set_exposure` →
  `Camera.set_param` → `CameraLoader.set_param` → daemon `param` command →
  `rcc.set_param(gain/exposure=scalar or {serial: val})`. Use these, not `_config*`.

## Gotchas
- `get_image()` returns `None` on a grab timeout (P4 fix) — the loop draws a black
  tile instead of hanging, so a flaky camera won't freeze the tuner.
- Single-threaded sequential grab: fine for a low-fps viewer, not a capture path.
- camera.json key is `exposure` (not `exposure_time`).
