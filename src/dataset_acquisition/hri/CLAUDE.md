# CLAUDE.md — src/dataset_acquisition/hri

## Purpose
Keyboard-triggered, camera-only capture of HRI sessions.

## Files
- `capture.py` — parses `--device/--arm/--hand/--name` (only `--name` used). `listen_keyboard({"c": save_event, "q": exit_event, "s": stop_event})`. Builds `CaptureSession(camera=True)`. Loop: wait for `c` → `cs.start("capture/hri_mingi/<name>/<ts>")` → spin until `s`/`q` → `cs.stop()`. On `q`: `cs.end()`.

## paradex modules used
- `paradex.dataset_acqusition.capture.CaptureSession`
- `paradex.utils.keyboard_listener.listen_keyboard`

## Data flow & IO
- Output: `~/shared_data/capture/hri_mingi/<name>/<ts>/raw/videos/...` (default mode="video", fps=30). Camparam + C2R saved on stop.
- Camera-only → CaptureSession has no TimestampMonitor; sync via UTGE900 still active.

## When working here
- `--device/--arm/--hand` are dead args (CaptureSession is hardcoded camera-only). Don't assume teleop/arm is wired.
- Output prefix is hardcoded `hri_mingi` (note the personal suffix).

## Gotchas
- Events are `threading.Event`; loop polls at 20 ms. `save_event` must be cleared after stop (it is).
