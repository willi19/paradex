# CLAUDE.md — src/dataset_acquisition/object_turntable

## Purpose
Keyboard-triggered, camera-only capture of a turntable-mounted object for downstream COLMAP / mask / 6D-rotation processing.

## Files
- `capture.py` — parses `--name`. `listen_keyboard({"c": save_event, "q": exit_event, "s": stop_event})`. `CaptureSession(camera=True)`. Loop: wait for `c` → `cs.start("capture/object_turntable/<name>/<ts>")` → spin until `s`/`q` → `cs.stop()`. On `q`: `cs.end()`. (Identical control structure to `../hri/capture.py`.)

## paradex modules used
- `paradex.dataset_acqusition.capture.CaptureSession`
- `paradex.utils.keyboard_listener.listen_keyboard`

## Data flow & IO
- Output: `~/shared_data/capture/object_turntable/<name>/<ts>/raw/videos/...` (mode="video", fps=30). Camparam + C2R saved on stop.
- Camera-only → no TimestampMonitor; UTGE900 sync active.

## When working here
- Consumed by `src/process/object_turntable/` (extract_video → generate_colmap → extract_mask_sam3 → get_rotation). Keep output layout (`raw/videos/<serial>.*` + camparam) intact for those steps.

## Gotchas
- No `--arm/--hand/--device` args here (unlike hri/miyungpa); strictly camera-only.
- Module spelled `dataset_acqusition` (missing 'i').
