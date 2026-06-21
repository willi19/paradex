# CLAUDE.md — src/dataset_acquisition/miyungpa

## Purpose
XSens-teleoperated camera + arm + hand demonstration capture, driven by the teleop state machine.

## Files
- `capture.py` — parses `--device/--arm/--hand/--name` (`--name` required). `CaptureSession(camera=True, arm=args.arm, hand=args.hand, teleop=args.device)`. Loop: `cs.teleop()` (prepare); if `"exit"` break. Else `cs.start("capture/miyungpa/<name>/<ts>")`, `cs.teleop()` (record), `cs.stop()`; break if `"exit"`. Finally `cs.end()`.

## paradex modules used
- `paradex.dataset_acqusition.capture.CaptureSession` (+ its `teleop()` retarget loop → `Retargetor`, `HandStateExtractor`, `XSensReceiver`).

## Data flow & IO
- Output: `~/shared_data/capture/miyungpa/<name>/<ts>/raw/{videos,arm,hand,teleop,timestamps}` + `state/{state_hist,state_time}.npy`. Camparam + C2R saved on stop.
- Arm+hand present → CaptureSession adds a TimestampMonitor for cross-sensor sync.

## When working here
- Control flow is the teleop hand-state machine, NOT keyboard. `cs.teleop()` returns `"stop"`/`"exit"` (counters >90 frames at 10 ms). State recorded only while `save_path` is set (during record phase).
- `occulus` is a valid `--device` choice in argparse but NOT implemented in `CaptureSession` (commented out) — only `xsens` works.

## Gotchas
- Module spelled `dataset_acqusition` (missing 'i').
- Matching processor lives in `src/process/miyungpa/`.
