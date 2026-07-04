# CLAUDE.md — src/validate

## Purpose
Diagnostic smoke-tests for the distributed rig's subsystems. Each script directly
drives one `paradex/io/...` or `paradex/calibration/...` API and prints/observes the
result. They are NOT imported by anything; run them standalone.

## Subdirs (this doc pass = PART A)
- `calibration/` — re-run hand-eye solve on cached captures; measure extrinsic drift across sessions
- `camera_system/` — PySpin camera, CameraLoader, MultiCameraReader, UTGE900 signal gen, TimestampMonitor
- `command_sender/` — main-PC ↔ capture-PC command + data round-trip
- `data_sender/` — DataPublisher (capture PC) → DataCollector (main PC) pub/sub
Other subdirs (`robot/`, `robot_controller/`, `teleop/`, `upload_raw_video/`,
`visualizer/`) are a separate doc pass.

## Distributed pattern
Main-PC script calls `paradex.io.capture_pc.ssh.run_script("python src/validate/.../<client>.py")`
to launch the capture-PC counterpart over SSH, then communicates via
`CommandSender`/`CommandReceiver` and `DataCollector`/`DataPublisher`
(`paradex.io.capture_pc.command_sender` / `data_sender`).

## When working here
- Don't "fix" these to be robust; they are throwaway probes (bare `except`, hardcoded
  serials like `"22645026"`, fixed `sleep` durations are intentional).
- Confirm any API you reference still exists in the real module before documenting.

## Gotchas
- `camera_system/sync.py` is EMPTY (0 bytes).
- `signal_generator.py` and `signal_generator_debug.py` are byte-identical.
- `camera_reader.py` / `camera_reader_remote.py` carry stale `<<<<<<<`-style merge
  leftovers in comments (harmless, commented out).
- Module typo `dataset_acqusition` is intentional; do not rename.
