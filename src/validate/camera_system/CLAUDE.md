# CLAUDE.md — src/validate/camera_system

## Purpose
Standalone probes for `paradex.io.camera_system`. Each mirrors one layer of the
camera stack so you can confirm hardware/daemons/sync work before a real capture.

## Files
- `pyspin_camera.py` — `get_serial_list` → `autoforce_ip` → `load_camera(serial)`;
  per camera exercises `start("single"|"continuous", ...)` + `get_image()` + `release()`.
  Everything wrapped in bare `try/except` printing pass/fail.
- `camera.py` — `Camera("pyspin", serial)` looped `start("image")`/`start("full", fps=30)`/`stop`/`end`.
- `camera_sync.py` — same `Camera` but `start("video", True, "test.avi", fps=30)` (syncMode=True), 5 s.
- `camera_loader.py` — `CameraLoader().start("full"|"image", False, save_path, fps=30)`,
  polls `get_all_errors()` (dict serial→(err, traceback)), `stop`/`end`.
- `camera_reader.py` — `CameraLoader.start("stream", fps=10)` + `MultiCameraReader(cl.camera_names)`;
  `get_images(copy=True)` → `{name: (img, frame_id)}`, `merge_image`, `cv2.imshow`.
- `camera_reader_remote.py` — instantiates bare `MultiCameraReader()` then closes; meant to run
  on a capture PC against already-running camera daemons. Body is mostly commented out.
- `remote_camera_controller.py` — main-PC: `remote_camera_controller("test").start("video", True, name, fps=30)`,
  `input()` to block, `stop()`/`end()`.
- `signal_generator.py` / `signal_generator_debug.py` — IDENTICAL: `UTGE900(**network_info["signal_generator"]["param"]).start(fps=30)`, 5 s, stop/end.
- `timestamp.py` — `TimestampMonitor(**network_info["timestamp"]["param"]).start("tmp")` + `UTGE900.start(fps=30)`,
  `listen_keyboard({"q": end_event})`, 10 s, stop/end.
- `sync.py` — EMPTY (0 bytes).

## paradex modules used
`paradex.io.camera_system`: `camera_loader.CameraLoader`, `camera.Camera`,
`camera_reader.MultiCameraReader`, `remote_camera_controller.remote_camera_controller`,
`signal_generator.UTGE900`, `timestamp_monitor.TimestampMonitor`,
`pyspin.{get_serial_list,autoforce_ip,load_camera}`.
`paradex.image.merge.merge_image`; `paradex.utils.system.network_info`;
`paradex.utils.keyboard_listener.listen_keyboard`.

## Data flow & IO
- `network_info` (from `system/current/`) provides `signal_generator`/`timestamp` `param` dicts (IP/port).
- Camera modes seen: `single`, `continuous` (PySpin level); `image`, `full`, `video`, `stream`
  (Camera/CameraLoader level). `MultiCameraReader.get_images` returns `(img, frame_id)` tuples.
- Outputs are throwaway: `tmp.png`, `test*.avi`, `test_camloader`, `tmp` timestamp log.

## When working here
- Hardcoded serial `"22645026"` in `camera.py`/`camera_sync.py`; swap to `get_serial_list()` if testing real rig.
- syncMode/video modes require the UTGE900 trigger running — start `signal_generator.py` first.

## Gotchas
- `sync.py` is empty.
- `signal_generator.py` == `signal_generator_debug.py` byte-for-byte.
- `camera_reader.py` / `camera_reader_remote.py` contain stale git-merge leftover lines in
  trailing comments (`>>>>>>> <hash>`); harmless.
- `camera_reader.py` swallows the `MultiCameraReader` construction error then keeps using `cr` (would NameError);
  it's a probe, not robust code.
