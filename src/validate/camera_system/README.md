# Camera System Validation

Smoke-tests for the Flir Blackfly / PySpin camera stack: single-camera control,
the multi-camera `CameraLoader`, live frame readers, the remote (capture-PC) camera
controller, and hardware sync via the UTGE900 signal generator + timestamp monitor.

## Scripts
| File | Purpose |
|------|---------|
| `pyspin_camera.py` | Lowest level: enumerate serials, `autoforce_ip`, load each camera, exercise `single` / `continuous` capture modes, release. |
| `camera.py` | Drive one `Camera("pyspin", serial)` through `image` → `full` modes repeatedly. |
| `camera_sync.py` | One `Camera` in `video` mode with `syncMode=True` (waits on hardware trigger). |
| `camera_loader.py` | Multi-cam `CameraLoader`: cycle `full` (with error polling) → `image`, twice. |
| `camera_reader.py` | `CameraLoader` in `stream` mode + `MultiCameraReader.get_images`, display merged grid with `cv2.imshow`. |
| `camera_reader_remote.py` | Bare `MultiCameraReader()` init/close (intended to run on a capture PC against running daemons). |
| `remote_camera_controller.py` | Main-PC side: `remote_camera_controller` records `video` on all capture PCs, Enter to stop. |
| `signal_generator.py` | Start UTGE900 at 30 fps for 5 s, then stop. |
| `signal_generator_debug.py` | Byte-identical duplicate of `signal_generator.py`. |
| `timestamp.py` | Run `TimestampMonitor` + UTGE900 together for 10 s to verify trigger timestamps; `q` to quit. |
| `sync.py` | **Empty file** (placeholder). |

## Usage
Most run on the **main PC**. `pyspin_camera.py` / `camera.py` / `camera_sync.py` /
`camera_reader_remote.py` are meant to run directly **on a capture PC** (they need
local PySpin / running daemons).

```bash
# Capture-PC, low level
python src/validate/camera_system/pyspin_camera.py     # enumerate + single/continuous
python src/validate/camera_system/camera.py            # image/full cycle (serial hardcoded "22645026")
python src/validate/camera_system/camera_sync.py       # syncMode video (needs signal gen running)

# Main PC
python src/validate/camera_system/camera_loader.py     # multi-cam full/image cycle
python src/validate/camera_system/camera_reader.py     # live merged stream view
python src/validate/camera_system/remote_camera_controller.py  # record video on all capture PCs (Enter=stop)

# Hardware sync (main PC, talks to signal-generator IP from network_info)
python src/validate/camera_system/signal_generator.py  # 30 fps for 5 s
python src/validate/camera_system/timestamp.py         # signal gen + timestamp monitor, 10 s ('q' quits)
```

Keyboard: `timestamp.py` uses `q` to set the end event; `remote_camera_controller.py`
waits on **Enter** at an `input()` prompt to stop.

## What it validates
- `pyspin_camera.py` / `camera.py`: each step prints success/failure per serial — a passing
  run captures images and N continuous frames without "failed" lines.
- `camera_loader.py`: no entries from `get_all_errors()`; mode switches cleanly.
- `camera_reader.py`: a live OpenCV window shows the merged camera grid with advancing `frame_id`s.
- `camera_sync.py` / `timestamp.py`: frames/timestamps only advance while the signal generator
  is running — confirms hardware trigger sync. `timestamp.py` should print monotonically
  increasing frame ids tied to the 30 fps trigger.

## Related
These directly exercise [`paradex/io/camera_system/`](../../../paradex/io/camera_system):
- `camera_loader.py` (`CameraLoader`), `camera.py` (`Camera`), `pyspin.py`
  (`get_serial_list`, `autoforce_ip`, `load_camera`)
- `camera_reader.py` (`MultiCameraReader`), `remote_camera_controller.py`,
  `signal_generator.py` (`UTGE900`), `timestamp_monitor.py` (`TimestampMonitor`)
- [`paradex/image/merge.py`](../../../paradex/image/merge.py) — `merge_image`
- `network_info` (`system/current/`) supplies signal-generator / timestamp `param`
