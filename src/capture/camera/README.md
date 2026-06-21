# Camera Capture

Scripts to capture images, video, and live streams from the multi-camera rig.
Each capture mode comes in two flavors: a **local** script that drives the
cameras on the PC it runs on, and a **`_remote`** orchestrator that runs on the
**main PC** and commands the capture-PC daemons over the network.

## Scripts
| File | Runs on | Purpose |
|------|---------|---------|
| `image.py` | Capture PC (or single PC) | Capture still images locally on `c` keypress; indexes save dirs. |
| `image_remote.py` | Main PC | Orchestrates synchronized still capture across all capture PCs; saves cam params per shot. |
| `video.py` | Capture PC (or single PC) | Record video locally between `c` (start) and `s` (stop). |
| `video_remote.py` | Main PC | Orchestrates video recording across capture PCs; optionally drives the UTGE900 hardware sync generator. |
| `stream.py` | Capture PC (or single PC) | Start/stop a live camera stream locally. |
| `stream_client.py` | Capture PC | Reads cameras, downscales + JPEG-encodes frames, publishes them to the main PC. SSH-launched by `stream_remote.py`. |
| `stream_remote.py` | Main PC | Launches `stream_client.py` on capture PCs, collects frames, shows a merged live preview. |

## Usage

### Prerequisite (capture PCs)
On every capture PC, the camera command server must be running:
```bash
python src/camera/server_daemon.py
```
The `_remote.py` scripts connect to these; if a PC is unreachable they abort with
a `ConnectionError` naming it.

### Images
- **Distributed (normal):** on the **main PC**
  ```bash
  python src/capture/camera/image_remote.py --save_path <dataset>
  ```
  Keys: `c` = capture one shot, `q` = quit. Each `c` writes a timestamped folder
  and snapshots the current camera params.
- **Single PC / local:**
  ```bash
  python src/capture/camera/image.py --save_path <dataset>
  ```
  Keys: `c` = capture, `q` = quit. Auto-increments numeric index dirs.

### Video
- **Distributed (normal):** on the **main PC**
  ```bash
  python src/capture/camera/video_remote.py --save_path <dataset> [--sync_mode] [--fps 30]
  ```
  Keys: `c` = start recording, `s` = stop recording, `q` = quit. With `--sync_mode`
  the UTGE900 signal generator hardware-triggers the cameras.
- **Single PC / local:**
  ```bash
  python src/capture/camera/video.py --save_path <dataset> [--sync_mode] [--frame_rate 30]
  ```
  Keys: `c` = start, `s` = stop, `q` = quit.

### Live stream (preview only, no saving)
- **Distributed:** on the **main PC**
  ```bash
  python src/capture/camera/stream_remote.py
  ```
  Auto-SSH-launches `stream_client.py` on the capture PCs and shows a merged
  OpenCV window. Key: `q` = quit. (Do not run `stream_client.py` by hand.)
- **Single PC / local:**
  ```bash
  python src/capture/camera/stream.py [--sync_mode] [--frame_rate 30]
  ```
  Keys: `c` = start stream, `s` = stop, `q` = quit.

## Inputs & Outputs
- Output root: `~/shared_data/<save_path>/` (the `*_remote.py` scripts pass
  `shared_data/<save_path>/<timestamp>/raw` as the capture target).
  - Images/video: per-capture timestamped (or indexed) subfolders with a `raw/` dir.
  - `image_remote.py` / `video_remote.py`: also save the active camera params
    (`save_current_camparam`) alongside each capture.
- `stream_client.py` → `stream_remote.py`: 1/8-scale JPEG frames over ZMQ (port 1234);
  not persisted to disk.

## Related
- [`paradex/io/camera_system/camera_loader.py`](../../../paradex/io/camera_system/camera_loader.py) — `CameraLoader.start(mode, sync, ...)`.
- [`paradex/io/camera_system/remote_camera_controller.py`](../../../paradex/io/camera_system/remote_camera_controller.py) — main-PC ZMQ controller.
- [`paradex/io/camera_system/camera_reader.py`](../../../paradex/io/camera_system/camera_reader.py) — `MultiCameraReader` (stream client).
- [`paradex/io/capture_pc/ssh.py`](../../../paradex/io/capture_pc/ssh.py) — `run_script` (SSH launch).
- [`paradex/io/capture_pc/data_sender.py`](../../../paradex/io/capture_pc/data_sender.py) — `DataPublisher` / `DataCollector`.
- [`paradex/io/camera_system/signal_generator.py`](../../../paradex/io/camera_system/signal_generator.py) — `UTGE900` hardware sync.
- Capture-PC daemons: [`../../camera`](../../camera).
