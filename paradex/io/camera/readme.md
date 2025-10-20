# camera_loader.py
### CameraManager

`CameraManager` is a multi-threaded wrapper around the FLIR *Spinnaker* (PySpin) SDK that lets you connect, synchronize, and capture from multiple cameras simultaneously.  
Supported capture modes are **single image**, **continuous video**, and **real-time stream**.

---

#### ✨ Key Features
- Parallel capture: one dedicated thread per camera
- Three modes: `"image"` (`.png`), `"video"` (`.avi` + timestamps), `"stream"` (in-memory frames)
- Auto-Force-IP for misconfigured GigE cameras
- User-supplied per-lens exposure, gain, and FPS (loaded from JSON)
- Thread-safe getters for live images and frame IDs
- Graceful shutdown with event-based coordination

---

Two config files must be placed under `config/camera/` (or adjust `config_dir`):

* `camera.json` – map camera serial → lens id 
* `lens_info.json` – map lens id → `{Gain, Exposure, fps}`  



---

#### 📑 API Reference

| Method | Description |
|--------|-------------|
| **`__init__(mode, path=None, serial_list=None, syncMode=True)`** | Build the manager, auto-detect cameras, spin up one thread per camera. `syncMode` is forced to *False* when `mode=="image"`. |
| **`set_save_dir(save_dir)`** | Change/define the output directory *after* construction. |
| **`get_serial_list()`** | Return a list of serial numbers of all connected cameras. |
| **`autoforce_ip()`** | On each interface with exactly one camera, forces a valid GigE IP if the current address is outside the expected subnet (not starting with `11.`). |
| **`get_videostream(savePath)`** | Create and open a `SpinVideo` object (`.avi`, 30 FPS, 2048×1536). |
| **`wait_for_connection()`** | Block until every camera thread reports success or failure. Returns **True** only if all cameras connected. |
| **`run_camera(index)`** | **Thread target.** Initializes a `Camera` object, starts capture, and keeps looping until `exit` is set. Handles all per-mode logic (video/image/stream). |
| **`wait_for_capture_end()`** | Block until every camera thread sets its `capture_end_flag` (i.e., file flush complete). |
| **`start()`** | Raise `start_capture`; all camera threads begin grabbing. |
| **`end()`** | Lower `start_capture`; running loops finish gracefully. |
| **`quit()`** | Set `exit` and join every camera thread. Call this before program termination. |
| **`get_frameid(index)`** | *(stream-only)* Thread-safe getter for the latest frame ID of camera *index*. |
| **`get_data(index)`** | *(stream-only)* Thread-safe getter → `{"image": np.ndarray, "frameid": int}` for camera *index*. |

#### Capture Modes
| Mode | Behaviour | Disk Output |
|------|-----------|-------------|
| `image` | Capture **one** frame per camera then stop | `<save_dir>/<serial>.png` |
| `video` | Continuous capture until `end()` | `<home>/capturesN/<save_dir>/<serial>.avi` + `<serial>_timestamps.json` |
| `stream` | Continuous RAM buffer; user pulls via `get_data()` | none |

---

#### ⚙️ Threading & Events

| Event Flag | Purpose |
|------------|---------|
| `start_capture` | Raised by `start()`, cleared by `end()` or after one PNG in `image` mode. |
| `exit` | Raised by `quit()` to break all loops. |
| `connect_flag[i]` | Set by thread *i* once connection attempt finished. |
| `connect_success[i]` | Set by thread *i* only on successful camera init. |
| `capture_end_flag[i]` | Cleared when a new capture session begins; set again once the thread finishes flushing data. |


                capture PC                               |           MAIN PC

camera1 --->                                             |
camera2 ---> camera loader ---> remote_camera_loader   --->  Remote_camera_controller     
camera3 --->                                             |
camera4 --->                                             |

