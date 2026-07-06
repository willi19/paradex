# Camera System — API

Method reference for the camera stack. For the architecture and how these fit
together, see the {doc}`overview <camera_system>`. Signatures are verified against
the code; internal/private methods are omitted. Click a class to expand.

:::{dropdown} `start()` — parameters
:open:

Same signature on `Camera`, `CameraLoader`, and `remote_camera_controller`.

```{py:function} start(mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None)
:no-index:

:param mode: ``image`` / ``video`` / ``stream`` / ``full``.
:param syncMode: Use the hardware trigger (wait for pulses) instead of free-running.
:param save_path: Relative output dir for ``image`` / ``video``; ``None`` for ``stream``.
:param fps: Frame rate when free-running (default ``30``).
:param exposure_time: Microseconds; ``None`` → the per-camera ``camera.json`` value.
:param gain: dB; ``None`` → the per-camera ``camera.json`` value.
```
:::

:::{dropdown} `remote_camera_controller` — main PC

Drives every capture PC over ZMQ. `start()`/`stop()` set events; the background
`run()` loop does the actual send.

```{py:function} remote_camera_controller(name, pc_list=None, auto_reload=False, stall_timeout=3.0)
:no-index:

Construct; spawn the ``run()`` thread. ``pc_list=None`` → all capture PCs.
```

```{py:function} start(mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None)
:no-index:

Begin capture on all PCs. Raises the init error if the daemon connection failed.

:rtype: None
```

```{py:function} stop()
:no-index:

Stop capture on all PCs.

:rtype: None
```

```{py:function} end()
:no-index:

Release the lock, join the loop thread.

:rtype: None
```

```{py:function} reload_cameras()
:no-index:

Ask every daemon to re-init its cameras.

:rtype: None
```

```{py:function} force_takeover()
:no-index:

Grab the lock even if another controller holds it.

:returns: per-PC result
:rtype: dict
```

```{py:function} is_error()
:no-index:

:returns: ``True`` if any camera reported an error.
:rtype: bool
```

```{py:function} get_status()
:no-index:

Live health snapshot.

:returns: error flag, stalled serials, per-PC status, last raw responses
:rtype: dict
```
:::

:::{dropdown} `CameraLoader` — capture PC

Owns the local cameras and drives them together.

```{py:function} CameraLoader(types=["pyspin"])
:no-index:

Detect and open every local camera.
```

```{py:function} start(mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None)
:no-index:

Start all cameras concurrently (blocks until all started).

:rtype: None
```

```{py:function} stop()
:no-index:

Stop all cameras.

:rtype: None
```

```{py:function} end()
:no-index:

Stop + release all cameras (DeInit + free SHM).

:rtype: None
```

```{py:function} get_status_list()
:no-index:

Per-camera status dict (see ``Camera.get_status``).

:rtype: list[dict]
```

```{py:function} get_summary()
:no-index:

Compact daemon payload: counts, serials, per-camera states / frame ids / errors.

:rtype: dict
```

```{py:function} get_all_errors()
:no-index:

``{serial: (msg, traceback)}`` for cameras in error.

:rtype: dict[str, tuple]
```

**Attributes** — `camera_names` (`list[str]` serials), `cameralist` (`list[Camera]`).
:::

:::{dropdown} `Camera` — capture PC

One camera + its capture thread. See the {doc}`lifecycle contract <camera_system>`.

```{py:function} Camera(cam_type, name, frame_shape=(1536, 2048, 3), cfg=None)
:no-index:

Open the camera, allocate SHM, spawn the capture thread. ``cfg`` = the per-serial ``camera.json`` entry.
```

```{py:function} start(mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None)
:no-index:

Begin capture in ``mode``; blocks until acquiring.

:rtype: None
```

```{py:function} stop(timeout=5.0)
:no-index:

Stop capture; finite wait (warns and returns on timeout).

:rtype: None
```

```{py:function} end(timeout=5.0)
:no-index:

Stop + release (DeInit + free SHM); finite wait.

:rtype: None
```

```{py:function} get_status()
:no-index:

``{state, frame_id, name, mode, fps, syncMode, save_path, time}``.

:rtype: dict
```

```{py:function} get_state()
:no-index:

One of ``CONNECTING`` / ``READY`` / ``STARTING`` / ``CAPTURING`` / ``ERROR`` / ``STOPPED``.

:rtype: str
```

```{py:function} get_frame_id()
:no-index:

Last processed frame id.

:rtype: int
```

```{py:function} get_error()
:no-index:

``(has_error, (message, traceback))``.

:rtype: (bool, (str, str))
```
:::

:::{dropdown} `PyspinCamera` — driver

Thin wrapper over the PySpin SDK. `Camera` holds one as `self.camera`.

```{py:function} get_image()
:no-index:

Grab one frame. ``(None, None)`` on grab timeout (``GRAB_TIMEOUT_MS``); ``(None, frame_data)`` on incomplete/zero-size. ``dict = {pc_time, frameID}``.

:rtype: (ndarray, dict) | (None, dict) | (None, None)
```

```{py:function} start(mode, syncMode, frame_rate=None, gain=None, exposure_time=None)
:no-index:

Re-apply changed config + ``BeginAcquisition``. ``mode`` = ``"single"`` | ``"continuous"``.

:rtype: None
```

```{py:function} stop()
:no-index:

``EndAcquisition`` (camera stays connected).

:rtype: None
```

```{py:function} release()
:no-index:

``DeInit`` (disconnect).

:rtype: None
```

**Module-level** — `get_serial_list()` → `list[str]`, `autoforce_ip()`,
`load_camera(serial, cfg=None)` → `PyspinCamera`,
`load_timestamp_monitor(serial, cfg=None)` → `PyspinTimestampMonitor`; constant
`GRAB_TIMEOUT_MS = 1000`.
:::

:::{dropdown} Implementation notes — the Event handshake

For readers going into `camera.py`: the capture thread and the caller synchronize
through an **Event handshake**, not plain flags.

```python
# run(): the capture thread
while not self.event["exit"].is_set():
    if self.event["start"].is_set():
        self.continuous_acquire() if self.mode in ["full","video","stream"] else self.single_acquire()
    time.sleep(0.001)

# start(): set the event, then wait until the thread actually begins
self.event["start"].set()
self.event["acquisition"].wait()

# continuous_acquire(): the body — grab, route to sinks, exit on stop
self.camera.start("continuous", self.syncMode, self.fps, ...)   # BeginAcquisition
self.event["acquisition"].set()
while self.event["start"].is_set() and not self.event["exit"].is_set():
    frame, frame_data = self.camera.get_image()
    if frame is None: continue          # timeout → re-check the while condition
    if save_video: video_writer.write(frame)
    if stream:     ...                  # SHM double-buffer (write_flag toggles)
self.camera.stop(); self.event["stop"].set()
```

`stop()` clears `event["start"]` and waits on `event["stop"]`; the loop exits on its
next iteration and sets `event["stop"]`. The P4 hang lived exactly here — a blocking
`get_image()` meant the loop never iterated, so `event["stop"]` was never set and
`stop()` never returned.
:::
