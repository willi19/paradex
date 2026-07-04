# Camera Daemons (Capture-PC Side)

Long-running daemons that run on each **capture PC**. They expose the local
cameras to the network so the main PC can orchestrate capture remotely, and
provide a web monitor for camera health.

## Scripts
| File | Purpose |
|------|---------|
| `server_daemon.py` | Camera command server. Wraps `CameraLoader` behind a ZMQ command server so the main PC can start/stop image/video/stream capture remotely. Must be running on every capture PC. |
| `monitor_daemon.py` | Starts a `CameraMonitor` web dashboard (port 1234) for live camera status. |
| `reset_cameras.py` | **Main-PC recovery.** Force-kills (`-9`) and relaunches the daemons on all capture PCs when cameras hang. Run from the main PC, not the capture PC. |

## Usage
`server_daemon.py` / `monitor_daemon.py` run **on the capture PCs** (not the main
PC). They are typically launched over SSH by the main-PC orchestrators
(`run_script` / `remote_camera_controller`), but can be started manually:

```bash
# On each capture PC:
python src/camera/server_daemon.py     # blocks forever, serves camera commands
python src/camera/monitor_daemon.py    # web monitor at http://<pc>:1234
```

`server_daemon.py` must be alive before any `src/capture/camera/*_remote.py`
orchestrator can connect — otherwise the controller raises `ConnectionError`
naming the unreachable PCs.

### Recovery (cameras hung / won't restart)
If cameras hang (LAN drop, frame loss) and a plain `pkill run_auto` on the main
PC leaves the capture-PC daemon stuck so the next run can't start cameras, run
**from the main PC**:

```bash
python src/camera/reset_cameras.py                  # hard-reset all capture PCs
python src/camera/reset_cameras.py --pc_list capture1 capture2
python src/camera/reset_cameras.py --no_restart     # only kill
```

## Inputs & Outputs
- `server_daemon.py`: no CLI args. Listens on ZMQ ports (ping 5480, monitor 5481,
  command 5482). Capture results are written to disk by the `CameraLoader`
  it drives (paths come from the remote `start` command).
- `monitor_daemon.py`: serves a web page; reads camera status only.
- `reset_cameras.py`: `--pc_list`, `--no_restart`. SSHes each capture PC to
  `pkill -9` the daemons, then relaunches `server_daemon.py`.

## Related
- [`paradex/io/camera_system/camera_server_daemon.py`](../../paradex/io/camera_system/camera_server_daemon.py) — the actual server implementation.
- [`paradex/io/camera_system/monitor_daemon.py`](../../paradex/io/camera_system/monitor_daemon.py) — `CameraMonitor`.
- [`paradex/io/camera_system/camera_loader.py`](../../paradex/io/camera_system/camera_loader.py) — camera control backend.
- Main-PC counterpart: [`../capture/camera`](../capture/camera) (the `*_remote.py` scripts).
