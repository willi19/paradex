# Video Upload / Processing Monitor

Distributed raw-video processing with a live web dashboard. Each capture PC processes its raw videos and publishes progress over ZMQ; the main PC collects all streams and serves a Flask + SocketIO web monitor.

## Scripts
| File | Purpose |
|------|---------|
| `process.py` | **Main PC.** Kills stale remote clients, launches `client.py` on every capture PC over SSH, collects progress (ZMQ) and serves a web dashboard. |
| `client.py` | **Capture PC.** Runs `RawVideoProcessor`, publishes per-video progress to the main PC over ZMQ port 1234. |
| `templates/video_monitor.html` | Dashboard UI (SocketIO client, per-video progress cards + summary). |

## Usage
Run only the main-PC script — it orchestrates the clients automatically:
```bash
python src/util/upload_video/process.py
```
This:
1. SSHes to every PC from `get_pc_list()` and `pkill`s leftover `client.py` (frees ZMQ 1234).
2. `run_script('python src/util/upload_video/client.py', log=True)` on the capture PCs.
3. Starts a `DataCollector` on ZMQ 1234 and a Flask/SocketIO server.

Open the dashboard at **http://localhost:8081** (note: `__main__` passes `web_port=8081`, though the class default is 8080). The page auto-updates via SocketIO; `/api/progress` returns the current JSON snapshot.

To run a client manually on a capture PC (debug):
```bash
python src/util/upload_video/client.py
```

## Inputs & Outputs
- `client.py`: reads raw videos discovered by `RawVideoProcessor` (`videopath_list`); processes them in place; emits metadata `{name, status, progress, current_frame, total_frames, fps, eta, message, video_path}`. No web/file output of its own.
- `process.py`: no file output — serves HTTP/WebSocket. Aggregates per-PC progress and computes summary stats (total / completed / processing / failed / avg_progress).

## Inter-process flow
```
capture PC: RawVideoProcessor -> DataPublisher(port=1234) ──ZMQ──▶ main PC: DataCollector(port=1234) -> Flask/SocketIO ──▶ browser
```

## Related
- [`paradex/video/raw_video_processor.py`](../../../paradex/video/raw_video_processor.py) — `RawVideoProcessor` (`process`, `finished`, `get_progress`, `log`).
- [`paradex/io/capture_pc/data_sender.py`](../../../paradex/io/capture_pc/data_sender.py) — `DataPublisher`, `DataCollector`.
- [`paradex/io/capture_pc/ssh.py`](../../../paradex/io/capture_pc/ssh.py) — `run_script`, `ssh_port`.
- [`paradex/utils/system.py`](../../../paradex/utils/system.py) — `get_pc_list`, `get_pc_ip`.
- Sibling capture apps in [`src/capture/`](../../capture).
