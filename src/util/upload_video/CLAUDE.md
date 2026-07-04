# CLAUDE.md — src/util/upload_video

## Purpose
Distributed raw-video processing pipeline with a live progress dashboard. Capture PCs process videos and publish progress; the main PC aggregates and serves a Flask + SocketIO web UI.

## Files
- `process.py` — MAIN PC entry. `kill_remote_clients()` SSHes each `get_pc_list()` PC and `pkill -f "src/util/upload_video/client.py"` to free ZMQ 1234. `VideoProgressMonitor(web_port, zmq_port)` wraps Flask + `flask_socketio.SocketIO(async_mode='threading')` + `DataCollector(port=zmq_port)`. Routes: `/` -> `video_monitor.html`, `/api/progress` -> JSON snapshot+summary. `update_loop` (daemon thread) diffs `collector.get_data()` each second and emits `progress_update`. `__main__`: kill clients, `run_script('python src/util/upload_video/client.py', log=True)`, then `VideoProgressMonitor(web_port=8081, zmq_port=1234).start()`.
- `client.py` — CAPTURE PC entry. `VideoProgressPublisher(port=1234)` = `RawVideoProcessor()` + `DataPublisher(port, name="video_processor")`. `start_processing(update_interval)` calls `processor.process()`, then loops while `not processor.finished()` sending `publisher.send_data(metadata=[...], data=[])` per video, sleeping `update_interval`; sends a final snapshot, prints `processor.log`, closes publisher.
- `templates/video_monitor.html` — SocketIO dashboard (~500 lines): summary header + per-video progress cards, listens for `initial_data` / `progress_update`.

## paradex modules used
- `paradex.video.raw_video_processor.RawVideoProcessor`
- `paradex.io.capture_pc.data_sender.DataPublisher` / `DataCollector`
- `paradex.io.capture_pc.ssh.run_script`, `ssh_port`
- `paradex.utils.system.get_pc_list`, `get_pc_ip`

## Data flow & IO
RawVideoProcessor -> DataPublisher (ZMQ 1234, capture PC) -> DataCollector (ZMQ 1234, main PC) -> Flask/SocketIO -> browser. No persistent file output; videos are processed in place by RawVideoProcessor. Metadata keys: name, status, progress, current_frame, total_frames, fps, eta, message, video_path.

## When working here
- Run `process.py` only; it launches `client.py` remotely. Run `client.py` standalone only for single-PC debugging.
- ZMQ port 1234 is shared; `kill_remote_clients` exists specifically to release it from zombie clients.

## Gotchas
- Port mismatch: `VideoProgressMonitor` class default `web_port=8080`, but `__main__` instantiates with `8081` — the live dashboard is on 8081.
- `client.py` hard-codes port 1234; `DataCollector` in `process.py` must match (it does).
- `process.py` SSH/pkill string includes the literal path `src/util/upload_video/client.py`; renaming/moving the client breaks the cleanup.
- Flask runs with `allow_unsafe_werkzeug=True` — dev server, not production-hardened.
