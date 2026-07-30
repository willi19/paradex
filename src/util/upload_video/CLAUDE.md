# CLAUDE.md — src/util/upload_video

## Purpose
Distributed raw-video undistort + upload, on the `paradex.process` framework. Each
capture PC undistorts its own local raw `.avi`s (NVENC) and uploads them to NAS; the
main PC shows a live aggregated dashboard (per-PC counts, per-video frame progress,
rig ETA) in the console and/or a browser.

## Files
- `main.py` — MAIN PC entry. argparse wrapper over `run_distributed("python src/util/upload_video/worker.py", web_port=..., console=..., linger=...)`: `--web` / `--web-port` (default 8080) / `--quiet` (web only) / `--linger` (keep the page up N s after finish).
- `worker.py` — CAPTURE PC entry (also `--local`). `discover()` = one `Job` per raw `.avi` under `capture_path_list` (id = path relative to home). `process(job, ctx)` reuses `undistort_raw_video` unchanged, forwarding its per-frame progress into `ctx.status(frame=, total=)` via the `_CtxProgress` adapter. Data is local per PC → no `shard`.

## paradex modules used
- `paradex.process` (`Job`, `run_jobs`, `serve_jobs`, `run_distributed`, `serve_progress`)
- `paradex.video.raw_video_processor` (`get_raw_videopath_list`, `undistort_raw_video`)
- `paradex.utils.path.home_path`

## Data flow & IO
`worker.discover()` finds local raw videos → `undistort_raw_video` (torch undistort +
NVENC ffmpeg) writes the undistorted `.avi`, rsyncs it to NAS, deletes the local
source. Progress (`current_frame`/`total_frames`/`status`) is adapted into
`ctx.status`; the framework derives elapsed/fps/ETA and publishes it to the main PC
over the standard `paradex.process` ZMQ channel (port 1234).

## When working here
- Run `python src/util/upload_video/main.py [--web]` on the main PC; it launches the
  workers. Single-PC debugging: `python src/util/upload_video/worker.py --local`.
- The web dashboard itself is generic and lives in `paradex/process/web.py`
  (`build_snapshot` + `ProgressWebServer`/`serve_progress`) — every `run_distributed`
  caller gets it via `web_port=`. Edit the page/JSON there, not here.
- The undistort transform itself (skip/incomplete-file handling, NVENC flags, dropped-
  frame detection, NAS upload) lives in `paradex/video/raw_video_processor.py` — edit
  there, not here. `RawVideoProcessor` (its old standalone Pool driver) is still used by
  `src/validate/upload_raw_video/`, so keep `undistort_raw_video`'s signature stable.
- Tune throughput via `num_workers` in `worker.py`'s `__main__`.

## Gotchas
- Replaced the old Flask/SocketIO web monitor (`process.py`) + `DataPublisher` client
  (`client.py`), both in `deprecated/`. The web view is back as `paradex.process.web`
  — stdlib `http.server` + 1 s `/api/progress` poll, no Flask and no CDN assets (the
  old page loaded socket.io from a CDN and was blank on the offline rig).
- `worker.py` id = raw path relative to `home_path`; it's the status key + cache subdir.
