# Video Undistort / Upload

Distributed raw-video undistort + upload, built on the [`paradex.process`](../../../paradex/process/)
batch framework. Each capture PC undistorts its own local raw videos (NVENC) and
uploads them to NAS; the main PC SSH-launches the workers and shows a live console
dashboard with per-video frame progress and a rig-wide ETA.

## Scripts
| File | Purpose |
|------|---------|
| `main.py` | **Main PC.** `run_distributed(...)` — SSH-launches `worker.py` on every capture PC and prints the aggregated dashboard until all finish. |
| `worker.py` | **Capture PC** (or `--local`). Discovers local raw `.avi`s and undistorts each via `undistort_raw_video`, reporting frame-level progress. |

## Usage
Run only the main-PC script — it orchestrates the workers automatically:
```bash
python src/util/upload_video/main.py
```
This launches `worker.py` on every capture PC (each processes its own local raw
videos) and aggregates status over ZMQ (port 1234). The dashboard prints per-PC
counts, each in-flight video's `frames @fps • ETA`, and a rig-wide ETA.

Single machine (debug / one PC):
```bash
python src/util/upload_video/worker.py --local
```

## Inputs & Outputs
- `worker.py` discovers raw videos via `get_raw_videopath_list()` (local `capture_path_list`),
  undistorts each in place, rsyncs the result to NAS, and removes the local source.
- No web server — status is the shared `paradex.process` dashboard. The published
  status items still carry `frame`/`total`/`fps`/`eta`/`elapsed`, so a web UI can be
  rebuilt on them if desired.

## Inter-process flow
```
capture PC: worker.serve_jobs (undistort_raw_video) ──ZMQ status──▶ main PC: run_distributed dashboard
```

## Related
- [`paradex/process/`](../../../paradex/process/) — the batch framework (`Job`, `serve_jobs`, `run_distributed`).
- [`paradex/video/raw_video_processor.py`](../../../paradex/video/raw_video_processor.py) — `undistort_raw_video`, `get_raw_videopath_list` (and the legacy `RawVideoProcessor`).
- [`src/process/template/`](../../process/template/) — the copy-me worker/main template.
