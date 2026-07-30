# Video Undistort / Upload

Distributed raw-video undistort + upload, built on the [`paradex.process`](../../../paradex/process/)
batch framework. Each capture PC undistorts its own local raw videos (NVENC) and
uploads them to NAS; the main PC SSH-launches the workers and shows live progress —
in the console, in a browser, or both.

## Scripts
| File | Purpose |
|------|---------|
| `main.py` | **Main PC.** `run_distributed(...)` — SSH-launches `worker.py` on every capture PC and shows the aggregated dashboard (console and/or web) until all finish. |
| `worker.py` | **Capture PC** (or `--local`). Discovers local raw `.avi`s and undistorts each via `undistort_raw_video`, reporting frame-level progress. |

## Usage
Run only the main-PC script — it orchestrates the workers automatically:
```bash
python src/util/upload_video/main.py              # console dashboard
python src/util/upload_video/main.py --web        # + browser page on :8080
python src/util/upload_video/main.py --quiet      # web only, silent terminal
```
This launches `worker.py` on every capture PC (each processes its own local raw
videos) and aggregates status over ZMQ (port 1234). Both views show per-PC counts,
each in-flight video's `frames @fps • ETA`, and a rig-wide ETA.

| Flag | Default | Meaning |
|------|---------|---------|
| `--web` | off | serve the browser dashboard at `http://<main-pc>:8080` |
| `--web-port` | 8080 | HTTP port for that page |
| `--quiet` | off | suppress the console dashboard (implies `--web`) |
| `--linger` | 300 | seconds to keep the web page up after all PCs finish |

The web page is plain stdlib HTTP + a 1 s poll of `/api/progress` — no Flask, no
CDN assets, so it works on the offline rig. Open it from any machine that can
reach the main PC.

Single machine (debug / one PC):
```bash
python src/util/upload_video/worker.py --local
```

## Inputs & Outputs
- `worker.py` discovers raw videos via `get_raw_videopath_list()` (local `capture_path_list`),
  undistorts each in place, rsyncs the result to NAS, and removes the local source.
- Status is the shared `paradex.process` dashboard; published items carry
  `frame`/`total`/`fps`/`eta`/`elapsed`, which both the console and the web view render.

## Inter-process flow
```
capture PC: worker.serve_jobs (undistort_raw_video) ──ZMQ status──▶ main PC: run_distributed
                                                                      ├─ console dashboard
                                                                      └─ web dashboard :8080 (--web)
```

## Related
- [`paradex/process/`](../../../paradex/process/) — the batch framework (`Job`, `serve_jobs`, `run_distributed`).
- [`paradex/video/raw_video_processor.py`](../../../paradex/video/raw_video_processor.py) — `undistort_raw_video`, `get_raw_videopath_list` (and the legacy `RawVideoProcessor`).
- [`src/process/template/`](../../process/template/) — the copy-me worker/main template.
