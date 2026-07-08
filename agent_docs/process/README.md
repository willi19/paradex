# agent_docs/process — agent orientation

Docs for **AI agents working on `paradex/process/`** — a small, declarative **batch-processing
framework**. You write two functions — a `discover()` that lists `Job` specs and a
`process(job, ctx)` that transforms one job — and the framework handles the boring-but-essential
scaffolding: skip-if-done, download inputs from NAS, run your transform, upload outputs to NAS,
report live status + ETA, and catch & record per-job errors. The library is only two source
modules; read the **one** file for your task.

Mental model: **`discover() -> [Job]` → `Processor` runs each job in a pool → per-job:
`skip? → cache inputs (NAS→local) → your process(job, ctx) → upload outputs (local→NAS) →
status`.** Everything flows through the NAS root (`shared_dir`, i.e. `~/shared_data`): a job's
`inputs`/`outputs` are paths **relative to that root**, and the framework rsyncs them in and out
of a local cache (`~/download/process_cache/<job.id>`), so your `process` only ever touches local
files.

**This is the reusable LIBRARY module `paradex/process/`, NOT the application scripts in
`src/process/` or `src/util/upload_video/`.** Those hold concrete post-processing *apps*
(`miyungpa`, `object_turntable`, `upload_video`, and a `template/`) that are all built **on** this
library — each defines a `discover()` + `process(job, ctx)` and calls `run_jobs`/`serve_jobs`.
[`src/process/template/`](../../src/process/template/) is the copy-me reference worker. This doc is
about the framework, not those pipelines.

| Your task | Read |
|-----------|------|
| **Write** a pipeline: `discover()` + `process(job, ctx)`, run single vs distributed | [`usage.md`](usage.md) |
| **Edit** the framework: job lifecycle, skip/cache/upload, distributed dispatch | [`internals.md`](internals.md) |
| Copy-me worked example (a real worker + orchestrator) | [`src/process/template/`](../../src/process/template/) |

Rule of thumb: **using** the framework to build a pipeline → `usage.md`; **changing** how the
framework itself works → `internals.md`.

## File map
| File | What it is |
|------|-----------|
| [`processor.py`](../../paradex/process/processor.py) | Core single-machine framework: `Job`, `Ctx`, `Skip`, `Processor`, `run_jobs`. Caching, skip, upload, console table, error capture, and derived timing/ETA (`fmt_dur`/`fmt_job`/`batch_eta`). No zmq. |
| [`distributed.py`](../../paradex/process/distributed.py) | Optional multi-PC layer over `Processor`: `serve_jobs` (worker), `run_distributed` (main-PC orchestrator), `shard` (hash-partition a shared job list). Needs zmq + capture-PC config. |
| [`__init__.py`](../../paradex/process/__init__.py) | Re-exports the public API. Imports `distributed` **lazily inside a `try/except`** so single-machine use still works when zmq / PC config is absent. |

All paths relative to [`paradex/process/`](../../paradex/process/).

## Who calls this
| Caller | Uses |
|--------|------|
| [`src/process/template/`](../../src/process/template/) | `Job`, `run_jobs`, `serve_jobs`, `shard`, `run_distributed` — reference worker + orchestrator. |
| [`src/process/miyungpa/`](../../src/process/miyungpa) | `worker.py` (`serve_jobs`/`shard`) + `main.py` (`run_distributed`) — robot-demo overlay; "local IO in meta" pattern (`Job(done=)`, no framework inputs/outputs, does its own rsync). |
| [`src/util/upload_video/`](../../src/util/upload_video) | `worker.py` (`serve_jobs`, **no** `shard` — local-per-PC data) + `main.py` (`run_distributed`) — raw-video undistort, reuses `undistort_raw_video` via a `_CtxProgress` adapter. |
| [`src/process/object_turntable/`](../../src/process/object_turntable) | `worker.py` (`run_jobs`, `num_workers=1`) — 5-stage scan reconstruction, one Job per scan. |

All four are **apps built on this framework** — `template/` is the copy-me starting point, the
others are live pipelines. Changes here can affect all of them.

Note the intentional repo-wide typo `dataset_acqusition` (missing 'i') — don't "fix" imports.
