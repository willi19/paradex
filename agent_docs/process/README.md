# agent_docs/process — agent orientation

Docs for **AI agents working on `paradex/process/`** — a small, declarative **batch-processing
framework**. You write two functions — a `discover()` that lists `Job` specs and a
`process(job, ctx)` that transforms one job — and the framework handles the boring-but-essential
scaffolding around them: skip-if-done, download inputs from NAS, run your transform, upload
outputs to NAS, report live status, and catch & record per-job errors. Read this one file; the
library is only two source modules.

**This is the reusable LIBRARY module `paradex/process/`, NOT the application scripts in
`src/process/`.** `src/process/` holds concrete post-processing *apps* (`miyungpa`,
`object_turntable`, and a `template/`). The apps named `miyungpa`/`object_turntable` are
standalone and **do not** import this library today — only `src/process/template/` (the reference
worker) uses it. This doc is about the framework, not those pipelines.

Mental model: **`discover() -> [Job]` → `Processor` runs each job in a pool → per-job:
`skip? → cache inputs (NAS→local) → your process(job, ctx) → upload outputs (local→NAS) →
status`.** Everything flows through the NAS root (`shared_dir`, i.e. `~/shared_data`): a job's
`inputs`/`outputs` are paths **relative to that root**, and the framework rsyncs them in and out
of a local cache (`~/download/process_cache/<job.id>`) so your `process` only ever touches local
files.

## File map
| File | What it is |
|------|-----------|
| `processor.py` | The core single-machine framework: `Job`, `Ctx`, `Skip`, `Processor`, `run_jobs`. Caching, skip, upload, console table, error capture. No zmq. |
| `distributed.py` | Optional multi-PC layer over `Processor`: `serve_jobs` (worker side), `run_distributed` (main-PC orchestrator), `shard` (hash-partition a shared job list). Needs zmq + capture-PC config. |
| `__init__.py` | Re-exports the public API. Imports `distributed` **lazily inside a `try/except`** so single-machine use still works when zmq / PC config is absent. |

All paths relative to [`paradex/process/`](../../paradex/process/).

---

## Who calls this
| Caller | Uses |
|--------|------|
| [`src/process/template/worker.py`](../../src/process/template/worker.py) | `Job`, `run_jobs`, `serve_jobs`, `shard` — the reference capture-PC worker (discover local jobs → process → publish status). |
| [`src/process/template/main.py`](../../src/process/template/main.py) | `run_distributed` — the reference main-PC orchestrator that launches the worker on every PC. |

That is the **only** consumer in the repo. The older concrete pipelines
[`src/process/miyungpa/`](../../src/process/miyungpa) and
[`src/process/object_turntable/`](../../src/process/object_turntable) roll their own processing
and do **not** use this framework — do not assume changes here affect them.

---

## The core framework (`processor.py`)

### `Job` — one unit of work (dataclass)
| Field | Meaning |
|-------|---------|
| `id` | Unique id; used as the status key **and** the per-job cache subdir, so keep it filesystem-safe (slashes nest dirs, that's fine). Must be unique across PCs when distributed. |
| `inputs` | `{name: nas_relpath}` — pulled NAS→local cache before `process`. Reach via `ctx.input(name)`. |
| `outputs` | `{name: nas_relpath}` — files you write locally (path from `ctx.output(name)`); uploaded NAS after `process` returns. |
| `meta` | Arbitrary extra data for your `process` fn (`job.meta[...]`). |
| `done` | Optional `(job) -> bool` completion predicate. When set it is **authoritative** for the skip check; else the default skips only when *all* declared `outputs` already exist on NAS. Must be picklable when `num_workers > 1`. |

### `Ctx` — the handle passed to `process(job, ctx)`
Gives the transform exactly three things and nothing else:
- `ctx.input(name)` → local path of a cached (already-downloaded) input.
- `ctx.output(name)` → local path to **write** output `name` (parent dir pre-created; lives under the cache dir's `out/`). Whatever you write there is uploaded to `job.outputs[name]`.
- `ctx.status(message="", progress=None)` → report progress; `progress` is `0..1`, clamped and stored as `0..100` percent.

### `Skip` — raise to bail out cleanly
`raise Skip("reason")` inside `process` marks the job `skipped` (not `failed`).

### `Processor` / `run_jobs` — the runner
```python
from paradex.process import Job, run_jobs, Skip

def process(job, ctx):
    src = ctx.input("video")     # already downloaded from NAS
    dst = ctx.output("video")    # write here; uploaded automatically
    ...
    ctx.status("halfway", progress=0.5)

run_jobs(discover(), process, num_workers=4)   # -> final progress dict
```
`Processor(jobs, process_fn, num_workers=4, keep_cache=False, on_update=None, quiet=False)`.
Key behaviours verified in code:
- **Status lifecycle:** `pending → caching → processing → uploading → completed`, plus `skipped`
  and `failed` as terminal states. Held in a `Manager().dict()` so pool workers can update it.
- **`num_workers == 1` runs in-process (no `Pool`)** — the recommended way to debug (real
  tracebacks, no pickling). Workers are capped at `cpu_count()`.
- **Per-job error capture:** `_run_job` never raises; any exception is recorded as `failed` with
  the full traceback stored in the progress entry, and the batch keeps going.
- **Skip check (`_is_done`) runs *before* caching** — a job whose outputs already exist on NAS is
  skipped without downloading anything.
- **Output contract:** after `process` returns, every declared output must exist locally or the
  job `failed`s with "declared but not written".
- **`on_update(progress)`** hook is called every `update_interval` (default 1 s) with the full
  progress dict — this is the extension point the distributed layer and the web dashboard hook
  into. `quiet=True` suppresses the built-in console table.
- Queries: `get_progress()`, `summary()` (counts per status), `failures()` (id→entry for failed).

---

## The distributed layer (`distributed.py`)

Thin wrapper: same `Job` / `process` you already wrote, now spread across capture PCs. Core
`Processor` is unchanged and stays usable single-machine.

- **`serve_jobs(jobs, process_fn, ..., port=1234)`** — worker side, drop-in replacement for
  `run_jobs` on a capture PC. Builds a `Processor` with `quiet=True` and wires its `on_update`
  to a `DataPublisher` (ZMQ PUB) so the main PC sees live status. Each PC publishes one item per
  job (keyed by `job.id`) **plus** a summary sentinel `_pc::{pc}` carrying its counts and a
  `finished` flag.
- **`run_distributed(worker_cmd, pc_list=None, port=1234, ...)`** — main-PC orchestrator. Starts
  a `DataCollector`, then `run_script`s `worker_cmd` on every PC (the command must call
  `serve_jobs`), prints an aggregated dashboard, and returns when every PC's sentinel reports
  `finished` (or `timeout`).
- **`shard(jobs, pc_list=None, pc=None)`** — partition **one shared** job list across PCs by
  `zlib.crc32(job.id) % n` (stable across processes, unlike `hash()`). Only needed when every PC
  discovers the *same* list (e.g. all reading shared NAS); if each PC discovers its own local
  files, the data is already partitioned and you skip `shard`.

---

## Gotchas for editors
- **Library vs. apps:** `paradex/process/` (this framework) and `src/process/*` (concrete apps)
  are different things. Only `src/process/template/` consumes this library; `miyungpa` and
  `object_turntable` do not. Don't refactor this expecting to touch those pipelines.
- **Everything is a move through NAS.** `inputs`/`outputs` are paths **relative to `shared_dir`**
  (`~/shared_data`), rsynced via `rsync_copy` into/out of `~/download/process_cache/<job.id>`.
  Absolute paths, or forgetting an input already lives on NAS, break the caching step
  (`_download` raises `FileNotFoundError: NAS input missing`).
- **`job.id` is doubly load-bearing:** it's the status key *and* the cache subdir *and* (in
  distributed mode) must be globally unique across PCs and the CRC shard key. A non-filesystem-safe
  or colliding id corrupts caching or status silently.
- **The default skip only checks output-file existence on NAS** — a truncated/partial prior upload
  still counts as "done" and gets skipped. Use `job.done` for real completion when file existence
  isn't proof.
- **Cache-size guard is `os.path.getsize` equality only.** `_download` re-uses a cached file when
  local and NAS sizes match; same-size-but-different content would not re-download.
- **`process_fn` (and `job.done`) must be top-level picklable functions** when `num_workers > 1`
  (they cross the `multiprocessing.Pool`). Lambdas/closures only work at `num_workers=1`.
- **`keep_cache=False` (default) deletes the whole per-job cache dir in a `finally`** — including
  on failure. If you need the partial outputs of a failed job to debug, pass `keep_cache=True`.
- **`distributed` import is deliberately swallowed** by a bare `except Exception` in `__init__.py`
  — if you add an import error inside `distributed.py`, `serve_jobs`/`run_distributed`/`shard`
  simply vanish from `paradex.process` instead of raising. Check `__all__` if they go missing.
- **ZMQ PUB drops messages with no subscriber:** `run_distributed` sleeps 0.3 s after starting the
  collector before launching workers, and `serve_jobs` re-flushes final status + sleeps 0.3 s so
  the terminal `finished` state isn't lost. Preserve that ordering if you edit either side.
