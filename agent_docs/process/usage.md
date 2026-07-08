# paradex.process — How to Write a Pipeline (for humans & agents)

Read this before writing a batch job. You write **two functions**; the framework does the rest
(skip-if-done, NAS download, upload, status, error capture, parallelism, distribution).

> **Editing the framework itself** (job lifecycle / cache / upload / distributed dispatch), not
> just using it? Read [`internals.md`](internals.md) instead.

Fastest start: **copy [`src/process/template/`](../../src/process/template/)** — `worker.py`
(discover + process) and `main.py` (orchestrator) are a working, runnable example.

## TL;DR

```python
from paradex.process import Job, run_jobs, Skip

def discover():                                  # what work exists -> list[Job]
    return [
        Job(id=f"{sess}/{serial}",
            inputs={"video": f"{sess}/raw/videos/{serial}.avi"},   # NAS-relative
            outputs={"video": f"{sess}/videos/{serial}.avi"},      # NAS-relative
            meta={"serial": serial})
        for sess, serial in find_work()
    ]

def process(job, ctx):                           # how to do ONE unit
    src = ctx.input("video")      # local cached path (already downloaded from NAS)
    dst = ctx.output("video")     # local path to write; uploaded to NAS automatically
    n = frame_count(src)
    for i, frame in enumerate(read(src)):
        ctx.status(frame=i, total=n)             # frame-rate ETA on the dashboard
        write(dst, transform(frame))

run_jobs(discover(), process, num_workers=4)     # -> final progress dict
```

## The mental model: everything is a move through NAS

`inputs`/`outputs` are paths **relative to `shared_dir`** (`~/shared_data`, the NAS). Per job the
framework rsyncs inputs NAS→local into `~/download/process_cache/<job.id>/in/`, runs your
`process` (which only touches **local** paths via `ctx`), then rsyncs everything you wrote under
`out/` back to NAS. Your transform never opens a NAS path directly — always go through `ctx`.

## `Job` — declare one unit of work

| Field | Meaning |
|-------|---------|
| `id` | Unique id; the status key **and** the per-job cache subdir. Keep it filesystem-safe (slashes nest dirs, fine). Must be unique **across PCs** when distributed. |
| `inputs` | `{name: nas_relpath}` — pulled NAS→local before `process`. Reach via `ctx.input(name)`. |
| `outputs` | `{name: nas_relpath}` — files you write locally (path from `ctx.output(name)`); uploaded to NAS after `process` returns. |
| `meta` | Arbitrary extra data for your `process` (`job.meta[...]`). Use for the pattern-B local-path trick below. |
| `done` | Optional `(job) -> bool` completion predicate; **authoritative** for the skip check when set (see skip section). Must be picklable when `num_workers > 1`. |

## `ctx` — the handle passed to `process(job, ctx)`

Gives your transform exactly three things and nothing else:

- `ctx.input(name)` → local path of a cached (already-downloaded) input.
- `ctx.output(name)` → local path to **write** output `name` (parent dir pre-created; lives under
  the cache dir's `out/`). Whatever you write there is uploaded to `job.outputs[name]`.
- `ctx.status(message="", progress=None, frame=None, total=None)` → report progress. Pass a
  `progress` fraction `0..1` (clamped, stored as `0..100` percent), **or** pass `frame`/`total`
  (current item / total items) and progress is derived as `frame/total`. With `frame`/`total` the
  dashboard shows a "340/1200 @30fps" readout and a frame-rate ETA (more accurate than a bare
  percentage). **You only report where you are** — `elapsed`/`fps`/`eta` are computed for you from
  the job's processing start. Call it periodically inside frame loops for a live dashboard.

You also get `job` itself (`ctx.job`), so `job.meta[...]` is available in `process`.

## Two data-layout patterns (pick one per `discover`)

- **A — inputs live on NAS:** declare `inputs`, the framework downloads them. Every PC sees the
  same NAS, so you must **`shard`** the list across PCs (see distributed run below). This is the
  template's default.
- **B — inputs are local to each capture PC:** put the local path in `meta`, read it directly in
  `process` (not via `ctx.input`), declare only `outputs`. No `shard` — the data is already
  partitioned by machine.

## Skip-if-done (idempotency — reruns are cheap)

Checked **before** any download, so an already-done job costs nothing:

- **Default** (no `job.done`): skip only when **all** declared `outputs` already exist on NAS.
  A job with **no** declared outputs is **never** skipped (doneness can't be inferred → always run).
- **`Job(done=fn)`**: your predicate is authoritative — use it when file-existence isn't proof
  (frame-count match, multi-file manifest, external quality gate). Exceptions from it propagate
  and fail the job (not silently skip).
- Inside `process`, `raise Skip("reason")` bails out cleanly → job marked `skipped`, not `failed`.

## Running it

**Single machine** (`run_jobs` / `Processor`):

```python
run_jobs(jobs, process, num_workers=4)   # process pool
run_jobs(jobs, process, num_workers=1)   # in-process — best for DEBUGGING
```

`num_workers=1` runs in-process (no `Pool`): real tracebacks, no pickling — always debug here
first. `num_workers>1` uses `multiprocessing.Pool`, so **`process` (and `job.done`) must be
top-level picklable functions** — lambdas/closures only work at `num_workers=1`. Workers are
capped at `cpu_count()`.

**Distributed** (spread across capture PCs):

```python
# worker.py, runs on each capture PC:
from paradex.process import serve_jobs, shard
serve_jobs(shard(discover()), process, num_workers=4)   # pattern A: shard shared list
# serve_jobs(discover(), process)                        # pattern B: already partitioned

# main.py, runs on the MAIN PC:
from paradex.process import run_distributed
run_distributed("python src/process/template/worker.py")   # SSH-launches workers, live dashboard
```

`serve_jobs` is a drop-in for `run_jobs` that also publishes live status to the main PC over ZMQ.
`run_distributed` SSH-launches `worker_cmd` on every capture PC (`pc_list=` to subset, `timeout=`
to bound), aggregates a dashboard, and returns when every PC reports finished. Push the latest
code to workers first (`from paradex.io.capture_pc.ssh import git_pull; git_pull("branch")`).

## Monitoring

Both runners return the final progress dict; also `Processor.summary()` (counts per status) and
`Processor.failures()` (`id→entry` for failed, with `traceback`). For a custom dashboard/log pass
`on_update=fn` (called each refresh with the full progress dict) and `quiet=True` to suppress the
built-in console table — this is exactly how the distributed layer feeds the ZMQ dashboard.

Every progress entry carries **derived timing** the framework stamps once a job is `processing`:
`elapsed` (seconds), `eta` (seconds remaining — frame-rate-based when you reported `frame`/`total`,
else percentage-based; `0.0` when done, `None` when not estimable), plus `fps`/`frame`/`total` when
you reported them. The console table prints a whole-batch ETA (`batch_eta`) and per-job `fmt_job`
lines; the distributed dashboard reuses the same precomputed numbers.

## Gotchas

- **Never open a NAS path in `process`.** Use `ctx.input`/`ctx.output`; inputs not on NAS raise
  `FileNotFoundError: NAS input missing` during caching.
- **`job.id` is triply load-bearing:** status key + cache subdir + (distributed) globally-unique
  CRC shard key. A colliding or non-filesystem-safe id corrupts caching/status silently.
- **Default skip trusts file existence only** — a truncated prior upload still counts as "done".
  Use `Job(done=fn)` when that matters.
- **Declared outputs must be written.** If `process` returns without writing a declared output,
  the job `failed`s ("declared but not written"). Declare only what you actually produce.
- **`num_workers>1` needs picklable `process`/`done`** — top-level functions, no closures.
- **Cache is deleted on success *and* failure** (`keep_cache=False` default). To inspect a failed
  job's partial outputs, pass `keep_cache=True`.
- **Debug at `num_workers=1`** — the pool swallows nothing (errors are captured with tracebacks),
  but in-process gives you a live traceback and no pickling constraints.
- Intentional repo typo `dataset_acqusition` (missing 'i') — don't "fix" it.
