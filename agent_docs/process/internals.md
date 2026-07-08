# paradex.process — Internals (for agents editing this module)

**You are here because you are changing the framework itself** — job lifecycle, skip/cache/upload
mechanics, the distributed dispatch, the threading/process model. If you only want to *write* a
pipeline, read [`usage.md`](usage.md) instead.

The library is two files. [`processor.py`](../../paradex/process/processor.py) is the whole
single-machine engine (no zmq). [`distributed.py`](../../paradex/process/distributed.py) is a thin
layer that reuses it. [`__init__.py`](../../paradex/process/__init__.py) re-exports the API.

---

## 1. The job lifecycle (`_run_job` in `processor.py`)

Every job runs through one function, `_run_job(job, process_fn, progress_dict, keep_cache)`, which
**never raises** — it captures all outcomes into the shared `progress_dict`. Status lifecycle:

```
pending -> [skipped]                                   # _is_done() true, before any download
        -> caching -> processing -> uploading -> completed
        -> failed   (from ANY point, with traceback recorded)
        -> skipped  (process raised Skip)
```

`STATUSES` / `DONE_STATUSES = ("completed","skipped","failed")` are module constants; the
distributed layer imports `DONE_STATUSES` to decide when a PC is finished. The steps:

1. **Skip check** — `_is_done(job)` runs **before** caching. True → mark `skipped` (progress 100),
   return. No download happens for done jobs.
2. **Cache inputs** — for each `job.inputs[name] = rel`, `_download(rel, cache_dir/in/basename)`
   pulls NAS→local; `ctx._inputs[name]` gets the local path.
3. **Process** — mark `processing` **stamping `started_at = time.time()`**, then
   `process_fn(job, ctx)`. Your code only touches local paths via `ctx`. That `started_at` stamp is
   what lets `_update` derive `elapsed`/`fps`/`eta` on every subsequent entry (see §4).
4. **Upload outputs** — for each `job.outputs[name] = rel`: the declared local `ctx.output(name)`
   **must exist** (else `FileNotFoundError: declared but not written` → `failed`), then
   `rsync_copy(local, shared_dir/rel)`.
5. **`finally`** — unless `keep_cache`, `_rmtree_quiet(cache_dir)` deletes the whole per-job cache
   dir — **on success and on failure**. Keep this in `finally` or a crash leaks cache dirs.

`Ctx.output(name)` writes under `cache_dir/out/<basename(rel)>` (parent pre-created). Note it
takes only the **basename** of the NAS relpath, so two outputs whose relpaths share a basename
would collide locally — a real trap if you edit `Ctx.output`.

---

## 2. Skip-if-done detection (`_is_done`)

```python
if job.done is not None:
    return bool(job.done(job))          # authoritative; exceptions PROPAGATE (fail the job)
if not job.outputs:
    return False                        # no outputs -> can't infer doneness -> always run
return all(os.path.exists(os.path.join(shared_dir, rel))
           for rel in job.outputs.values())
```

- `job.done` errors are **deliberately not caught here** — a bad predicate fails the job rather
  than silently bypassing work.
- The default is existence-only: a **truncated/partial** prior upload still counts as done. This
  is the framework's main correctness caveat — document it, don't "fix" it silently.

---

## 3. NAS download/upload mechanics (`_download`, `rsync_copy`)

- Root is `shared_dir` (`~/shared_data`) from `paradex.utils.path`; cache root is
  `CACHE_ROOT = download_dir/process_cache`.
- `_download` raises `FileNotFoundError: NAS input missing` if the source isn't on NAS.
- **Cache reuse guard is `os.path.getsize` equality only:** if a local copy exists and its size
  equals the NAS file's size, `_download` **skips** the transfer. **Same-size-but-different
  content is NOT re-downloaded** — a real staleness trap.
- Both directions go through `rsync_copy` (`paradex.utils.upload_file`); a falsy return raises
  `RuntimeError: rsync {download,upload} failed`.

---

## 4. Threading / process model (`Processor.run`)

- `num_workers = max(1, min(num_workers, cpu_count()))`.
- **`num_workers == 1` runs in-process** — a plain `for` loop over `_run_job`, refresh after each
  job. No `Pool`, so real tracebacks and no pickling. This is the debug path.
- **`num_workers > 1` uses `multiprocessing.Pool`** — `apply_async(_run_job, ...)` per job, with
  `callback`/`error_callback` appending to a `Manager().list()`. So **`process_fn` and `job.done`
  must be top-level picklable** (they cross the pool).
- Shared state is a `Manager().dict()` (`self.progress`) so pool workers can update status;
  `_update` does read-copy-merge-write (Manager dicts aren't updated in place) and, on every entry
  that has `started_at`, **stamps derived timing**: `elapsed`, `fps` (when `frame`/`total` were
  reported), and `eta` — seconds remaining, frame-rate-based when `frame`/`total` are present else
  percentage-based, `0.0` once the job is in `DONE_STATUSES`, `None` when not estimable. Every
  consumer (console table, ZMQ items, web UI) reads these precomputed numbers rather than
  recomputing. A progress entry can thus carry: `status`, `progress`, `message`, `meta` (seeded),
  `started_at`, `elapsed`, `frame`, `total`, `fps`, `eta`, and `traceback` (on failure).
- `run(update_interval=1.0)`: pool path loops `_refresh(); sleep(interval)` until
  `all(r.ready())`. `_refresh` calls `on_update(progress)` (guarded — hook errors are printed, not
  fatal) and, unless `quiet`, prints the console table. `stop()` terminates+joins the pool.
- **Display helpers** (module-level, exported so `distributed.py`'s dashboard reuses them):
  `fmt_dur(seconds)` → `"1m35s"`/`"1h02m"`; `fmt_job(info)` → one line
  `"25% • 300/1200 @30fps • 10s • ETA 30s - msg"`; `batch_eta(progress, num_workers)` → whole-batch
  seconds remaining (mean wall-time of **completed** jobs ÷ `num_workers`, falling back to the max
  in-flight job `eta` before any job has completed; `None` when nothing is estimable). `_print_table`
  prints a batch-ETA header plus a `fmt_job` line per running job.
- Queries: `get_progress()`, `summary()` (counts per status), `failures()` (`id→entry`, includes
  the stored `traceback`). `run_jobs(...)` is just `Processor(...).run()`.

---

## 5. Distributed dispatch (`distributed.py`) — confirmed from source

Same `Job`/`process_fn`, spread across capture PCs. **The core `Processor` is unchanged**; the
distributed layer only wires its `on_update` hook to ZMQ and reuses SSH launch.

- **`shard(jobs, pc_list=None, pc=None)`** — partitions **one shared** job list by
  `zlib.crc32(job.id.encode()) % n == idx`, where `pc_list = sorted(pc_list or get_pc_list())`
  and `idx = pc_list.index(pc)`. `crc32` is used deliberately (**stable across processes/runs,
  unlike `hash()`**). A PC not in `pc_list` runs **everything** (returns `jobs` unsharded). Only
  needed for pattern A (all PCs read the same NAS); skip it for pattern B (local files).
- **`serve_jobs(jobs, process_fn, num_workers=4, ..., port=1234)`** — worker side, drop-in for
  `run_jobs`. Builds a `Processor(quiet=True, on_update=...)` whose hook calls
  `DataPublisher.send_data`. Each PC publishes **one item per job** (keyed by `job.id`) via
  `_progress_to_items(progress, pc, num_workers)` — each item now also carries the derived timing
  `frame`/`total`/`fps`/`elapsed`/`eta`/`started_at` — **plus** a summary sentinel `_pc::{pc}`
  (`SENTINEL_PREFIX = "_pc::"`) carrying per-status `counts`, `total`, `num_workers`, a per-PC `eta`
  (from `batch_eta`), and a `finished` flag (`all(status in DONE_STATUSES) and total>0`).
  `num_workers` is threaded through so the sentinel's `batch_eta` divides by the right slot count.
  After `proc.run()` it does a **final `on_update` flush + `time.sleep(0.3)`** so the terminal state
  isn't lost, then `publisher.close()` in `finally`.
- **`run_distributed(worker_cmd, pc_list=None, port=1234, ...)`** — main-PC orchestrator. Starts a
  `DataCollector`, **`sleep(0.3)` before launching workers** (ZMQ PUB drops messages with no
  subscriber), then `run_script(worker_cmd, pc_list, log)` SSH-launches the worker on every PC.
  Polls `collector.get_data()` every `poll_interval`, prints `_print_dashboard`, returns the
  aggregated `{job_id_or_sentinel: item}` when `_all_finished` (every PC's sentinel `finished`) or
  `timeout`. `collector.end()` in `finally`.
- **`_print_dashboard`** renders a **rig-wide ETA** (max of the still-working PCs' sentinel `eta`s —
  PCs run in parallel, so the slowest bounds the batch), a per-PC `flag {counts} of total [ETA …]`
  line, and **live per-job `fmt_job` lines** (frame-of-N + ETA) for whatever is currently
  `processing` (capped at 12), then any failures.
- **`_all_finished`**: needs a sentinel from **every** PC in `pc_list` and all `finished` true —
  a silent/crashed worker that never publishes its sentinel makes this **wait forever** unless
  `timeout` is set. Pass `timeout` for unattended runs.

Wire dependencies: `pc_name`, `get_pc_list`, `run_script` (`paradex.io.capture_pc.ssh`),
`DataPublisher`/`DataCollector` (`paradex.io.capture_pc.data_sender`). `worker_cmd` runs under the
conda env on each PC (see `run_script`).

---

## 6. The lazy-import contract (`__init__.py`)

```python
from paradex.process.processor import Job, Ctx, Processor, run_jobs, Skip
try:
    from paradex.process.distributed import serve_jobs, run_distributed, shard
    __all__ = [... , "serve_jobs", "run_distributed", "shard"]
except Exception:          # zmq / pc config missing
    __all__ = ["Job", "Ctx", "Processor", "run_jobs", "Skip"]
```

The `distributed` import is **deliberately swallowed by a bare `except Exception`** so
single-machine use works without zmq/PC config. Trap: if you introduce an import error *inside*
`distributed.py`, `serve_jobs`/`run_distributed`/`shard` silently **vanish** from
`paradex.process` instead of raising. If they go missing, check `__all__` and import
`paradex.process.distributed` directly to see the real error.

---

## 7. Traps that look like bugs but aren't

- **Cache reuse is size-only** (§3) and **default skip is existence-only** (§2) — both trust file
  metadata over content. Intentional; the escape hatch is `Job(done=fn)`.
- **`Ctx.output` uses only the basename** of the NAS relpath for the local path (§1) — distinct
  outputs with the same basename collide locally.
- **`keep_cache=False` deletes cache on failure too** — pass `keep_cache=True` to debug a failed
  job's partial outputs.
- **`run_distributed` blocks forever without a sentinel from every PC** (§5) — set `timeout`.
- **`_run_job` never raises** — a "silent" failure is a `failed` entry in the progress dict with a
  `traceback`, not an exception; check `Processor.failures()`.
- **Swallowed distributed import** (§6) can make the distributed API disappear rather than error.
- Intentional repo-wide typo `dataset_acqusition` (missing 'i') — don't "fix" imports.
