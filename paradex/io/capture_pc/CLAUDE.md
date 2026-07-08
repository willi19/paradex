# CLAUDE.md — paradex/io/capture_pc

## Purpose
The main-PC ↔ capture-PC plumbing: a **data transport** (stream frames / status) and
a **command channel** (start/stop/save/exit), plus SSH helpers to launch remote
scripts. One general message format serves both realtime execution (camera frames,
teleop/robot state) and batch data-processing (job status).

## Files
| File | Role |
|------|------|
| `envelope.py` | The wire format. `encode/decode` a multipart `[topic, header(msgpack), buf0, buf1, …]` message; header = `{seq, ts, src, meta, n}`. `meta` is msgpack-serializable (dict or list of per-item dicts); `bufs` are raw bytes (JPEG/`.npy`) sent zero-copy. `seq`+`ts` give drop detection + latency for free. |
| `transport.py` | `Publisher` / `Collector` — the real transport. Two selectable delivery modes: **`realtime`** (PUB/SUB, tiny HWM → lossy-latest, newest-wins) and **`lossless`** (PUSH/PULL, backpressured, no drops). Producer binds; the main-PC collector connects to every PC. `Collector.get(name)` = newest item per name; `get_stats()` = per-source `{recv, drops, latency_ms}`. |
| `data_sender.py` | Backward-compat shims: `DataPublisher`/`DataCollector` (thin `Publisher`/`Collector` subclasses, realtime mode) so existing callers keep working. **New code: use `Publisher`/`Collector`.** |
| `command_sender.py` | `CommandSender` (main PC) / `CommandReceiver` (capture PC) over REQ/REP. Reliable via **lazy-pirate**: on timeout the sender rebuilds that PC's socket and retries, so a dead/restarting peer can't wedge the orchestrator. |
| `ssh.py` | `run_script` (nohup a remote command in the `flir_env` conda env), `git_pull`, `load_cache`; `ssh_port=77`, `repo_path=~/paradex`. |

## Topology
Producer **binds** a port on each capture PC; the main-PC **collector connects** to
all of them (holds for both realtime SUB and lossless PULL, so switching modes never
changes wiring). Command channel is the reverse: `CommandReceiver` binds on each
capture PC, `CommandSender` connects to all.

## When working here
- Adding a stream (frames / state / status)? Prefer `Publisher(mode=...)` +
  `Collector(mode=...)`. Realtime for "newest wins" (previews, teleop, live status);
  lossless only when every message must arrive (recording pipelines).
- `meta` must be msgpack-serializable — plain dicts/lists/scalars. Stray numpy scalars/
  arrays are coerced (`_default` in `envelope.py`), but keep meta small; put bulk data
  in `bufs`.
- The `paradex.process` distributed dashboard rides on this exact channel
  (`serve_jobs`/`run_distributed` use `DataPublisher`/`DataCollector`).

## Gotchas
- ZMQ `CONFLATE` can't be used (it doesn't support multipart) — realtime "latest-wins"
  is done with a small HWM (drop-oldest) plus the collector's per-name latest store.
- PUB drops messages to subscribers that connect late; realtime `Publisher` sleeps
  ~0.1s after bind, and `run_distributed` subscribes before launching workers.
- `send(copy=False)` keeps image buffers zero-copy — don't mutate a buffer after
  handing it to `send`.
- Hot paths (edit with care): `remote_camera_controller.py`, `camera.py`, `pyspin.py`
  live in the sibling `camera_system/`, not here.
