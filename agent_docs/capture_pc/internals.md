# capture_pc — Internals (for agents editing this module)

**You are here because you are changing the *transport*** — the ZMQ socket patterns, threading,
message framing, or SSH launch mechanics. If you only want to *use* it from the main PC, read
[`usage.md`](usage.md). This layer carries commands and telemetry; it has **no knowledge of
cameras** — that logic lives in [`agent_docs/camera_system/`](../camera_system/README.md).

Five live files, ~660 lines total (`envelope.py`, `transport.py`, `data_sender.py`,
`command_sender.py`, `ssh.py`). `__init__.py` is **empty (0 bytes)** — not a stub. `processor.py`
was deleted (it was an empty placeholder); don't re-create it.

---

## 1. The two channels (confirmed from code)

| Channel | File | Socket pattern | Direction | Port |
|---------|------|----------------|-----------|------|
| Command | [`command_sender.py`](../../paradex/io/capture_pc/command_sender.py) | **REQ ↔ REP** | main → capture | 6890 |
| Data | [`transport.py`](../../paradex/io/capture_pc/transport.py) (+ [`envelope.py`](../../paradex/io/capture_pc/envelope.py) wire format) | **PUB → SUB** (`realtime`) or **PUSH → PULL** (`lossless`) | capture → main | 1234 |
| Launch | [`ssh.py`](../../paradex/io/capture_pc/ssh.py) | `subprocess` → `ssh` | main → capture | 77 |

The data channel replaced the old JSON-multipart PUB/SUB (`data_sender.py`) with the msgpack
`envelope` in `transport.py` — same producer-binds/collector-connects topology, but now with two
selectable delivery modes and built-in seq/latency/drop observability. `DataPublisher`/`DataCollector`
survive as thin realtime-mode shims in `data_sender.py` so existing callers keep working; the earlier
history (a ROUTER-DEALER design was dropped for fire-and-forget PUB/SUB) still applies to the
realtime mode. Don't reintroduce a registration handshake without a reason.

---

## 2. Command channel — REQ-REP

**`CommandSender` (main PC).** In `__init__` opens **one `zmq.REQ` socket per PC** from
`get_pc_list()`/`get_pc_ip()` via `_make_socket` (each with `LINGER=0` and `RCVTIMEO`/`SNDTIMEO =
timeout`, default 60000 ms, then `connect(tcp://ip:port)`). Uses `zmq.Context.instance()`. Extra
ctor args: `retries` (default 3).
- `send_command(cmd, wait, cmd_info)` spawns one **daemon thread per PC** (`_send_to_pc`) and
  `join()`s them all — so PCs are driven in parallel but the call blocks until the slowest replies
  or exhausts retries. Each thread does `send_json({"command","is_wait","info"})` → `recv_json()`.
- **Lazy-pirate retry.** On a `zmq.ZMQError` (timeout / peer gone) `_send_to_pc` calls `_reset_socket`
  — `close(linger=0)` then rebuild+reconnect the REQ socket — and retries, up to `retries` attempts.
  This is why a dead/restarting peer can no longer permanently wedge the sender: the stuck REQ socket
  is discarded rather than waited on forever. A genuinely dead peer still burns `timeout × retries`,
  prints the failure, then the thread returns (that PC is skipped). Nothing is raised.
- `end()` calls `send_command('exit')` first, then `socket.close(linger=0)` all. (No `context.term()`
  — the context is the shared `zmq.Context.instance()`.)

**`CommandReceiver` (capture PC).** Binds a **`zmq.REP`** socket `tcp://*:port` (`LINGER=0`, shared
`Context.instance()`) and **auto-starts** its recv thread in `__init__` (calls `self.start()`). Holds
`event_dict: {cmd_str: Event}` and `event_info: {cmd_str: info}`, plus an `exit_event`.
- `_recv_loop`: uses a **`zmq.Poller`** — `poller.poll(timeout=100)`; on no event it `continue`s (no
  busy-spin, no throwaway sleep Event), otherwise `recv_json()` once and replies exactly once.
- On a **known** command: stash `cmd_info` in `event_info[cmd]`, `event_dict[cmd].set()`, and if
  `is_wait` then `event_dict[cmd].wait()` **inside the loop** before replying
  `{"state":"success","message":...}`. Unknown command → `{"state":"error","message":"Unknown command: ..."}`.
- `end()`: **sets `exit_event`**, `thread.join(timeout=2)`, then `socket.close(linger=0)`.

**The delivery mechanism:** a command doesn't run code here — it **flips a `threading.Event` the
client's own loop already watches**. The client decides what the command means. `is_wait=True`
makes the REP thread block on that same Event until the client clears it, turning the ack into a
completion signal.

> Trap: because the REP thread **blocks in `.wait()`** while `is_wait=True`, a client that never
> clears the Event still stalls that PC's REP socket — but the sender no longer hangs forever on it:
> its REQ attempt times out at `RCVTIMEO`, the socket is rebuilt, and it retries `retries` times
> before giving up. REQ-REP still allows only one in-flight message per socket (no pipelining).

---

## 3. Data channel — envelope + transport

### 3a. Wire format ([`envelope.py`](../../paradex/io/capture_pc/envelope.py))

One format serves **both** realtime frames and job status. `encode(topic, meta, bufs, seq, src) ->
List[bytes]` builds the multipart `[topic, header, buf0, buf1, ...]` where `header` is
**msgpack** of `{seq, ts, src, meta, n}` (`ts = time.time_ns()`, `n = len(bufs)`). `decode(parts) ->
Message` reverses it and raises `ValueError` on a short/malformed frame (< 2 parts or a bad header)
so the receive loop can count-and-skip rather than crash.
- `Message` is a dataclass `{topic, seq, ts_ns, src, meta, bufs}` with a `latency_ms` property
  (`(time.time_ns() - ts_ns)/1e6`). `seq`+`ts` are what give **drop detection + latency** for free.
- `meta` is any msgpack-serializable payload (a dict, or a list of per-item dicts). `bufs` stay raw
  bytes — **no base64/JSON re-encode**, so JPEG/`.npy` cross the wire zero-copy. A `_default`
  coerces stray numpy 0-d scalars (`.item()`) / arrays (`.tolist()`) so robot/teleop meta doesn't
  blow up a send; anything else raises `TypeError`.

### 3b. Transport ([`transport.py`](../../paradex/io/capture_pc/transport.py))

Producer **binds**, collector **connects** — same topology in both modes. Constants `REALTIME` /
`LOSSLESS`. `_REALTIME_HWM = 4`, `_LOSSLESS_HWM = 1000`. Both classes use `zmq.Context.instance()`.

**`Publisher` (capture PC).** `Publisher(port=1234, name=None, mode="realtime")`. `mode="realtime"` →
`zmq.PUB` with `SNDHWM=4` (drop-oldest → lossy-latest); `mode="lossless"` → `zmq.PUSH` with
`SNDHWM=1000` + `SNDTIMEO=-1` (blocks/backpressure, no drops). `LINGER=0`, then `bind(tcp://*:port)`.
Realtime sleeps `0.1` s after bind (slow-joiner mitigation); lossless doesn't (PUSH queues).
- `send(meta=None, bufs=None, topic="data") -> seq` increments a **lock-guarded** `_seq`, `encode`s,
  and `send_multipart(parts, copy=False)` (zero-copy). `send_data(metadata, data, topic)` is an alias.
  `close()` sleeps `0.05` s then `socket.close()` (no `context.term()` — shared context).

**`Collector` (main PC).** `Collector(pc_list=None, port=1234, mode="realtime", topics=None)`. Opens
**one socket per PC** — `zmq.SUB` (realtime, `setsockopt_string(SUBSCRIBE, t)` for each topic in
`topics or ['']`) or `zmq.PULL` (lossless) — `RCVHWM=hwm`, `LINGER=0`, `connect(tcp://ip:port)`, all
registered in one `zmq.Poller`.
- `_loop` (background daemon, gated by `self.collecting`): `poller.poll(100ms)`, then for each ready
  socket **drain** with `recv_multipart(NOBLOCK, copy=True)` until `zmq.Again`, `_ingest` each.
- `_ingest`: `decode` (malformed → print + skip). Update per-source `stats` (`recv`, `last_ts_ns`,
  `latency_ms`); **drop detection** via per-`msg.src` monotonic seq (`prev+1 < seq` adds the gap to
  `drops`). Then for each item in `meta` (a list, or `[meta]` if a single dict): copy it, stamp
  `src`/`pc`/`seq`/`ts_ns`, attach a buffer via `data_index` (or **positionally** when
  `len(items) == len(bufs)`), and store `self.latest[item['name']]` (newest-wins). Guarded by
  `_store_lock`.
- `start()`/`stop()` toggle `self.collecting` + `join(timeout=2)`; `get(name)` returns
  `latest.get(name)` or a `dict(...)` copy of all; `get_stats()` returns a copy of per-source
  `{recv, drops, latency_ms, last_ts_ns}`; `end()` = stop + close all sockets.

**`data_sender.py` shims.** `DataPublisher`/`DataCollector` subclass `Publisher`/`Collector` in
`REALTIME` mode and add only prints + the old method names (`send_data(metadata, data)`,
`get_data(name)` → `Collector.get`). The module also re-exports `Publisher`, `Collector`, `REALTIME`,
`LOSSLESS`. New code should import `Publisher`/`Collector` directly.

> **Buffer indexing.** An item is linked to a buffer by its `meta` `data_index` (index **into
> `bufs`**), or **positionally** when item-count == buffer-count. Unlike the old code there is no
> `+2` topic/JSON offset baked into the collector — `decode` already returns `bufs` as a clean list;
> the `n` field in the header, not a hard-coded offset, says how many buffers follow.

> **`latest` is keyed by item `name`, not PC.** `_ingest` writes `latest[item['name']]`, so two PCs
> emitting the same `name` clobber each other, and `get(name)`/`get_data(name)` only resolves if item
> names equal the intended key. This is a real footgun consumers rely on — not a bug to silently "fix".

> **`copy=False` on send is zero-copy.** The producer hands the buffer straight to ZMQ; mutating that
> buffer after `send` can corrupt the in-flight frame. Copy first if you must reuse the array.

---

## 4. SSH launch mechanics ([`ssh.py`](../../paradex/io/capture_pc/ssh.py))

All three functions loop over `pc_list` (default `get_pc_list()`) sequentially, build a
`ssh -p 77 pc@ip "<remote_cmd>"` string, run it via `subprocess.run(shell=True, check=True)`, and
catch **only** `CalledProcessError` to `print` — no exception propagates, no return value.

`run_script`'s remote command is the load-bearing one:
```
cd ~/paradex && nohup bash -i -c 'source ~/anaconda3/etc/profile.d/conda.sh && \
  conda activate flir_env && <script> &' </dev/null > <log> 2>&1 &
```
- `bash -i` (interactive) so conda init from the profile is available.
- Double background (`... & ' ... &`) + `nohup` + `</dev/null` = fully **detached**; `ssh` returns
  as soon as the remote shell forks, so success ≠ the script actually ran.
- Hard dependencies baked in: an `anaconda3` install, a **`flir_env`** conda env, ssh port 77, and
  the repo at `~/paradex`. None are checked.

Module constants `ssh_port`, `repo_path`, `cache_path`, `paradex_cache_path` are importable;
`ssh_port` is imported by several callers (e.g. upload_video).

---

## 5. Traps that look like bugs but aren't

- **`__init__.py` is empty (0 bytes); `processor.py` was deleted** — nothing imports either. Don't
  scaffold or re-create them.
- **ZMQ `CONFLATE` is deliberately not used** — it doesn't support multipart, so lossy-latest is a
  small `SNDHWM`/`RCVHWM` (drop-oldest) plus the collector's per-`name` `latest` store, not the
  socket option. Don't "simplify" it to `setsockopt(CONFLATE, 1)` — that breaks multi-buffer frames.
- **`send(copy=False)` is zero-copy** — don't mutate a buffer after handing it to `send`/`send_data`.
- **`Publisher`/`Collector` `mode` must match** end-to-end (PUB↔SUB or PUSH↔PULL). A realtime
  Collector against a lossless Publisher (or vice-versa) silently receives nothing.
- **`meta` must be msgpack-serializable.** The `_default` in `envelope.py` rescues numpy 0-d scalars
  and arrays only; other exotic types raise `TypeError` at send. Keep `meta` small, put bulk in `bufs`.
- **Best-effort in realtime, backpressured in lossless.** REQ-REP now retries (lazy-pirate) but still
  isn't exactly-once; realtime PUB/SUB drops slow-joiner + over-HWM messages (now *observable* via
  `seq`/`ts` + `get_stats()`); `lossless` blocks instead of dropping. Pick the mode for the guarantee.
- **PC targeting is import-time global config** via `system/current/pc.json`. The ZMQ classes take
  `pc_list` but no IP override; changing the fleet means changing the `system/current` symlink.
- **`dataset_acqusition` typo** (missing `i`) is intentional repo-wide — don't "fix" imports.

---

## 6. Relationship to camera_system

This module is **generic plumbing**; the camera pipeline in
[`agent_docs/camera_system/`](../camera_system/README.md) builds a richer protocol on the same
REQ-REP + PUB-SUB idea. The **only** hard code link: `camera_system/monitor_daemon.py` calls
`ssh.run_script("python src/camera/server_daemon.py", [pc])` to (re)launch a capture-PC daemon.
If you change `run_script`'s signature or the conda/env assumptions, that call site breaks — grep
`run_script` before editing.
