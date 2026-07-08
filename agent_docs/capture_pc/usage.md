# capture_pc — How to Use (from the main PC)

Read this before writing a distributed capture orchestrator. This layer is **transport only** —
it launches remote scripts and moves commands/telemetry between the main PC and the 6 capture PCs.
It knows nothing about cameras; the camera daemon is [`agent_docs/camera_system/`](../camera_system/README.md).

> **Editing the ZMQ/threading/SSH internals**, not just calling? Read [`internals.md`](internals.md).

## TL;DR — the two-sided pattern

Every use is a **main-PC driver** + a **capture-PC client** launched by the driver over SSH.

```python
# ---- main PC (driver) ----
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.io.capture_pc.data_sender import DataCollector

run_script("python src/myapp/client.py")   # spawn client on EVERY capture PC (detached)

sender    = CommandSender(port=6890)        # REQ socket per PC
collector = DataCollector(port=1234)        # SUB socket per PC
collector.start()                           # background poll thread

sender.send_command("start", cmd_info={"save_path": "dataset/001"})
preview = collector.get_data("pc1")         # latest telemetry item (see gotcha on keying)
sender.send_command("stop")

sender.end()                                # broadcasts "exit", closes sockets
collector.end()
```

```python
# ---- capture PC (client, launched by run_script) ----
import threading
from paradex.io.capture_pc.command_sender import CommandReceiver
from paradex.io.capture_pc.data_sender import DataPublisher

events = {"start": threading.Event(), "stop": threading.Event(), "exit": threading.Event()}
recv = CommandReceiver(events, port=6890)   # REP socket; auto-starts its recv thread
pub  = DataPublisher(port=1234, name="pc1") # PUB socket

while not events["exit"].is_set():
    if events["start"].is_set():
        events["start"].clear()
        info = recv.event_info["start"]     # the cmd_info dict the sender sent
        # ... do work; publish previews back ...
        pub.send_data([{"name": "pc1", "data_index": 0}], [jpeg_bytes])
recv.end(); pub.close()
```

You never hardcode IPs — `CommandSender`/`DataCollector`/`run_script` all resolve targets from
`get_pc_list()`/`get_pc_ip()` (`system/current/pc.json`).

> `DataPublisher`/`DataCollector` are now thin **realtime-mode shims** over `Publisher`/`Collector`
> (in [`transport.py`](../../paradex/io/capture_pc/transport.py)) on the msgpack `envelope` format —
> the shims keep the `send_data`/`get_data` API so the example above is unchanged. **New code should
> use `Publisher`/`Collector` directly**: they expose the two delivery modes (`realtime` lossy-latest
> vs `lossless` backpressured), topic filtering, and `get_stats()` (per-source recv/drops/latency).

## Ports & defaults

| Thing | Default | Set by |
|-------|---------|--------|
| SSH port | `77` | `ssh.ssh_port` (module constant) |
| Command channel (REQ-REP) | `6890` | `CommandSender(port=)` / `CommandReceiver(port=)` |
| Data channel (PUB/SUB or PUSH/PULL) | `1234` | `Publisher(port=)` / `Collector(port=)` (and the `DataPublisher`/`DataCollector` shims) |
| Conda env on capture PC | `flir_env` | hardcoded in `run_script` |
| Repo path on capture PC | `~/paradex` | `ssh.repo_path` |
| REQ per-attempt timeout | `60000` ms | `CommandSender(timeout=)` |
| REQ retries per PC | `3` | `CommandSender(retries=)` |
| Data delivery mode | `realtime` | `Publisher(mode=)` / `Collector(mode=)` (shims are always `realtime`) |

Main-PC `send_command` and capture-PC `CommandReceiver` **must use the same port**; likewise
`Publisher`/`Collector` (and a `Collector` must match its `Publisher`'s `mode`). Two apps running at
once need distinct port pairs.

## `run_script` — launching clients

`run_script(script, pc_list=None, log=False)` shells out to `ssh -p 77` for **each** PC and runs
`script` **detached** under conda `flir_env` (double-fork + `nohup`). It returns immediately — it
does **not** wait for, verify, or capture the remote process. `log=True` writes remote
stdout/stderr to `test.log` on the capture PC, else `/dev/null`. On `ssh` failure it only `print`s
(no exception). Requires `anaconda3` + a `flir_env` env + ssh port 77 reachable as `pc@ip` on
every capture PC.

Operator utilities (destructive — keep a human in the loop, don't automate):
- `git_pull(branch, pc_list=None)` — `git fetch && git reset --hard origin/<branch> && git clean -fd` on each PC. **Discards local changes + untracked files.**
- `load_cache(pc_list=None)` — `cp -r ~/shared_data/cache ~/paradex/paradex/cache` on each PC.

## Command channel (REQ-REP)

- `send_command(cmd, wait=False, cmd_info={})` fans the command out to **all** PCs on daemon
  threads and `join()`s them. Each thread sends `{"command", "is_wait", "info"}` as JSON, reads
  one reply, and prints the reply `message`/`Error`. A command is a **string** the client already
  mapped to a `threading.Event`; the receiver just `.set()`s that Event.
- **Reliable via lazy-pirate.** On a ZMQError/timeout the sender discards and rebuilds that PC's REQ
  socket and retries up to `retries` (default 3), so a dead/restarting peer can't permanently wedge
  the orchestrator. A genuinely dead peer still burns `timeout × retries` then is skipped (error
  printed); other PCs are unaffected — each has its own thread.
- `wait=True` makes the receiver `.wait()` on the Event before it acks — use it when the sender
  must block until the client finishes handling the command (the client clears the Event to release).
- `cmd_info` is an arbitrary JSON-able dict delivered to the client as `recv.event_info[cmd]` —
  this is how you pass `save_path`, frame counts, etc.
- `end()` broadcasts `"exit"`, then closes sockets. (`CommandReceiver.end()` sets its exit event,
  joins the recv thread, then closes; its loop uses a `zmq.Poller` and replies exactly once per recv.)

## Data channel (Publisher / Collector)

Reverse direction: capture PCs publish, main PC collects. The wire format is the msgpack `envelope`
(`[topic, header, buf0, …]`, `header = {seq, ts, src, meta, n}`); `seq`+`ts` give **drop detection +
latency for free**. Two selectable delivery modes:

- **`realtime`** (default) — PUB/SUB with a tiny HWM. Under load the publisher *drops* rather than
  blocks (lossy-latest, newest wins); slow-joiner messages are still dropped (see gotchas). Right
  for previews, teleop, live status.
- **`lossless`** — PUSH/PULL with a big HWM + `SNDTIMEO=-1`. Under load the publisher *blocks*
  (backpressure) instead of dropping. Use when every message must arrive (recording pipelines).

- Publisher (`Publisher(port, name, mode)` — the producer **binds**): `send(meta=None, bufs=None,
  topic="data") -> seq` sends the envelope (`send_multipart(copy=False)`, zero-copy); `send_data`
  is an alias. `meta` is a dict or a list of per-item dicts; each item carries a `name` and either a
  `data_index` (or is matched positionally when item-count == buffer-count). `close()` flushes + closes.
- Collector (`Collector(pc_list, port, mode, topics)` — **connects** to every PC): `start()` spawns
  the drain thread; `get(name=None)` returns the latest item for a `name` (or a copy of the whole
  map); `get_stats()` returns per-source `{recv, drops, latency_ms, last_ts_ns}`; `stop()`/`end()`
  tear down. Each stored item is stamped with `src`/`pc`/`seq`/`ts_ns`.
- The `DataPublisher`/`DataCollector` shims keep the old `send_data(metadata, data)` / `get_data(name)`
  API (realtime mode); `get_data` delegates to `Collector.get`.

## <a id="who-calls-this"></a>Who calls this — real consumers

| Caller (role) | Uses |
|---------------|------|
| [`src/calibration/extrinsic/capture.py`](../../src/calibration/extrinsic/capture.py), `intrinsic/capture.py`, `handeye/capture.py` (main) | `run_script` + `CommandSender` + `DataCollector` — launch client, drive capture, pull previews. |
| [`src/calibration/extrinsic/client.py`](../../src/calibration/extrinsic/client.py), `intrinsic/client.py` (capture) | `CommandReceiver` + `DataPublisher` — counterpart side. |
| [`src/capture/camera/stream_remote.py`](../../src/capture/camera/stream_remote.py) (main) / [`stream_client.py`](../../src/capture/camera/stream_client.py) (capture) | both channels + `run_script`. |
| [`src/util/upload_video/main.py`](../../src/util/upload_video/main.py) (main) / [`worker.py`](../../src/util/upload_video/worker.py) (capture) | Built on `paradex.process` (`run_distributed`/`serve_jobs`), which rides `DataPublisher`/`DataCollector` — distributed upload dashboard. (Replaced the old `process.py` + `client.py`.) |
| [`src/process/miyungpa/main.py`](../../src/process/miyungpa/main.py) (main) / [`worker.py`](../../src/process/miyungpa/worker.py) (capture) | `paradex.process.run_distributed` + `serve_jobs` — miyungpa demo processing. (Replaced the old `process_main.py`.) |
| [`paradex/process/distributed.py`](../../paradex/process/distributed.py) | `run_script` + `DataPublisher`/`DataCollector` — generic distributed-job harness (the layer the two rows above build on). |
| [`paradex/io/camera_system/monitor_daemon.py`](../../paradex/io/camera_system/monitor_daemon.py) | `run_script("python src/camera/server_daemon.py", [pc])` — camera monitor (re)launch. |
| `src/util/camera_tuning/remote_tuner.py`, `src/camera/reset_cameras.py`, `src/validate/robot_controller/xarm_test.py` | `run_script` only — remote launch, no channels. |
| `src/validate/command_sender/*`, `src/validate/data_sender/*`, `src/object6d/loftr_client.py` | round-trip smoke tests / LoFTR client wiring the two channels. |

`git_pull` is referenced only in a comment (`src/process/template/main.py`); `load_cache` has no
in-repo callers.

## Gotchas

- **PC list is global config.** `get_pc_list()`/`get_pc_ip()` read `system/current/pc.json`.
  Flip the `system/current` symlink and every socket silently targets a different fleet. The ZMQ
  classes take only `pc_list`, no per-call IP override.
- **`run_script` is detached and unverified.** It succeeds as long as `ssh` exits 0; the remote
  script can crash a millisecond later and you'll never know. Pass `log=True` and read `test.log`
  on the capture PC to debug a client that "didn't start".
- **REQ-REP now retries (lazy-pirate), it no longer wedges.** `send_command` still can't pipeline to
  a PC, but on timeout it rebuilds that PC's socket and retries up to `retries` (default 3), so a
  dead/restarting `CommandReceiver` can't permanently hang the orchestrator. A genuinely dead peer
  still burns `timeout × retries` (default 3×60 s) then is skipped. Other PCs are unaffected.
- **`Collector`/`DataCollector` keys by item `name`, not by PC.** Two PCs publishing the same `name`
  overwrite each other in the latest store. `get_data(pc_name)`/`get(name)` looks up that key — so it
  only works if you set each item's `name` equal to its PC name.
- **`send(copy=False)` is zero-copy.** Don't mutate a buffer after handing it to `send`/`send_data`;
  the bytes are sent without a copy and a late edit can corrupt the wire frame.
- **`meta` must be msgpack-serializable** (dict/list/scalars). A `_default` in `envelope.py` coerces
  stray numpy scalars/arrays, but keep `meta` small — put bulk binary in `bufs`, not in `meta`.
- **realtime PUB/SUB still drops slow-joiner + over-HWM messages**, but now it's observable: `seq`+`ts`
  ride in every envelope and `Collector.get_stats()` reports per-source drops + latency. The 0.1 s
  `sleep` after bind is the only slow-joiner mitigation; use `lossless` mode if you can't lose a frame.
- **ZMQ `CONFLATE` is *not* used** (it doesn't support multipart) — "latest-wins" is a small HWM
  (drop-oldest) plus the collector's per-`name` latest store, not the socket-level CONFLATE option.
