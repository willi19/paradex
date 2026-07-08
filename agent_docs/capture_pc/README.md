# agent_docs/capture_pc — agent orientation

Docs for **AI agents working on `paradex/io/capture_pc/`** — the thin **main-PC ↔ capture-PC
orchestration** layer for the distributed rig (1 main PC + 6 capture PCs). It is *transport
plumbing only*: SSH to launch remote scripts, a REQ-REP command channel, and a data-transport
channel (frames / status). **It knows nothing about cameras** — the actual camera daemon lives in
[`agent_docs/camera_system/`](../camera_system/README.md). The module is five live files — `ssh.py`,
`command_sender.py`, and the transport stack (`envelope.py` + `transport.py`, with `data_sender.py`
a thin back-compat shim); `__init__.py` is empty. Read the **one** file for your task.

Mental model: **main PC calls `ssh.run_script` to spawn a *client* on every capture PC, then
`CommandSender` (REQ) pushes commands to those clients (`CommandReceiver`, REP), while data
flows back the other way via a `Publisher` (capture) → `Collector` (main).** The data channel now
rides the msgpack `envelope` format and offers **two delivery modes** — `realtime` (PUB/SUB,
lossy-latest) and `lossless` (PUSH/PULL, backpressured). The old `DataPublisher`/`DataCollector`
are now thin realtime-mode shims over `Publisher`/`Collector`, so existing callers are unchanged.
Every socket target comes from the PC table in `system/current/` (`get_pc_list()`/`get_pc_ip()`)
— this module **never hardcodes IPs**. The two ZMQ channels are independent and run in opposite
directions; SSH only launches, it never carries data.

```
 main PC                                        capture PC (×6)
 ─────────                                       ───────────────
 ssh.run_script  ── ssh:77 launch ────────────▶  <app>_client.py  (conda flir_env, detached)
 CommandSender  ── REQ  tcp:6890 ──────────────▶  CommandReceiver (REP)  → sets threading.Event
 Collector      ◀─ SUB/PULL  tcp:1234 ─────────  Publisher       (PUB/PUSH) ← msgpack envelope
```
(`DataPublisher`/`DataCollector` = the realtime PUB/SUB shim over `Publisher`/`Collector`.)

| Your task | Read |
|-----------|------|
| **Use** it from the main PC: launch clients, send commands, pull telemetry, lifecycle | [`usage.md`](usage.md) |
| **Edit** the transport: ZMQ patterns, threading, framing, SSH mechanics, traps | [`internals.md`](internals.md) |
| Capture actual **camera** frames from another program | [`agent_docs/camera_system/`](../camera_system/README.md) — use `rcc`, not these primitives |

Rule of thumb: **calling** these classes → `usage.md`; **editing** them → `internals.md`.

## File map
| File | What it is |
|------|-----------|
| [`ssh.py`](../../paradex/io/capture_pc/ssh.py) | Fire-and-forget SSH helpers. `run_script` (launch a script under conda `flir_env`, detached), `git_pull`, `load_cache`. Module constants `ssh_port=77`, `repo_path=~/paradex`. |
| [`command_sender.py`](../../paradex/io/capture_pc/command_sender.py) | The **command channel** (ZMQ REQ-REP). `CommandSender` (main PC, fan-out to all PCs, lazy-pirate retry) + `CommandReceiver` (capture PC, `zmq.Poller`, maps command strings → `threading.Event`). |
| [`envelope.py`](../../paradex/io/capture_pc/envelope.py) | The **wire format** shared by all data transport. `encode(topic, meta, bufs, seq, src)` / `decode(parts) -> Message`. Multipart `[topic, header(msgpack), buf0, buf1, …]`; `header = {seq, ts, src, meta, n}`. `seq`+`ts` give drop detection + latency; binary `bufs` stay raw (zero-copy). |
| [`transport.py`](../../paradex/io/capture_pc/transport.py) | The **real data transport**. `Publisher` (capture PC, binds) + `Collector` (main PC, connects to all PCs, keeps newest item per `name`, tracks drops/latency). Two modes: `realtime` (PUB/SUB, lossy-latest) / `lossless` (PUSH/PULL, backpressured). Constants `REALTIME`/`LOSSLESS`. |
| [`data_sender.py`](../../paradex/io/capture_pc/data_sender.py) | **Back-compat shims.** `DataPublisher`/`DataCollector` are thin `Publisher`/`Collector` subclasses (realtime mode) so existing callers work unchanged; also re-exports `Publisher`/`Collector`/`REALTIME`/`LOSSLESS`. **New code: use `Publisher`/`Collector`.** |
| [`__init__.py`](../../paradex/io/capture_pc/__init__.py) | Empty. |

All paths relative to [`paradex/io/capture_pc/`](../../paradex/io/capture_pc/).

## Who calls this
The pattern is always **a `*_remote.py`/`capture.py` on the main PC paired with a `*_client.py`
on the capture PC** — the full consumer table lives in [`usage.md`](usage.md#who-calls-this). The
one hard link to the camera stack: [`camera_system/monitor_daemon.py`](../../paradex/io/camera_system/monitor_daemon.py)
calls `ssh.run_script("python src/camera/server_daemon.py", [pc])` to (re)launch a capture-PC daemon.

## Cross-reference: camera_system
This module is the **generic transport**; the camera pipeline builds its own richer protocol on
the same idea. For camera capture from another program use `remote_camera_controller` (`rcc`),
**not** these primitives directly — see [`../camera_system/README.md`](../camera_system/README.md).

> Note: the sibling capture module is spelled `dataset_acqusition` (missing `i`) — an intentional
> repo-wide typo. Don't "fix" it.
