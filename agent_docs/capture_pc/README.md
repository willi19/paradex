# agent_docs/capture_pc — agent orientation

Docs for **AI agents working on `paradex/io/capture_pc/`** — the thin **main-PC ↔ capture-PC
orchestration** layer for the distributed rig (1 main PC + 6 capture PCs). It is *transport
plumbing only*: SSH to launch remote scripts, a REQ-REP command channel, and a PUB-SUB
telemetry channel. It knows nothing about cameras — the actual camera daemon lives in
`camera_system/` (see cross-reference below). Read this one file; the module is only three
live files.

Mental model: **main PC uses `ssh.run_script` to spawn a *client* on every capture PC, then
`CommandSender` pushes commands to those clients (`CommandReceiver`), while data flows back the
other way `DataPublisher` (capture) → `DataCollector` (main).** Every socket target comes from
the PC table in `system/current/` — this module never hardcodes IPs.

## File map
| File | What it is |
|------|-----------|
| `ssh.py` | Fire-and-forget SSH helpers. `run_script` (launch a script under conda `flir_env`, detached), `git_pull`, `load_cache`. Module constants `ssh_port=77`, `repo_path=~/paradex`. |
| `command_sender.py` | The **command channel** (ZMQ REQ-REP). `CommandSender` (main PC, fan-out to all PCs) + `CommandReceiver` (capture PC, maps command strings → `threading.Event`). |
| `data_sender.py` | The **telemetry channel** (ZMQ PUB-SUB). `DataPublisher` (capture PC, binds + publishes multipart binary) + `DataCollector` (main PC, subscribes to all PCs, keeps latest keyed by item name). |
| `processor.py` | **Empty (0 bytes).** Placeholder; imports nothing, imported by nothing. |
| `__init__.py` | Empty. |

All paths relative to [`paradex/io/capture_pc/`](../../paradex/io/capture_pc/). The daemon this
layer talks to for cameras is documented in [`../camera_system/README.md`](../camera_system/README.md).

---

## Who calls this
The three live files are imported widely; the pattern is always **a `*_remote.py`/`capture.py`
on the main PC paired with a `*_client.py` on the capture PC**.

| Caller (role) | Uses |
|---------------|------|
| [`src/calibration/extrinsic/capture.py`](../../src/calibration/extrinsic/capture.py), `intrinsic/capture.py`, `handeye/capture.py` (main) | `run_script` + `CommandSender` + `DataCollector` — launch client, drive capture, pull previews. |
| [`src/calibration/extrinsic/client.py`](../../src/calibration/extrinsic/client.py), `intrinsic/client.py` (capture) | `CommandReceiver` + `DataPublisher` — the counterpart side. |
| [`src/capture/camera/stream_remote.py`](../../src/capture/camera/stream_remote.py) (main) / [`stream_client.py`](../../src/capture/camera/stream_client.py) (capture) | `run_script`+`CommandSender`+`DataCollector` / `CommandReceiver`+`DataPublisher`. |
| [`src/util/upload_video/process.py`](../../src/util/upload_video/process.py) (main) / [`client.py`](../../src/util/upload_video/client.py) (capture) | `run_script`+`ssh_port`+`DataCollector` / `DataPublisher` — distributed upload dashboard. |
| [`paradex/process/distributed.py`](../../paradex/process/distributed.py) | `run_script` + `DataPublisher`/`DataCollector` — generic distributed-job harness. |
| [`paradex/io/camera_system/monitor_daemon.py`](../../paradex/io/camera_system/monitor_daemon.py) | `run_script("python src/camera/server_daemon.py", [pc])` — how the camera monitor (re)launches a capture-PC daemon. |
| `src/util/camera_tuning/remote_tuner.py`, `src/camera/reset_cameras.py`, `src/process/miyungpa/process_main.py`, `src/validate/robot_controller/xarm_test.py` | `run_script` only — remote launch, no channels. |
| `src/validate/command_sender/*`, `src/validate/data_sender/*`, `src/object6d/loftr_client.py` | Round-trip smoke tests / LoFTR client wiring of the two channels. |

`git_pull` is only referenced in a comment (`src/process/template/main.py`); `load_cache` has
no in-repo callers. Both are operator utilities.

---

## `ssh.py` — remote launch
Shells out to the system `ssh` binary (port **77**) for each PC in `pc_list` (defaults to
`get_pc_list()` from `system/current/`). All three functions loop over PCs sequentially and only
`print` on `CalledProcessError` — no exception propagates.

- **`run_script(script, pc_list=None, log=False)`** — the workhorse. Wraps the script in
  `cd ~/paradex && nohup bash -i -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate
  flir_env && <script> &' </dev/null > <log> 2>&1 &`. So it: activates the **`flir_env`** conda
  env, runs **detached** (double-fork + `nohup`), and returns immediately — it does **not** wait
  for or capture the remote process. `log=True` writes remote stdout/stderr to `test.log`,
  else `/dev/null`.
- **`git_pull(branch, pc_list=None)`** — `git fetch origin && git reset --hard origin/<branch>
  --quiet && git clean -fd` on each PC. **Destructive**: discards local changes and untracked
  files on the capture PCs.
- **`load_cache(pc_list=None)`** — `cp -r ~/shared_data/cache ~/paradex/paradex/cache` on each PC.

Module constants (`ssh_port`, `repo_path`, `cache_path`, `paradex_cache_path`) are importable;
`ssh_port` is imported by several callers.

---

## `command_sender.py` — command channel (REQ-REP)

### `CommandSender` (main PC)
Opens one **REQ** socket per PC (`get_pc_list()`/`get_pc_ip()`), port default **6890**, with
`LINGER=0` and 60 s send/recv timeouts.
- `send_command(cmd, wait=False, cmd_info={})` — fans the command out to **all** PCs on daemon
  threads and `join()`s them; each thread sends `{"command", "is_wait", "info"}` as JSON and
  reads one reply, printing the reply's `message` (or `Error`).
- `end()` — broadcasts `"exit"`, closes sockets, terminates the context.

### `CommandReceiver` (capture PC)
Binds a **REP** socket (`tcp://*:<port>`) and **auto-starts** its receive thread in `__init__`.
Constructed with an `event_dict: {command_str: threading.Event}`.
- On a known command it `.set()`s the mapped Event, stashes `cmd_info` in `self.event_info[cmd]`,
  optionally `.wait()`s the Event when the sender asked `is_wait=True`, then replies
  `{"state":"success"}`; unknown commands reply `{"state":"error"}`.
- The loop uses non-blocking `recv_json(NOBLOCK)` with a 10 ms idle sleep. `end()` joins + closes.

So a command is delivered by **setting a threading.Event the client already listens on** — the
client's own loop decides what the command means; this module just flips the flag and (optionally)
blocks until the client clears/acks it via the same Event.

---

## `data_sender.py` — telemetry channel (PUB-SUB)
Reverse direction: capture PCs publish, main PC collects. The header comment notes this replaced
an earlier ROUTER-DEALER design — it is deliberately **fire-and-forget** (no registration, no
delivery guarantee).

### `DataPublisher` (capture PC)
Binds a **PUB** socket (port default **1234**), sleeps 0.1 s to let subscribers connect.
- `send_data(metadata: List[Dict], data: List[bytes])` — sends a **multipart** message
  `[b'data', json({'timestamp','publisher','items':metadata}), <bytes0>, <bytes1>, ...]`. The
  JSON `items` carry small metadata (e.g. `name`, `data_index`); large binary (JPEG frames) rides
  as raw trailing frames. `close()` sleeps then tears down.

### `DataCollector` (main PC)
Opens one **SUB** socket per PC (subscribe-all) and a `zmq.Poller`.
- Background `_collection_loop` polls (100 ms), reads each ready socket's multipart, JSON-decodes
  part 1, and for each `item` pops `data_index` → attaches `parts[idx+2]` as `item['data']`, tags
  `item['pc']/'timestamp'/'publisher']`, then stores it in `self.latest_data[item['name']]`.
- `get_data(pc_name=None)` returns the latest item for that key (note: keyed by **item `name`**,
  not PC — see gotcha), or a copy of the whole dict. `start()/stop()/end()` manage the thread.

---

## Cross-reference: camera_system
This module is the **generic transport**; the camera pipeline builds its own richer protocol on
top of the same idea. For camera capture from another program use `remote_camera_controller`
(`rcc`), not these primitives directly — see [`../camera_system/README.md`](../camera_system/README.md).
The one hard link between the two: `camera_system/monitor_daemon.py` calls `ssh.run_script` to
launch `src/camera/server_daemon.py` on a capture PC.

---

## Gotchas for editors
- **PC list is global config, not an argument default you can trust.** `get_pc_list()` /
  `get_pc_ip()` read `system/current/pc.json` at import time. Change the active symlink and every
  `CommandSender`/`DataCollector`/`run_script` silently targets a different fleet. There is no
  per-call IP override for the ZMQ classes (only `pc_list`).
- **`run_script` is detached and unverified.** It returns success as long as `ssh` exits 0 — the
  remote script may crash a millisecond later and you won't know. It also hard-requires an
  `anaconda3` install with a **`flir_env`** conda env and `ssh` port **77** reachable by
  `pc_name@ip`. None of that is checked.
- **`git_pull` / `load_cache` are destructive / stateful** (`reset --hard`, `clean -fd`, `cp -r`).
  Don't wire them into an automated path without an operator in the loop.
- **REQ-REP is lockstep and blocking.** A `CommandSender` REQ socket must receive its reply
  before it can send again; if a capture-PC `CommandReceiver` is dead the call blocks until the
  60 s `RCVTIMEO` and then that PC is skipped (error printed, others still processed because each
  PC has its own thread).
- **`DataCollector` keys by item `name`, not by PC.** Two PCs publishing the same item `name`
  overwrite each other in `latest_data`. `get_data(pc_name)` actually looks up
  `latest_data[pc_name]` — i.e. it only works if item names happen to equal PC names.
- **`DataPublisher.send_data` takes `(metadata_list, data_list)`.** Some validate/demo clients
  call it with a single dict (`dp.send_data({...})`) — that path never exercises the binary
  frames and does not match the real signature; don't copy those calls as canonical.
- **PUB-SUB drops the "slow-joiner" first messages.** The 0.1 s sleeps are the only mitigation;
  data sent before a subscriber connects is lost by design.
- **`processor.py` is empty** — not a stub to fill in blindly; nothing imports it.
