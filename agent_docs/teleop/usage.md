# Teleop Receivers — How to Use (for humans & agents)

Read this before writing code that reads a motion-capture stream. These classes are **input-only**:
they receive poses; they never command a robot. To *retarget* poses onto an arm/hand, go through
`CaptureSession.teleop()` ([`agent_docs/dataset_acquisition/`](../dataset_acquisition/README.md)).

> **Editing the socket loop / packet parsing / pose math**, not just reading it? Read
> [`internals.md`](internals.md) instead.

## TL;DR (XSens — the only working device)

```python
from paradex.io.teleop.xsens.receiver import XSensReceiver

rx = XSensReceiver(port=9763)          # binds UDP 0.0.0.0:9763 + spawns recv thread NOW
                                       # (receiving starts immediately; no start() needed)

data = rx.get_data()                   # snapshot the latest poses (thread-safe copy)
# data = {"Left": {joint: (4,4)} | None, "Right": {joint: (4,4)} | None, "time": <float>}
if data["Right"] is not None:          # None until the first packet arrives — always guard
    wrist_T = data["Right"]["wrist"]   # 4x4, meters, pelvis-relative, wrist-aligned frame

rx.start("dataset/001/raw/teleop")     # (optional) begin logging every packet to .npy
# ... operator moves ...
rx.stop()                              # dump time.npy / left.npy / right.npy into that dir
rx.end()                               # set exit_event, JOIN the recv thread, close socket
```

Construction alone gives you a live stream via `get_data()`. `start`/`stop` are **only** for
recording to disk; you can teleop without ever calling them.

## What `get_data()` returns

| Device | Return shape |
|--------|--------------|
| `XSensReceiver` | `{"Left": {joint→(4,4)} \| None, "Right": {joint→(4,4)} \| None, "time": float}` |
| `OculusReceiver` (hands) | `{"Left": {joint→(4,4)} \| None, "Right": …, "Head": (4,4) \| None, "Root": (4,4) \| None}` — **no `time`** |
| `OculusReceiver(is_body=True)` | the body pose stack or `None` |

- **`Left`/`Right` are `None` until the first valid packet** — the caller must skip. `CaptureSession`
  does exactly `if data["Right"] is None: continue`.
- XSens hand joints: the 20 names in `xsens_joint_name` (`wrist`, then 4 bones × thumb/index/middle/
  ring/pinky). Oculus hand joints: the 26 names in `occulus_hand_joint_name` (adds `palm` + `*_tip`).
- `time` (XSens only) is `time.time()` at the snapshot — **not** the packet's own timestamp (that
  goes into `time.npy` during recording, see below).

## Coordinate frame & units (read this — it is not obvious)

Every returned 4×4 for a hand joint is:

```
XSENS2GLOBAL @ inv(pelvis_pose) @ raw_segment @ inv(XSENS2WRIST_{side})
```

i.e. **pelvis/root-relative** (the operator's pelvis is the origin), then the wrist axes are rotated
into the robot-friendly convention via a per-side `XSENS2WRIST_*` (Oculus: `OCCULUS2WRIST_*`) matrix.
Translations are in **meters**. Left and right use **different** correction matrices — never reuse one
side's frame for the other. The device→world matrices are mirrored in
[`transforms/coordinate.py`](../../paradex/transforms/coordinate.py) (`xsens_left`, `xsens_right`).

## Devices

| Class | Transport | Construct | Config source | Lifecycle | Records? |
|-------|-----------|-----------|---------------|-----------|----------|
| `XSensReceiver` | UDP, binary MVN | `XSensReceiver(port)` | you pass `port` | `get_data`/`start`/`stop`/`end`/`is_error` | ✅ `time/left/right.npy` |
| `OculusReceiver` | UDP, ASCII | `OculusReceiver(ip, is_body=False)` | `network.json` `metaquest` | `get_data`/`quit` | ❌ (no `start`/`stop`) |

## Recording (XSens only)

`start(save_path)` sets `save_event` and inits fresh buffers; from then on **every** message id-2
packet appends all 40 joint poses. `stop()` clears the flag and writes into `save_path`:

```
time.npy    # per-packet header timestamp (header field, uint) — NOT wall clock
left.npy    # np.save of a dict {joint_name: [ (4,4), (4,4), ... ]}  (object array)
right.npy   # same for the right hand
```

`left.npy`/`right.npy` are **pickled dicts** — load with `np.load(..., allow_pickle=True).item()`.
`end()` auto-`stop()`s if a recording is still open, then joins the thread.

## Gotchas

- **Only XSens works.** Constructing `OculusReceiver` raises (undefined `config_dir` + wrong config
  path — see [`internals.md`](internals.md) §Traps). Don't reach for it expecting parity.
- **`CaptureSession(teleop="xsens")` is broken against the shipped config.** It does
  `XSensReceiver(**network_info["xsens"]["param"])`, but `system/current/network.json` ships
  `"xsens": 9763` (a bare int) → `9763["param"]` `TypeError`. Fix at the call site (pass `port=` the
  int) or the config, not by patching the receiver. This is a real, current breakage.
- **`get_data()` can return `None` sides** — guard before indexing a joint.
- **`is_error()` is sticky and coarse** (XSens): it flips `True` after a **single** 1 s socket
  `recvfrom` timeout and is **never cleared**. It means "no packet for ~1 s at least once", not
  "currently dead". Don't treat it as a live health gauge.
- **`XSensReceiver`'s recv thread is NOT a daemon.** If you skip `end()`, the thread keeps the
  process alive. Always `end()` (Oculus uses `quit()` and *is* a daemon).
- **Units are meters, frame is pelvis-relative** — absolute room position is not recoverable from
  these poses (the pelvis was divided out).
- **No `src/validate/teleop/` smoke test ships** for a bare receiver; the practical exerciser is a
  `CaptureSession(teleop="xsens", arm=..., hand=...)` in `src/dataset_acquisition/miyungpa/`.
