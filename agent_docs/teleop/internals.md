# Teleop Receivers — Internals (for agents editing this module)

**You are here because you are changing a receiver's *internals* (socket loop / packet parsing / pose
transform / lifecycle), not just reading its stream.** If you only want the latest poses from another
program, read [`usage.md`](usage.md) instead.

Both receivers share one shape: **a UDP socket read on a background thread → decode → write a shared
pose buffer under a lock → `get_data()` snapshots it.** The differences that matter are the wire
format (binary vs ASCII) and which lifecycle methods exist.

---

## 1. The shared shape

```
__init__      open state, spawn ONE recv Thread (starts immediately), NO connect handshake
  run():      while not stop: sock.recvfrom(...) → parse → transform → under lock, store latest
  get_data()  under lock, deep-copy the latest poses out
  (xsens)     start/stop/end/is_error   |   (oculus) quit
```

| Class | File | Transport / port | Thread | Shutdown |
|-------|------|------------------|--------|----------|
| `XSensReceiver` | [`xsens/receiver.py`](../../paradex/io/teleop/xsens/receiver.py) | UDP `0.0.0.0:port`, `settimeout(1.0)` | plain `Thread` (**not** daemon) | `end()` (join) |
| `OculusReceiver` | [`occulus/receiver.py`](../../paradex/io/teleop/occulus/receiver.py) | UDP `metaquest.ip:port`, no timeout | `Thread(daemon=True)` | `quit()` (join) |

---

## 2. XSens wire protocol (MVN network streaming)

Datagram = **24-byte header + N×32-byte segment records**. The header is unpacked with
`struct.Struct("!6s I B B I B B B B 2s H")` (big-endian):

- `header[0]` = 6-byte id string, asserted to start with `"MXTP"`; `message_id = int(id[4:])`.
- **Only `message_id == 2` is processed** (the "pose: position + quaternion" datagram in the MVN
  protocol). Any other message id is silently ignored — this is why non-pose streams look "dead".
- `header[3]` is the field stored as the per-packet timestamp during recording.

Each 32-byte segment (`parse()`, `struct.unpack("!I 3f 4f")`):

```
[0]        segment id   (uint, discarded)
[1:4]      x, y, z      (float, meters)
[4:8]      quaternion   as (w, x, y, z)  ← MVN order
```

The quaternion is **reordered to scipy's xyzw** before use:
`quaternion = np.array([*decoded[5:], decoded[4]])` (i.e. `[x, y, z, w]`) → `R.from_quat(...)`.
**If you touch this, keep the reorder** — MVN sends w-first, scipy wants w-last.

### Segment slicing (magic indices)

`parse()` returns a flat list; `run()` slices it:

```
pelvis_pose      = parsed[0]          # global root
left_hand_pose   = parsed[23 : 23+20] # 20 joints
right_hand_pose  = parsed[43 : 43+40] # -> parsed[43:63], 20 joints
```

The `23` offset = the 23 body segments MVN streams before the finger segments; each hand is 20
joints matching `xsens_joint_name`. If the sender's segment layout changes, these constants break
silently (you get wrong joints, not an error).

### Pose transform (per joint, under `self.lock`)

```python
self.hand_pose[side][joint] = XSENS2GLOBAL @ inv(pelvis_pose) @ raw @ inv(XSENS2WRIST_{side})
```

`XSENS2GLOBAL` is identity; `XSENS2WRIST_Left`/`_Right` are per-side axis flips (diag-ish sign
matrices). Net effect: **pelvis-relative** poses in a **wrist-aligned** frame, meters. Left and
Right use different `XSENS2WRIST_*` — do not merge the two branches.

---

## 3. Oculus wire protocol (ASCII)

Text UDP, decoded with `token.decode().strip()`. Hand frame format:

```
Left:v0/v1/.../v25|Right:v0/.../v25|Head:...|Root:...|      (trailing '|', dropped via split("|")[:-1])
```

Each `vN` is `x,y,z,qx,qy,qz,qw` (7 comma floats; quat is **xyzw** already, unlike XSens).
`parse_hand` → `{name: (26,4,4)}`. Poses are made root-relative and reframed:

```python
data[name] = unity2EuclidianPelvis_mat @ (inv(Root) @ data[name])
# then, per hand:  data["Right"] @= inv(OCCULUS2WRIST_Right);  data["Left"] @= inv(OCCULUS2WRIST_Left)
```

`unity2Euclidian_mat` handles Unity's left-handed (X-right, Y-up, Z-forward) → right-handed swap.
`is_body=True` takes a single `/`-joined 84-joint list and applies `unity2Euclidian_mat @ P @ mat.T`
(a similarity transform, no root division).

---

## 4. Lifecycle & recording (XSens)

- `start(save_path)` stores the path, allocates fresh `self.data` buffers, sets `save_event`.
- The recv loop appends to `self.data` **only while `save_event` is set** — and it does so **outside**
  the `self.lock` block (the append is after the `with self.lock:` that stores poses). `stop()` reads
  and clears `self.data` also without holding the lock ⇒ a **benign-but-real race**: don't add heavier
  buffer mutation without a lock, or a `stop()` mid-append can `AttributeError` on `None`.
- `stop()` writes `time.npy` (list of header timestamps), `left.npy`/`right.npy` (`np.save` of a
  `dict{joint: [poses]}` → object array, needs `allow_pickle=True` to load).
- `end()` sets `exit_event`, **joins** (non-daemon thread), then `stop()`s if still recording.
- `is_error()` returns `error_event`, set on any `socket.timeout` (1 s no-packet) and **never
  cleared** — sticky, coarse. If you want a live health signal, add a clear path; today it only ever
  latches on.

---

## 5. Traps that look like bugs but aren't — and ones that are

- **`occulus` / `dataset_acqusition` misspellings are intentional** and repo-wide. Imports depend on
  them. Do not rename.
- **`OculusReceiver` (live) does not import `config_dir`.** `occulus/receiver.py` imports
  `network_info` from `paradex.utils.system` but line ~84 references `config_dir` (never imported) →
  **`NameError` at construction.** It also builds the path `config_dir/"environment/network.json"`,
  but the real file is `config_dir/network.json` (no `environment/` dir). **Two** independent breaks;
  the class cannot currently be instantiated. `network_info` is imported but unused. This is why
  `CaptureSession` keeps the Oculus branch commented out.
- **`deprecated/occulus/receiver.py` is broken differently.** It does
  `from paradex.utils.file_io import config_dir`, but `config_dir` lives in `paradex.utils.system`,
  **not** `file_io` → **`ImportError` at module import.** It is not imported by `__init__`; leave it.
- **`OCCULUS2WRIST_Left` is a 1-tuple, not a matrix.** Its literal ends `...]),` (trailing comma) so
  it is `(array,)`. `np.linalg.inv(OCCULUS2WRIST_Left)` then operates on a `(1,4,4)` stack — the Left
  hand transform is almost certainly wrong. `OCCULUS2WRIST_Right` (no trailing comma) is a real matrix.
- **`OculusReceiver` has no `start`/`stop`/`end`.** Only `get_data` + `quit`. It is **not**
  interface-compatible with `XSensReceiver` or `CaptureSession` (which calls `start/stop/end`). Adding
  Oculus to a session means implementing those, not just uncommenting the import.
- **`XSensReceiver` recv thread is not `daemon=True`** (Oculus's is). Forgetting `end()` leaves a
  live thread holding the process open. Keep `end()` reachable in every exit path.
- **Config-schema mismatch is a live breakage** (see [`usage.md`](usage.md) §Gotchas):
  `CaptureSession` does `XSensReceiver(**network_info["xsens"]["param"])` but shipped
  `network.json` has `"xsens": 9763` (int, not `{"param": {...}}`). Fix the call site/config, not the
  receiver signature (`__init__(self, port)`).
- **Commented-out `zmq` scaffolding** sits at the top of `run()` in both Oculus files. The live path
  is plain UDP `socket`; ignore the `zmq` import (unused) unless you are reviving PULL sockets.
