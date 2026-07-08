# agent_docs/teleop — agent orientation

Docs for **AI agents working on `paradex/io/teleop/`** — the layer that turns a **motion-capture
suit / VR headset** into a live stream of **hand (and body) poses** the rest of the system can
retarget onto a robot. These are **INPUT receivers only**: they open a UDP socket, decode the
vendor's stream on a background thread, and hand you the latest poses via `get_data()`. Nothing here
moves a robot — that is [`agent_docs/robot_controller/`](../robot_controller/README.md); the two meet
only at the pose dict that [`agent_docs/dataset_acquisition/`](../dataset_acquisition/README.md)'s
`CaptureSession.teleop()` feeds through the retargetor.

Mental model: **construct a receiver → a socket-reading thread fills a shared pose buffer → `get_data()`
snapshots it.** Construction spawns the thread immediately (no `start()` needed to *receive*). Each
device streams **4×4 homogeneous transforms per joint**, already re-expressed **relative to the
pelvis/root** and rotated into a wrist-aligned frame (positions in **meters**). `start()/stop()` are a
separate *recording* concern (dump poses to `.npy`), and only **XSens** implements them.

> **Reality check before you build on this:** only **`XSensReceiver` actually works.** Both
> `OculusReceiver` variants are broken at construction (see the file map + `internals.md` §Traps) and
> the Oculus branch in `CaptureSession` is commented out. Treat Oculus as *reference/aspirational*.

| Your task | Read |
|-----------|------|
| **Use** a receiver: construct, `get_data()`, coord frame, record, shut down | [`usage.md`](usage.md) |
| **Edit** the socket loop / packet parsing / pose transform / lifecycle | [`internals.md`](internals.md) |
| Retarget these poses onto an arm/hand (the consumer) | [`agent_docs/dataset_acquisition/`](../dataset_acquisition/README.md) |
| Understand the device→world frame matrices | [`transforms/coordinate.py`](../../paradex/transforms/coordinate.py) (`xsens_left`/`xsens_right`) |

Rule of thumb: **reading a teleop stream** → `usage.md`; **changing how a packet becomes a pose** →
`internals.md`.

## File map
| File | What it is |
|------|-----------|
| `__init__.py` | Public surface: re-exports `XSensReceiver`, `OculusReceiver`, and the joint-name / parent tables (`xsens_joint_name`, `xsens_joint_parent_name`, `occulus_hand_joint_name`, `occulus_hand_joint_parent_name`). |
| `xsens/receiver.py` | **`XSensReceiver`** — the working device. UDP receiver for the **MVN/Xsens** binary network-stream protocol (`MXTP` header, message id 2). Full lifecycle: `get_data`/`start`/`stop`/`end`/`is_error`. 20 hand joints per side. |
| `occulus/receiver.py` | **`OculusReceiver`** (the one `__init__` imports) — UDP receiver for a **Meta Quest** ASCII stream. **Broken at construction:** references an unimported `config_dir` and a wrong `environment/network.json` path (`internals.md` §Traps). Has `get_data`/`quit` only — **no** `start/stop/end`, so it is not `CaptureSession`-compatible. |
| `deprecated/occulus/receiver.py` | Near-identical earlier `OculusReceiver`. **Scratch — not imported by `__init__`.** Its own import (`from paradex.utils.file_io import config_dir`) also fails (`config_dir` lives in `paradex.utils.system`). Keep as history; don't wire it up. |

All paths relative to [`paradex/io/teleop/`](../../paradex/io/teleop/). Note the intentional repo-wide
misspelling **`occulus`** (and the consumer module `dataset_acqusition`) — do **not** "fix" either;
imports depend on them. The `xsens_*`/`occulus_*` joint tables are the canonical kinematic skeletons
(parent maps) other code (retargetor, visualization) keys off — treat them as API, not internals.

Related: `CaptureSession` ([`agent_docs/dataset_acquisition/`](../dataset_acquisition/README.md)) is the
only in-repo consumer (`teleop="xsens"`); the robot half it drives is
[`agent_docs/robot_controller/`](../robot_controller/README.md).
