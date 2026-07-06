# Camera communication — design philosophy (why, not how)

Read this before you change how `remote_camera_controller` (main PC) and
`camera_server_daemon` (capture PC) talk. It is the *reasoning* behind the wire
design — the invariants below are load-bearing, and "cleaning them up" without
understanding why they exist is how the P4-class hangs and dead-man races come
back. For the layer map and per-function detail see [`internals.md`](internals.md);
for how to *use* rcc see [`usage.md`](usage.md).

## The one idea: separate planes by failure mode

Multi-PC camera control has three kinds of traffic, and each fails differently.
The whole design falls out of refusing to mix them on one channel:

| Plane | Transport | Port | Fails by | Principle |
|-------|-----------|------|----------|-----------|
| **Command** | ZMQ REQ/REP | 5482 | a command is slow / rejected | rare, needs a reply, must be ack'd |
| **Health/telemetry** | ZMQ PUB/SUB | 5481 | a sample is lost | continuous, latest-wins, loss-tolerant |
| **Liveness** | REQ/REP ping + dead-man | 5480 / 5482 | the *peer* died | must not depend on the other two |

The cardinal rule: **health must never ride the command reply.** It looks
convenient (you already have the socket, just stuff status into the ack) and it
was the original design — but it couples a *loss-tolerant, high-rate* signal to a
*lockstep, must-ack* one. The moment health introspection (`get_summary()` over N
cameras) got slow, it slowed the keepalive, which fed back into the dead-man. So
health now flows on its own PUB channel and the command channel only carries
commands. If you're tempted to return status in a command reply again, don't.

## Why REQ/REP for commands, PUB/SUB for health

- **Commands are REQ/REP** because you genuinely need to know the daemon accepted
  a `start`/`stop` — an ack is the point. REQ/REP's rigidity (exactly one recv per
  send) is acceptable *because commands are rare and serialized per PC*.
- **Health is PUB/SUB** because the daemon should shout its state at its own cadence
  and never block on whether anyone is listening. A subscriber that misses a frame
  just reads the next one 100 ms later. Backpressure/ack would be exactly wrong here.
- **Liveness is deliberately its own concern.** Ping (5480) proves the process is
  up before we ever connect; the dead-man (RCVTIMEO on 5482) proves the *controller*
  is still there. Neither may be inferred from health — a daemon can be PUBing
  healthy frames while its controller has vanished, and vice versa.

## The dead-man switch: recovery must be peer-independent

The daemon releases its single-controller lock and stops cameras if no command
arrives within an idle window. This is the *only* app-independent recovery path —
if the main-PC process crashes, nothing else will free the hardware. Philosophy:
**every acquired resource must have an owner-independent release.** Keep it. The
keepalive exists solely to feed this timer, which is why the keepalive must stay
cheap (no camera introspection on that path).

## Single-owner sockets → one worker per PC

ZMQ sockets are not thread-safe, and REQ/REP has a strict send→recv state machine.
Two principles follow, and together they dictate the worker model:

1. **A socket has exactly one owning thread.** Ever. The per-PC worker owns that
   PC's command socket; nothing else touches it. This is why `reload`/`force_takeover`
   go through the worker queue instead of sending from the caller's thread.
2. **No cross-PC barrier on the hot path.** The earlier code broadcast a command
   and *joined all PCs* every tick. That barrier meant a slow `start` on one PC
   withheld the keepalive from every other PC — so a fast PC would sit idle in
   `recv` and trip *its own* dead-man while waiting on a slow sibling. The fix is
   structural, not a tuned timeout: **each PC is independent.** One worker per PC,
   each sends its own keepalive when idle. A slow PC now delays only its own
   command's return, never another PC's liveness.

Invariant to preserve: *a slow or stuck PC must never be able to affect another
PC's liveness or health.* Any change that reintroduces a join-all on the keepalive
path breaks this.

## What an editor must not break (checklist)

- Health/errors/frame-ids travel on **PUB (5481)**, not command replies.
- The keepalive/heartbeat path does **no** `get_summary()`/`get_all_errors()` — cheap only.
- Each command socket is touched by **one** thread (its worker).
- No **join-all barrier** gates the keepalive.
- The **dead-man** stays: an idle daemon releases the lock + stops cameras.
- Liveness is judged from **PUB silence** (main side) and **command silence** (daemon
  side) — not from a single missed message.

## Failure detection & recovery (each side watches the other)

A process can't detect its own death, so "both sides know" means **each detects the
other's**. The guarantees, and the invariants behind them:

- **daemon dies → controller knows** via PUB silence (`pub_timeout`, ~2 s) *and*
  command timeout (`_send_one` RCVTIMEO → error dict, never a hang). If a capture was
  running, a **sticky `capture_interrupted`** latches and survives the daemon's recovery
  — so an interrupted recording never silently reads healthy again.
- **controller (app) dies → daemon knows** via the dead-man (`recv` RCVTIMEO, ~5 s):
  it stops the cameras and frees the lock. This is the only owner-independent release.
- **daemon restarted mid-session → controller re-registers automatically** but only when
  the daemon's PUB shows the lock *empty* — never when another named controller holds it
  (don't fight a real takeover). Re-register does **not** auto-resume capture; that's the
  caller's call, surfaced through `capture_interrupted`.

Invariants:

- **register only stops a running capture on a real takeover** (different controller),
  never on a same-controller re-register — otherwise one PC's restart would halt the
  healthy PCs that also receive the broadcast register.
- **the detection loops must not be able to die.** `_health_loop`, `run`,
  `monitor_thread`, `pingpong_thread` wrap each iteration in try/except — a transient
  error logs and continues. A dead detection thread = silent blindness, which defeats the
  whole scheme. Never remove those guards.
- **the lock is a real lock now**: register refuses a different live controller unless
  `force`. Optional `PARADEX_CAMERA_TOKEN` (see `protocol.py`) authenticates commands;
  `PROTOCOL_VERSION` on register warns on a controller/daemon git-drift.

## Deliberately *not* done (and why)

- **No ROS2.** rclpy is already a dependency and ROS2 gives topics/services/actions
  + QoS liveliness (a dead-man for free). It was considered and rejected: multi-machine
  DDS discovery/network setup is heavy, and the camera path is kept lean for latency
  and operational simplicity. This is a conscious trade, not an oversight — if you
  reach for a heavier middleware, know you're reversing that call.
- **No generic `capture_pc` reuse for camera control.** `paradex/io/capture_pc`
  has a generic `CommandSender`/`DataPublisher`, but camera control rolls its own raw
  zmq because it needs lock + dead-man + per-PC workers that the generic layer lacks.
  Duplication is accepted here in exchange for those guarantees; don't "unify" them
  without carrying the guarantees across.
