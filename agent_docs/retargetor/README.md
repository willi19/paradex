# agent_docs/retargetor — agent orientation

Docs for **AI agents working on `paradex/retargetor/`** — the layer that maps a **human
hand pose** (from XSens teleop) onto a **robot arm wrist pose + robot hand joint angles**.
It is the glue between the *input* half (teleop, [`agent_docs/teleop/`](../teleop/README.md))
and the *output* half (arm/hand controllers, [`agent_docs/robot_controller/`](../robot_controller/README.md)).
The two meet only at the pair `(arm_action_4x4, hand_action)` that `get_action()` returns.

Mental model: **`Retargetor.start(home_pose)` → per-frame `get_action(data)` → `(wrist_pose, hand_action)`.**
There is **no optimization and no IK here** — retargeting is a pile of **hand-tuned geometric
formulas** (`arctan`/`arccos` on relative joint rotations, with magic constants) that were
fit to *one specific* robot rig. The arm side is even simpler: it copies the human wrist's
**rotation delta** and **raw translation delta** onto the robot's home pose. Outputs go
**straight to `arm.move()` / `hand.move()`** with no limit/collision checking, so a bad
input becomes a bad command. Read the **one** file for your task.

| Your task | Read |
|-----------|------|
| **Use** it: construct, `start`, feed a pose, get robot actions out; shapes/units | [`usage.md`](usage.md) |
| **Edit** the mapping math / state machine / add a hand | [`internals.md`](internals.md) |
| Understand the **input** dict (`data["Right"]`, joint names, frames) | [`agent_docs/teleop/`](../teleop/README.md) |
| Understand where the **output** goes (`arm.move`/`hand.move` units) | [`agent_docs/robot_controller/`](../robot_controller/README.md) |

Rule of thumb: **calling** the retargetor → `usage.md`; **editing** the formulas → `internals.md`.

## The mental model in one paragraph

The **primary consumer** is `CaptureSession.teleop()` in
[`paradex/dataset_acqusition/capture.py`](../../paradex/dataset_acqusition/capture.py) (note the
repo-wide intentional typo `dataset_acqusition` — do **not** "fix" it). Each ~10 ms tick it
pulls `data = xsens.get_data()`, uses the operator's **left** hand as a *gesture state machine*
(`HandStateExtractor`) to start/pause/stop/exit, and — while in the active state — feeds the
operator's **right** hand into `Retargetor.get_action(data)`. The returned wrist pose drives the
arm and the returned finger vector drives the hand. **Only `data["Right"]` is ever retargeted;
the left hand only picks the mode.**

## File map
| File | What it is |
|------|-----------|
| `__init__.py` | Public surface: re-exports `Retargetor as Unimanual_Retargetor` and `HandStateExtractor`. |
| `unimanual.py` | `Retargetor` — the arm+hand driver. Holds `init_human_pose`/`init_robot_pose`, computes the wrist pose from wrist deltas, dispatches the hand to one of the functions in `hand_regargetor.py`. |
| `hand_regargetor.py` | Two pure functions `allegro(hand_pose_frame)` → `(16,)` rad and `inspire(hand_pose_frame)` → `(6,)` 0–1000 units. **Filename is misspelled `regargetor` — that's the real name, do not rename.** |
| `state.py` | `HandStateExtractor.get_state(pose_data)` → int gesture code (0/1/2/3) from finger straightness. Drives the teleop state machine, **not** the robot. |

All paths relative to [`paradex/retargetor/`](../../paradex/retargetor/).

## Cross-links
- Input source: [`agent_docs/teleop/`](../teleop/README.md) — the XSens receiver produces `data`.
- Output consumers: [`agent_docs/robot_controller/`](../robot_controller/README.md) — `arm.move(4x4)`,
  `hand.move((16,))`/`hand.move((6,))`.
- Orchestrator: `CaptureSession.teleop()` — [`agent_docs/dataset_acquisition/`](../dataset_acquisition/README.md).
- Frame constants (`DEVICE2WRIST`, `DEVICE2GLOBAL`) live in
  [`paradex/transforms/coordinate.py`](../../paradex/transforms/coordinate.py) — see [`agent_docs/transforms/`](../transforms/README.md).

## The one caveat to carry into any task
This is an **open-loop, un-calibrated heuristic**. Arm translation is the **raw** human-wrist
displacement (metres, global frame) added to `home_pose` — **not scaled, not passed through
`device2global`** (which `get_action` computes but never uses). The hand formulas are magic-number
maps tuned to a specific hand. Nothing clamps to joint limits. Treat every output as "send it and
watch the real robot," never "this is guaranteed reachable."
