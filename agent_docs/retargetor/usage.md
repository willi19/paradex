# Retargetor — How to Use (for humans & agents)

Read this before wiring XSens teleop into an arm/hand. You feed it a **human hand pose**,
you get back a **robot wrist pose (4×4)** and a **robot hand joint vector**. It does no IK
and no safety checking — see the gotchas.

> **Editing the mapping formulas / gesture state machine**, not just calling it?
> Read [`internals.md`](internals.md) instead.

## TL;DR

```python
from paradex.retargetor.unimanual import Retargetor        # or: Unimanual_Retargetor
from paradex.retargetor.state import HandStateExtractor

rt    = Retargetor(arm_name="xarm", hand_name="inspire")   # arm ∈ {None,"xarm","franka"}, hand ∈ {None,"inspire","allegro"}
state = HandStateExtractor()

home_pose = arm.get_data()["position"]                     # (4,4) robot wrist pose at start
rt.start(home_pose)                                        # MUST be called before get_action

while True:
    data = xsens.get_data()                                # {"Left": {...}|None, "Right": {...}|None}
    if data["Right"] is None:                              # no packet yet — skip
        continue
    if state.get_state(data["Left"]) != 0:                 # left hand chose a non-active mode
        rt.stop(); continue
    wrist_pose, hand_action = rt.get_action(data)          # (4,4), (16,)|(6,)|None
    arm.move(wrist_pose)                                    # 4x4 → XArmController cartesian
    hand.move(hand_action)                                 # (16,) allegro rad | (6,) inspire 0–1000
```

`get_action` reads **only `data["Right"]`**. The left hand is *not* retargeted — it is the mode
selector (see `HandStateExtractor` below).

## Constructor

`Retargetor(arm_name=None, hand_name=None)`:
- `arm_name` must be one of `None`, `"xarm"`, `"franka"` else `ValueError`.
- `hand_name` must be one of `None`, `"inspire"`, `"allegro"` else `ValueError`.
- Picks the hand map: `inspire`→`hand_regargetor.inspire`, `allegro`→`hand_regargetor.allegro`,
  `None`→no hand (returns `hand_action=None`).
- Loads `device2wrist = DEVICE2WRIST[arm_name]` (or `[hand_name]` if `arm_name is None`) and
  `device2global = DEVICE2GLOBAL[arm_name]` (else identity). **`device2global` is loaded but never
  used in `get_action`** — do not rely on it doing anything.

## Lifecycle: `start` → `get_action*` → `stop`

| Call | Effect |
|------|--------|
| `start(home_pose)` | Latches `init_robot_pose = home_pose.copy()`, `last_arm_pose = home_pose`, and **resets `init_human_pose = None`**. Call this every time you (re)enter the active state. |
| `get_action(data)` | First call after `start` latches `init_human_pose = data["Right"]["wrist"]` (the reference). Returns `(arm_action_4x4, hand_action)`. |
| `stop()` | Freezes: `init_human_pose = None` and `init_robot_pose = last_arm_pose`. Next `get_action` re-latches the human reference against the **current** robot pose → lets the operator "clutch" (reposition their hand without moving the robot). |

**You must call `start()` before the first `get_action()`** — otherwise `self.init_human_pose`
does not exist and you get `AttributeError`. `stop()` then `get_action()` (without a new `start`)
is fine and is the intended clutch/regrip flow (this is exactly what `teleop()` does on state 1/2).

## Input shape (`data`)

From `xsens.get_data()` ([`paradex/io/teleop/xsens/receiver.py`](../../paradex/io/teleop/xsens/receiver.py)):

```python
data = {"Right": {joint_name: (4,4) float}, "Left": {...}}   # or None per side if no packet
```

- 20 `joint_name`s: `"wrist"`, then `{thumb,index,middle,ring,pinky}_{metacarpal,proximal,intermediate,distal}`
  (thumb has no `intermediate`, but the key list still contains all four names per finger key set —
  see `receiver.py` `xsens_joint_name`).
- Each value is a **4×4 homogeneous pose in the global frame**, metres for translation.
- The retargetor keys into this dict by name (`hand_pose_frame["wrist"]`, `..["index_distal"]`, …).

## Output shapes & units

| `hand_name` | `hand_action` | Units | Consumer |
|-------------|---------------|-------|----------|
| `"allegro"` | `(16,)` np.float | radians | `AllegroController.move` (clips to `MAX_ANGLE=2.1`) |
| `"inspire"` | `(6,)` np.float | 0–1000 motor counts | `InspireControllerIP.move` |
| `None` | `None` | — | no hand |

`arm_action` is always a **`(4,4)` homogeneous wrist pose** in the robot's frame, suitable for
`XArmController.move(pose_4x4, is_servo=...)`. Its rotation = robot-home rotation ∘ human wrist
rotation-delta (frame-corrected via `device2wrist`); its translation = robot-home translation +
**raw** human wrist translation-delta.

## `HandStateExtractor` (the mode selector)

`get_state(pose_data)` takes **one side's** joint dict (teleop passes `data["Left"]`) and returns:

| Code | Comment in source | Straightness pattern (index, middle, ring, pinky) |
|------|-------------------|---------------------------------------------------|
| `0` | (default) | anything not matching below → **active / retarget** |
| `1` | "Fist pose" | all four "straight" |
| `2` | "V pose" | index+middle straight, ring+pinky not |
| `3` | "Spider pose" | index+pinky straight, middle+ring not |

"Straight" here means the finger's relative joint rotations stay below a threshold (`[2,1] ≤ 0.8`);
**the comment labels may not match your intuition of the gesture** — trust the numeric pattern, and
verify on the real glove. In `teleop()` these map to: `0`=drive robot, `1`=pause (clutch), `2`=stop-hold,
`3`=exit-hold (holds >90 ticks end the loop).

## Gotchas

- **Right hand only.** `get_action` hardcodes `data["Right"]`. There is no left-hand retargeting and
  no bimanual mode despite the module name context — `unimanual.py` is literally uni-manual.
- **`start()` is mandatory** before `get_action` (else `AttributeError` on `init_human_pose`).
- **Arm translation is 1:1 and un-transformed.** No scaling, no `device2global`. If human and robot
  base frames are not already aligned, the arm will move in the "wrong" direction. This is a known
  limitation, not a bug you can fix by flipping a flag.
- **No joint-limit / reachability / collision check.** Outputs feed `move()` directly. A hand formula
  can emit an out-of-range value (e.g. inspire outside 0–1000, allegro beyond `MAX_ANGLE`); the
  *controller* clamps some of it, the arm can *fault*. Watch the real robot.
- **Hand maps are rig-specific magic.** The constants (`0.35*1.5`, `1.2`, `*2000`, `3.5`, `-1.57`, …)
  were tuned to one hand. A different hand or a different XSens calibration silently degrades quality.
- **`franka` arm and `franka`/`allegro` frames**: `DEVICE2WRIST["franka"]` exists but the arm side is
  otherwise untested here, and no `franka` entry means `DEVICE2GLOBAL["franka"]` *is* present but the
  controller stack only implements `xarm`. Prefer `xarm`.
- Intentional repo typo: consumer module is `paradex.dataset_acqusition` (missing `i`). Filename
  `hand_regargetor.py` is also intentionally misspelled. Do not "correct" either.
