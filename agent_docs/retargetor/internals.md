# Retargetor — Internals (for agents editing this module)

**You are here because you are changing the retargeting math, the gesture state machine, or
adding a hand — not just calling it.** To *use* it, read [`usage.md`](usage.md).

The whole module is ~150 lines of numpy across three files. There is **no optimizer, no
Pinocchio, no learned model** — it is closed-form geometry with hand-tuned constants. Every
claim below is traceable to the source; the traps are where "obvious" edits silently break motion.

---

## 1. Data flow

```
xsens.get_data() ──► data = {"Left": {20×(4,4)}, "Right": {20×(4,4)}}   # global-frame joint poses
                         │                         │
      HandStateExtractor │                         │ Retargetor.get_action
      .get_state(Left) ──┘                         └─► arm_action (4,4)  ── arm.move
        → 0/1/2/3 mode                                 hand_action (16,)/(6,)/None ── hand.move
```

`HandStateExtractor` and `Retargetor` are **independent** objects; the loop in
`CaptureSession.teleop()` ([`../../paradex/dataset_acqusition/capture.py`](../../paradex/dataset_acqusition/capture.py))
wires them: `get_state(data["Left"])` picks the mode, and only in mode `0` does it call
`get_action(data)` (which reads `data["Right"]`).

---

## 2. `Retargetor` — arm wrist mapping ([`unimanual.py`](../../paradex/retargetor/unimanual.py))

State it holds: `init_human_pose` (reference right-wrist pose, latched lazily), `init_robot_pose`
(robot home, latched in `start`), `last_arm_pose` (last emitted, for the clutch on `stop`),
`device2wrist`, `device2global`.

The core of `get_action(data)`:

```python
if self.init_human_pose is None:                       # lazy latch on first frame after start/stop
    self.init_human_pose = data["Right"]["wrist"].copy()

delta_R = device2wrist[:3,:3].T @ inv(init_human_pose[:3,:3]) @ data["Right"]["wrist"][:3,:3] @ device2wrist[:3,:3]
delta_t = data["Right"]["wrist"][:3,3] - init_human_pose[:3,3]      # RAW metres, global frame

robot_wrist_pose[:3,:3] = init_robot_pose[:3,:3] @ delta_R
robot_wrist_pose[:3,3]  = init_robot_pose[:3,3]  + delta_t          # NOTE: delta_t added directly
robot_wrist_pose[3,3]   = 1
self.last_arm_pose = robot_wrist_pose.copy()
```

Key facts:
- **Rotation** is a *similarity transform* of the human rotation-delta into the robot wrist frame:
  `device2wrist.R.T @ Δ @ device2wrist.R`. This is the only place `device2wrist` is used.
- **Translation** is the raw human wrist displacement, **not** rotated by `device2wrist` and **not**
  touched by `device2global`. `device2global` is computed in `__init__` but referenced **nowhere** in
  `get_action`. If you think translation should be frame-corrected, that is a design change, not a
  fix-in-place — and it will alter every recorded demo's trajectory.
- `get_action` returns `robot_wrist_pose.copy()` as `arm_action` unconditionally (even when
  `arm_name is None` — the caller just ignores it).

### `start` / `stop` and the clutch

```python
start(home_pose):  init_robot_pose = home_pose;  last_arm_pose = home_pose;  init_human_pose = None
stop():            init_human_pose = None;        init_robot_pose = last_arm_pose
```

`stop()` is the **clutch/regrip**: it re-bases `init_robot_pose` to the last emitted pose and clears
the human reference, so the next `get_action` measures deltas from *here*. This is why `teleop()`
calls `stop()` on pause/stop states — the operator can reposition their arm without dragging the robot.
`start()` must precede the first `get_action`, or `init_human_pose` (referenced before assignment paths
exist) — actually `init_robot_pose` — is undefined → `AttributeError`.

---

## 3. Hand maps ([`hand_regargetor.py`](../../paradex/retargetor/hand_regargetor.py))

Two **pure functions** `allegro(hand_pose_frame)` and `inspire(hand_pose_frame)` taking the right-hand
joint dict (`hand_pose_frame == data["Right"]`) and returning the robot finger vector. Both work by
expressing each finger's pose **relative to its parent or to the wrist**, then reading specific matrix
elements or `arctan/arccos` of directions. No calibration, pure geometry + magic constants.

### `allegro` → `(16,)` radians
- Fingers **index, middle, ring** (i=0,1,2 → output joints `4i .. 4i+3`); **thumb** → joints `12..15`.
  **Pinky is not used** (Allegro is a 4-finger hand: index/middle/ring/thumb).
- Per finger: spread (`4i`) from the wrist-frame tip direction (`arctan(x/z)·(0.9−y)`), then three
  flexion joints from `arccos` of a relative-rotation element (`rot_mat[1,1]` guarded by `rot_mat[2,1]≥0`).
  A hand-tuned rescale on the proximal joint: `(angle−0.35)*1.5`.
- Thumb: metacarpal from `xyz`-euler of `wristᵀ·thumb_metacarpal` → `angles[12]=euler[0]`,
  `angles[13]=−euler[2]−1.57`; the two distal joints from `rot_mat[2,1]*1.2`.

### `inspire` → `(6,)` in 0–1000 counts
- Iterates `["thumb","index","middle","ring","pinky"]` (i=0..4). **Output index is `4-i`** for the four
  non-thumb fingers, so the vector is `[pinky, ring, middle, index, thumb_a, thumb_b]`:
  `[0]=pinky, [1]=ring, [2]=middle, [3]=index`, and **thumb writes `[4]` and `[5]`**.
- Non-thumb flexion: `arctan2(tip[2,1], tip[1,1])` (wrist-frame), wrapped to `[-π/2, …]`, mapped
  `(1 − clamp(angle/π,0,1)) · 1000` (0=fully flexed direction, 1000=extended — matches Inspire's
  "1000 open, 0 closed" convention).
- Thumb is a two-DOF special case with sign flips (`tip_direction[1]*=−1; [2]*=−1`) and two branches on
  `tip_direction[0]`, using constants `2000`, `3.5`, `1000` chosen "for better range" (per the source
  comment). This is the most fragile part; expect to re-tune it for any thumb geometry change.

**All magic numbers (`0.9`, `0.35`, `1.5`, `1.2`, `1.57`, `2000`, `3.5`) are rig-specific.** They are
not derived from URDF; they were fit by hand. Changing the robot hand ⇒ re-fit, not re-derive.

---

## 4. `HandStateExtractor` ([`state.py`](../../paradex/retargetor/state.py))

Gesture classifier over **one** hand's joint dict. `check_straight(pose_data)` returns a 4-bool list
for `[index, middle, ring, pinky]`; a finger is marked **not straight** (`False`) if any of its joints
bends past threshold:
- metacarpal: absolute `pose_data[f_metacarpal][2,1] > 0.8` → not straight;
- proximal/intermediate/distal: relative to parent, `inv(parent) @ joint`, element `[2,1] > 0.8` → not straight.

`get_state` maps the boolean tuple to a code (thumb is ignored entirely):

| Returns | Condition |
|---------|-----------|
| `2` | `[T,T,F,F]` — "V pose" |
| `1` | `[T,T,T,T]` — "Fist pose" (per comment) |
| `3` | `[T,F,F,T]` — "Spider pose" |
| `0` | anything else (default → active/retarget) |

The comment labels (`# Fist pose`, etc.) describe the operator's *intended* gesture and **need not
match the literal straightness math** — do not trust the label, trust the tuple. If you change the
`0.8` threshold you change all four gestures' sensitivity at once.

---

## 5. Traps that look like bugs but aren't (and one that is)

- **`device2global` unused in `get_action`.** Loaded in `__init__`, never read. Not dead-safe to
  "wire in" without redefining the arm mapping — leave it or change deliberately.
- **Raw translation delta.** No scale, no frame rotation on `delta_t`. Intentional in current code;
  changing it retroactively invalidates comparisons against existing demos.
- **`inspire` output index `4-i`** reverses finger order — it is deliberate, matching the Inspire
  register layout. Don't "simplify" it to `i`.
- **Pinky absent from `allegro`** — correct (4-finger hand). Don't add a 5th finger.
- **Filename `hand_regargetor.py`** and module `dataset_acqusition` are **intentional misspellings**
  repo-wide. Renaming breaks imports (`unimanual.py` imports `from paradex.retargetor.hand_regargetor
  import inspire, allegro`; `capture.py` imports from `paradex.retargetor.*`).
- **Genuine sharp edge:** `get_action` accesses `self.init_robot_pose` which only `start()` sets. Call
  `get_action` before `start` and it raises `AttributeError`. The lazy `init_human_pose` latch masks
  this for the human reference but not for the robot reference. Any new entrypoint must call `start`.

---

## 6. Adding a hand (the checklist)

1. Add a function `myhand(hand_pose_frame) -> (N,)` to [`hand_regargetor.py`](../../paradex/retargetor/hand_regargetor.py),
   keyed by the same joint names (`hand_pose_frame["wrist"]`, `"<finger>_<joint>"`).
2. Register it in `Retargetor.__init__`'s `hand_name` dispatch and add the name to the validation set.
3. Add `DEVICE2WRIST["myhand"]` (and, if used arm-less, that path reads `DEVICE2WRIST[hand_name]`) in
   [`../../paradex/transforms/coordinate.py`](../../paradex/transforms/coordinate.py).
4. Match the **controller's** DOF and units ([`agent_docs/robot_controller/`](../robot_controller/README.md)):
   the vector you return goes straight into `hand.move()` — get the length and the unit scale right, and
   respect the controller's clamp (Allegro `MAX_ANGLE=2.1` rad, Inspire 0–1000).
5. There is no test suite; validate with `src/validate/` teleop scripts on the real rig.
