# IsaacGym Simulator — How to Use

Read this before writing code that drives `paradex.simulator`. Editing the wrapper
itself (sim/env/actor creation, the step loop)? Read [`internals.md`](internals.md).
Class source: [`isaac.py`](../../paradex/simulator/isaac.py).

> **No in-repo call site exists** — this is reconstructed from the source. The default
> single-camera path works; the multi-camera and state-save paths are broken (see
> Gotchas). Verify against the code before relying on anything not shown here.

## Import order (do this first, always)

```python
from isaacgym import gymapi                       # or: import paradex.simulator
import torch                                       # torch AFTER isaacgym — mandatory
import numpy as np
from paradex.simulator import IsaacSimulator      # == isaac.Simulator
```

IsaacGym patches CUDA/torch at import; if `torch` (or anything importing it) loads
first, IsaacGym raises `PyTorch was imported before isaacgym modules`. Keep the
`isaacgym`/`paradex.simulator` import at the very top of your entry script.

## TL;DR — the working loop

```python
sim = IsaacSimulator(headless=True, plane=True)   # grabs gym, creates PhysX sim (GPU 0)

# 1. compile URDFs → asset templates (once per robot/object type)
sim.load_robot_asset("xarm", "allegro")           # rsc/robot/xarm_allegro.urdf (+ _vis copy)
sim.load_object_asset("pringles")                 # rsc/object/pringles/pringles.urdf

# 2. build an env and spawn actors from those assets
sim.add_env("env0", {
    "robot":      {"r0": ("xarm", "allegro")},    # PHYSICAL, PD-driven
    "robot_vis":  {},                              # ghost overlay (optional)
    "object":     {"o0": "pringles"},              # PHYSICAL, free dynamics
    "object_vis": {},
})

# 3. drive + advance
q = np.zeros(22, dtype=np.float32)                 # 6 (xarm) + 16 (allegro) DOF
for t in range(300):
    sim.step("env0", {"robot": {"r0": q}, "robot_vis": {}, "object_vis": {}})
    sim.tick()                                     # one dt = 1/30 s of physics + render

sim.terminate()                                    # destroy envs/viewer/sim
```

`load_robot_asset`/`load_object_asset` **must run before** `add_env` — `add_env` looks
the asset up by name and `KeyError`s if it was never loaded.

## Constructor & world params

`IsaacSimulator(headless, plane=True)`

| Param | Meaning |
|-------|---------|
| `headless=True` | no on-screen viewer; still renders offscreen cameras. `False` opens a GLFW window and `tick()` also draws + `sync_frame_time`. |
| `plane=True` | add a static ground plane (z-up, `distance=0.0525`, friction 1.0/0.8). |

Fixed sim params (in `generate_sim`, not exposed): `dt = 1/30`, `substeps = 2`,
`up_axis = Z`, `gravity = (0,0,-9.8)`, PhysX GPU solver (`solver_type=1`,
6 position / 1 velocity iterations, `contact_offset=0.01`). To change any of these you
edit [`isaac.py`](../../paradex/simulator/isaac.py) — there is no config hook.

## `env_info` — what `add_env(name, env_info)` expects

A dict with **exactly these four keys** (use `{}` for the ones you don't want):

| Key | Value shape | Meaning |
|-----|-------------|---------|
| `"robot"` | `{actor_name: (arm_name, hand_name)}` | physical robot; `arm_name=None` → uses `{hand}_float.urdf` |
| `"robot_vis"` | `{actor_name: (arm_name, hand_name)}` | ghost robot (gravity off, teleported) |
| `"object"` | `{actor_name: obj_name}` | physical free-body object |
| `"object_vis"` | `{actor_name: obj_name}` | ghost object |

`(arm, hand)` naming resolves via `load_robot_name`: both set → `arm_hand`; one `None`
→ the other; used to index the asset you loaded. Robot base is **fixed** at the world
origin (`fix_base_link=True`); objects spawn at a temporary `(0.5, 0, 0)` — move them
with `reset()` before stepping.

## `step` vs `reset` — targets vs teleport

Both take `(name, action_dict)`, but they are **not symmetric** — mind the keys:

| Method | Reads keys | robot / robot_vis | object handling |
|--------|-----------|-------------------|-----------------|
| `step` | `robot`, `robot_vis`, `object_vis` | robot → **PD position target** (`set_actor_dof_position_targets`); robot_vis → teleport DOF | `object_vis` → teleport pose; **physical `object` is left to physics** (not in step) |
| `reset` | `robot`, `robot_vis`, `object` | both → **teleport DOF** (`set_actor_dof_states`, instant) | `object` → teleport rigid-body pose; **no `object_vis`** |

- **DOF actions** (`robot`/`robot_vis`) are a flat vector over the merged URDF's DOFs in
  URDF order — e.g. `xarm_allegro` = 6 arm + 16 hand = 22. Radians for revolute joints.
- **Object poses** (`object`/`object_vis`) are `4x4` homogeneous transforms (numpy);
  the wrapper converts rotation via scipy `R.from_matrix(...).as_quat()` (xyzw) internally.
- Typical pattern: `reset(...)` once to place everything, then `step(...)` + `tick()` in
  a loop. `step` only *targets* the physical robot — it reaches the target over several
  `tick`s under its PD gains (arm stiffness 1000 / damping 10; hand 500 / 10).

## Reading state back

There is no convenience getter on `Simulator`; call the gym API on the stored handles:

```python
env   = sim.env_dict["env0"]
actor = sim.actor_handle_dict["env0"]["robot"]["r0"]
dof   = sim.gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)   # dof["pos"], dof["vel"]
rb    = sim.gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)  # ["pose"]["p"|"r"]
```

Object world pose lives in `rb["pose"]["p"]` (xyz) / `["r"]` (xyzw quat) on element 0.
`save_stateinfo` shows the full read-back recipe but its enabling method is commented
out — don't call it (see Gotchas).

## Offscreen video

```python
sim.load_camera("env0")                 # default: one camera, fov 75°, 2048x1536, pose (1.5,0,1.5)->(0,0,0.3)
sim.set_videopath("env0", "/path/out")  # writes /path/out/env0/default.mp4 (mp4v, 30 fps)
# ... step/tick loop ...  (tick renders + writes a frame each call while save_video is on)
sim.save()                              # release the VideoWriter(s) — REQUIRED or the mp4 is truncated
```

`tick()` grabs `IMAGE_COLOR`, drops alpha, and flips RGB→BGR for OpenCV automatically.
`terminate()` calls `save()` for you; call `save()` yourself if you keep the sim alive.

## Gotchas

- **Import isaacgym before torch** — the one true footgun (see top).
- **`load_*_asset` before `add_env`**, or `KeyError` on the asset name.
- **`step` never advances physics** — it only sets targets/teleports. You must call
  `tick()` to move time forward. Forgetting `tick()` = a frozen world.
- **`step` needs `robot_vis` and `object_vis` keys present** (even as `{}`); `reset`
  needs `robot`/`robot_vis`/`object`. Passing the wrong set → `KeyError`.
- **Physical `object` isn't in `step`** — to reposition a physical object mid-run use
  `reset` (teleport); otherwise it just falls/collides under physics.
- **Multi-camera `load_camera(name, cam_param_dict)` is broken** — it references a
  nonexistent `self.env` and writes the wrong handle dict. Only the default
  (`camera_param_dict=None`) single-camera branch works. Details in
  [`internals.md`](internals.md).
- **State-save (`save_state`) is dead** — `set_savepath` is commented out, so
  `self.history` is never created and `save_stateinfo` would `AttributeError`. Roll your
  own logging with the getters above.
- **`visualize_camera` is deprecated and broken** (undefined `self.env`/`self.actor_handle`).
- **`destroy_env(name)` assumes video was set** — it dereferences `self.out[name]`; call
  it only on envs you gave a `set_videopath`, else prefer `terminate()`.
- **Single GPU, device 0, no CPU path** — `create_sim(0, 0, SIM_PHYSX, ...)` is hardcoded.
- Repo-wide typo `dataset_acqusition` (missing 'i') is intentional — unrelated to this
  module but don't "fix" it if you touch neighbouring imports.
