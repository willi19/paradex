# agent_docs/simulator — agent orientation

Docs for **AI agents working on `paradex/simulator/`** — a thin wrapper around **NVIDIA
IsaacGym** (PhysX GPU physics) that loads paradex robot/object URDFs from
[`rsc/`](../../rsc/), builds one or more environments, drives DOFs, and (optionally)
renders offscreen video. One class, one file: `Simulator` in
[`isaac.py`](../../paradex/simulator/isaac.py), re-exported as `IsaacSimulator`
by [`__init__.py`](../../paradex/simulator/__init__.py).

```python
from paradex.simulator import IsaacSimulator     # == Simulator
```

Mental model: **build the world once, then hand-crank the clock.**
`Simulator(headless)` grabs the gym + creates the PhysX sim. You (1) **load assets**
(`load_robot_asset`, `load_object_asset` — compile URDF → asset templates), (2)
**instantiate envs** (`add_env` — spawns actors from those assets into a new env),
then loop: **`step(name, action_dict)`** writes DOF targets / teleports, **`tick()`**
advances physics one `dt` and renders. Read state back via the actor DOF / rigid-body
getters (see [`internals.md`](internals.md) §state). `terminate()` tears it all down.

There is **no daemon thread and no hardware** here — unlike
[`robot_controller/`](../robot_controller/README.md), everything runs synchronously in
your calling thread. This is the *physics* counterpart to the *analytic* kinematics in
[`robot/`](../robot/README.md) (`RobotWrapper` FK/IK); they share the same URDFs but do
not share code — pick physics (contacts, gravity, PD control) here, closed-form FK/IK
there.

> **Reality check:** nothing in `src/` or the rest of `paradex/` imports this module
> today (grep is clean). Treat it as a **standalone / experimental** rig — there is no
> in-repo call site to copy, and several methods are dead or broken (flagged in
> [`internals.md`](internals.md)). The working path is the one in
> [`usage.md`](usage.md); everything else, verify against the source before trusting.

| Your task | Read |
|-----------|------|
| **Use** the sim: build a world, drive a robot, record video, read state | [`usage.md`](usage.md) |
| **Edit** the wrapper: sim/env/actor creation, the tensor/state API, the step loop, traps | [`internals.md`](internals.md) |
| Get URDFs / IK / FK to feed `step()` (no physics) | [`robot/`](../robot/README.md) |
| Understand the merged URDFs (`xarm_allegro.urdf`, `{hand}_float.urdf`) | [`rsc/robot/`](../../rsc/robot/) + [`src/util/robot/`](../../src/util/robot/CLAUDE.md) |

Rule of thumb: **calling** the sim → `usage.md`; **changing what it does** → `internals.md`.

## Lifecycle at a glance

```
IsaacSimulator(headless)          # acquire gym + create PhysX sim (device 0) + ground plane
  └ load_robot_asset(arm, hand)   # compile rsc/robot/*.urdf -> reusable asset template (+ _vis)
  └ load_object_asset(obj)        # compile rsc/object/<obj>/<obj>.urdf (+ _vis)
  └ add_env(name, env_info)       # create env + spawn actors from the loaded assets
  └ reset(name, action_dict)      # teleport everything to a start pose (optional)
  ┌ loop:
  │   step(name, action_dict)     # set robot PD targets / teleport vis + object_vis
  │   tick()                      # advance one dt (1/30 s), render, write video frame
  └ terminate()                   # save() video, destroy envs/viewer/sim
```

`step` **writes**, `tick` **advances the clock** — they are separate on purpose; a loop
that calls `step` but never `tick` produces a frozen world. Time granularity is fixed at
`dt = 1/30 s` with `substeps = 2` (edit `generate_sim` to change it — no config hook).

## The two-actor pattern (understand this before anything else)

Every "thing" in the world can exist twice, in parallel dicts:

| Kind | Asset dict | Driven by | Gravity | Collision | Purpose |
|------|-----------|-----------|---------|-----------|---------|
| **robot** / **object** | `assets["robot"]`, `["object"]` | PhysX (PD targets / free dynamics) | on (robot base fixed) | real | the *simulated* body |
| **robot_vis** / **object_vis** | `assets["robot_vis"]`, `["object_vis"]` | teleport (`set_*_states`) | **off** | separate group | a *ghost overlay* — where you *want* it, colored `(0.4,0.4,0.6)` |

`step()` sends **position targets** to the physical robot (it obeys PD gains + physics)
but **teleports** the vis actors to an exact pose each frame. Use the physical actor to
simulate; use the vis actor to draw a reference/target that ignores physics. You do not
have to create both — `add_env` only spawns what you list in `env_info`.

## File map

| File | What it is |
|------|-----------|
| [`__init__.py`](../../paradex/simulator/__init__.py) | One line: `from .isaac import Simulator as IsaacSimulator`. |
| [`isaac.py`](../../paradex/simulator/isaac.py) | The whole module (~612 lines): the `Simulator` class — sim setup, asset loading, env/actor creation, `step`/`tick`/`reset`, offscreen camera video, save/teardown. |

Assets it consumes (paths relative to `rsc_path` = [`rsc/`](../../rsc/)):

| `env_info` key | URDF path template | Example |
|----------------|--------------------|---------|
| robot (arm+hand) | `robot/{arm}_{hand}.urdf` | `robot/xarm_allegro.urdf` |
| robot (hand only, arm `None`) | `robot/{hand}_float.urdf` | `robot/allegro_float.urdf` |
| object | `object/{obj}/{obj}.urdf` | `object/pringles/pringles.urdf` |

## The single most important setup caveat

**IsaacGym must be imported before `torch`.** [`isaac.py`](../../paradex/simulator/isaac.py)
does `from isaacgym import gymapi` at module top and imports **no** torch — but the moment
you combine this module with any torch-using paradex code (retargeting, CuRobo, learning),
**`import paradex.simulator` / `import isaacgym` has to come first in your entry script**,
or IsaacGym aborts with `PyTorch was imported before isaacgym modules`. See
[`usage.md`](usage.md) for the exact import block. Also: PhysX runs on **GPU device 0**,
hardcoded (`create_sim(0, 0, SIM_PHYSX, ...)`) — no CPU fallback, no device selection.
