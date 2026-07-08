# IsaacGym Simulator — Internals (for agents editing this module)

**You are here because you are changing how the wrapper builds the sim / creates actors
/ steps physics / renders — not just calling it.** To only *drive* the sim, read
[`usage.md`](usage.md). Everything below is the single class `Simulator` in
[`isaac.py`](../../paradex/simulator/isaac.py) (~612 lines); there is no other file.

There is **no test and no in-repo consumer** (grep of `src/` + `paradex/` is clean).
No call site pins the API, and the broken paths below have never been exercised. Prefer
to *fix* a dead path over inventing a new one.

---

## 1. Object model — four parallel worlds

Everything lives in four-key dicts, all keyed the same way, so loops stay uniform:

```
assets            = {"robot":{}, "robot_vis":{}, "object":{}, "object_vis":{}}   # name -> gym asset (template)
env_dict          = {env_name: gym env}
actor_handle_dict = {env_name: {"robot":{}, "robot_vis":{}, "object":{}, "object_vis":{}}}  # instances
camera_handle     = {env_name: {cam_name: handle}}     # default path only; see §7
out               = {env_name: {cam_name: cv2.VideoWriter}}
```

`add_env` is the only thing that turns templates into instances. `num_envs` is a counter
bumped at the *top* of `add_env`; its current value doubles as the **collision group** of
physical actors (§4). `_vis` actors are **ghosts**: gravity disabled per rigid body,
`driveMode=DOF_MODE_NONE`, tint `(0.4,0.4,0.6)`, high collision groups (1000/1001) so they
never collide with the physical world — they exist to be teleported to a reference pose.

---

## 2. Sim creation flow

```
__init__(headless, plane)
  gym = acquire_gym(); sim = generate_sim()   # create_sim(0,0, SIM_PHYSX, params)
  if plane: add_plane()                        # z-up ground, distance=0.0525, friction 1.0/0.8
  if not headless: set_viewer()                # GLFW window + camera lookat
  asset_root = rsc_path; init the four dicts; env bounds (spacing 1.5); flags
```

`generate_sim` hardcodes physics (`dt=1/30`, `substeps=2`, up=Z, gravity `-9.8`, PhysX GPU
`solver_type=1`, 6 pos / 1 vel iters, `contact_offset=0.01`). **`create_sim(0, 0, SIM_PHYSX, ...)`**
= compute device 0, graphics device 0, no CPU pipeline, no device arg — changing GPU means
editing this line. State getters (§5) use the **per-actor** API, not the flat GPU tensor
API, so they read correctly despite `use_gpu=True`. `set_viewer(ext_mat)` takes a `4x4`
extrinsic (default cam at `(2,0,1)`), `lookat` hardcoded to origin.

---

## 3. Asset loading

`load_robot_asset` / `load_object_asset` each compile **two** assets from the *same* URDF
— a physical and a `_vis` — differing only in `AssetOptions`:

| | Physical | `_vis` |
|--|----------|--------|
| robot | `fix_base_link=True`, `armature=0.001`, `thickness=0.002` | `fix_base_link=True` |
| object | `override_inertia=True`, `mesh_normal_mode=COMPUTE_PER_VERTEX`, `vhacd_enabled=True`, `vhacd_params.resolution=300000` | `disable_gravity=True` |

- **URDF selection** (`load_robot_name`): arm set → `robot/{arm}_{hand}.urdf`; arm `None`
  → `robot/{hand}_float.urdf`. Stored under `robot_name` (`{arm}_{hand}`, or hand when arm
  `None`); `add_env` must look it up by the *same* `(arm,hand)` tuple.
- **VHACD at `resolution=300000` is slow** on first load (minutes for complex meshes),
  cached per process — expect a first-`add_env` stall.
- Object `_vis` disables gravity at *asset* level; robot `_vis` does it later at
  *rigid-body* level in `load_vis_robot_actor` — two mechanisms, same intent.

---

## 4. Actor creation (from `add_env`)

```
num_envs += 1                                       # BEFORE actors -> first env is group 1, not 0
env = create_env(sim, env_lower, env_upper, 5)      # 5 actors per row
actor_handle[k] = {n: load_*_actor(env, n, ...) for n,... in env_info[k].items()}  # for each of 4 keys
env_dict[name]=env; actor_handle_dict[name]=actor_handle
```

`create_actor(env, asset, pose, name, collision_group, collision_filter)` — the ints matter:

| Loader | group | filter | driveMode | Notable |
|--------|-------|--------|-----------|---------|
| `load_robot_actor` | `num_envs` | 0 | `DOF_MODE_POS` **only if `arm=="xarm"`** | xarm gains: stiffness `[:6]=1000 / [6:]=500`, damping `10` |
| `load_vis_robot_actor` | 1000 | 1 | `DOF_MODE_NONE` | all bodies gravity-off, color |
| `load_object_actor` | `num_envs` | 0 | — | shape `restitution=0.01`, `friction=0.8` |
| `load_vis_object_actor` | 1001 | 0 | `DOF_MODE_NONE` | body0 gravity-off + color |

Traps: **only `xarm` gets PD drive configured** — add a gain block for any new robot type,
else `set_actor_dof_position_targets` does nothing. **Robot base is fixed** (`fix_base_link`)
→ move robots by DOF only; objects are free bodies spawned at temp `(0.5,0,0)`.

---

## 5. State API — per-actor, not tensor

The wrapper uses IsaacGym's **non-tensor** per-actor getters/setters everywhere (never
`acquire_*_state_tensor` / `refresh_*` / `set_*_tensor`) — CPU-readable but **O(actors)
Python calls/frame**, a rewrite for thousands of envs. State structs are numpy structured
arrays: DOF → `["pos"]`,`["vel"]`; rigid body → `["pose"]["p"]["x"|"y"|"z"]` and
`["pose"]["r"]["x"|"y"|"z"|"w"]` (quat **xyzw**).

| Writer | Effect | Used in |
|--------|--------|---------|
| `set_actor_dof_position_targets` | PD **target** (reached over ticks) | `step` physical robot |
| `set_actor_dof_states(..., STATE_POS)` | **teleport** DOF (instant) | `step` robot_vis, `reset` robot/robot_vis |
| `set_actor_rigid_body_states(..., STATE_POS)` | **teleport** body pose | `step` object_vis, `reset` object |

Teleports are read-modify-write: get the struct, overwrite `["pos"]` or `["pose"]`, set
back. Rotation goes `4x4` → `R.from_matrix(...).as_quat()` (xyzw) → `.fill((x,y,z,w))`.

---

## 6. The step / tick loop

**`step(name, action_dict)`** writes only, never advances time. Requires keys `robot`,
`robot_vis`, `object_vis`; **does not touch physical `object`** (physics owns it):
```
if save_state: save_stateinfo(name)                    # DEAD, see §8
action_dict["robot"]      -> set_actor_dof_position_targets   # PD
action_dict["robot_vis"]  -> read dof; ["pos"]=state; set_actor_dof_states       # teleport
action_dict["object_vis"] -> read rb; pose=quat/pos; set_actor_rigid_body_states # teleport
```

**`tick()`** is the clock:
```
simulate(sim); fetch_results(sim, True)      # one dt = 1/30 s (substeps=2)
step_graphics(sim)
if not headless: draw_viewer; sync_frame_time
if save_video: render_all_camera_sensors; per cam: get_camera_image(IMAGE_COLOR)
               .reshape(H,W,4)[:,:,:3][:,:,::-1] -> writer.write   # RGBA->RGB->BGR, 2048x1536
```

**`reset(name, action_dict)`** teleports everything (keys `robot`,`robot_vis`,`object` —
note **no `object_vis`**, asymmetric with `step`). Use it to place the world before the
first `tick`.

---

## 7. Rendering / camera paths

- **`load_camera(name, None)`** works: one default camera (`fov=75`, `2048x1536`,
  `(1.5,0,1.5)`→`(0,0,0.3)`) stored at `camera_handle[name]["default"]`.
- **The `camera_param_dict`-given branch is BROKEN**: it uses `create_camera_sensor(self.env, ...)`
  and `set_camera_location(..., self.env, ...)` but **`self.env` doesn't exist** (only
  `env_dict`), and writes `camera_handle[serial_num]=handle` (bare handle) instead of
  `camera_handle[name][serial_num]`, which then breaks `tick`'s nested loop. To use real
  intrinsics, fix both. The FOV math (`arctan(2048/(2*fx))*2`) is fine.
- **`set_videopath(env_name, path)`** makes one `cv2.VideoWriter` (`mp4v`, 30 fps,
  `2048x1536`) per camera at `path/env_name/{cam}.mp4`, sets `save_video=True`.
- **`visualize_camera` is deprecated/dead** — undefined `self.actor_handle`/`self.env`.

---

## 8. Teardown & the dead state-save path

- **`save()`** releases all writers (`save_video→False`) and, *if* `save_state`,
  pickles `self.history`.
- **`terminate()`** = `save()` + destroy every env + destroy viewer (if not headless) +
  `destroy_sim` — the normal exit.
- **`destroy_env(name)`** dereferences `self.out[name]` first → `KeyError` on an env that
  never got `set_videopath`. Prefer `terminate()` unless video was configured.
- **State save is entirely dead:** `set_savepath` (which would set `save_state=True`,
  `self.history`, `self.state_path`) is **commented out** → `save_state` stays `False`,
  `save_stateinfo` (called by `step`) would `AttributeError` on `self.history`, and
  `save()`'s pickle branch never runs. `save_stateinfo` has its own bug too: it passes
  `actor_handle["robot"]` (a dict) to `get_actor_dof_states` instead of the actor. To log
  state, either fix both or log externally with the §5 getters.

---

## 9. Bug vs. not-a-bug

| Symptom | Bug? | Note |
|---------|------|------|
| `PyTorch was imported before isaacgym` | caller | import `isaacgym`/`paradex.simulator` before `torch`. |
| First `add_env` hangs minutes | no | object VHACD `resolution=300000`. |
| First env is collision group `1` | no | `num_envs++` runs before actor creation. |
| Non-xarm robot ignores DOF targets | **yes** | only `xarm` gets `DOF_MODE_POS` + gains (§4). |
| `load_camera(name, dict)` crashes / no video | **yes** | undefined `self.env`, wrong key (§7). |
| `save_stateinfo` `AttributeError` | **yes** | `set_savepath` commented out (§8). |
| `visualize_camera` `AttributeError` | **yes** | deprecated dead method. |
| World never moves | caller | `step` only targets/teleports; you must call `tick()`. |

Repo-wide intentional typo `dataset_acqusition` (missing 'i') is unrelated to this file
but appears across the codebase — never "fix" it.
