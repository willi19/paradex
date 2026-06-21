# CLAUDE.md — src/validate/robot_controller

## Purpose
Direct smoke-test of the low-level `XArmController` Cartesian interface (instantiated directly, not via `get_arm`).

## Files
- `xarm_test.py` — `XArmController(**network_info["xarm"]["param"])`; `connect_event.wait()`; for 5s loops: reads `get_data()["position"]` (4x4 pose), adds `direction * speed * elapsed/1000` to the translation, calls `controller.move(pose)`; flips `direction` when `position[1,3]` crosses ±0.5. Ends with `controller.end(False)`.

## paradex modules used
- `paradex.io.robot_controller.xarm_controller.XArmController` — `get_data()` (returns dict with `position` 4x4 and `qpos`), `move(pose)`, `connect_event`, `end(bool)`.
- `paradex.utils.system.network_info` — `network_info["xarm"]["param"]`.
- Imports `save_current_camparam`, `get_handeye_calib_path`, `run_script` are present but unused in the active code path.

## Data flow & IO
- `get_data()["position"]` is a 4x4 homogeneous pose; translation in meters at `[:3, 3]`. `move` takes the same 4x4 form.
- No files written. Pure motion test.

## When working here
- Validation script; keep runnable as `python src/validate/robot_controller/xarm_test.py`.
- Do NOT edit the `.py` for doc tasks.

## Gotchas
- The commented `risky_target` block does a single 0.5 m jump — intentionally disabled; do not enable casually.
- `speed = 700 mm/s` and the `/1000` scaling mean the per-step increment is in meters; the loop is timing-dependent (`elapsed` from previous iteration), so motion magnitude depends on loop rate.
- Several imports (`os`, `save_current_camparam`, `get_handeye_calib_path`, `run_script`) are dead — leftover scaffolding.
