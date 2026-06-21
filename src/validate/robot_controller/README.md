# XArm Controller Validation

Exercises the low-level `XArmController` Cartesian move interface directly (bypassing the higher-level `get_arm` factory).

## Scripts
| File | Purpose |
|------|---------|
| `xarm_test.py` | Drives the XArm end-effector along the world Y axis, bouncing between ±0.5 m for 5 seconds using incremental Cartesian moves, to validate Cartesian servoing and position feedback. |

## Usage
```bash
python src/validate/robot_controller/xarm_test.py
```

Hardware required: **XArm arm via SDK** (`XArmController`, connected per `network_info['xarm']['param']`).

WARNING: this moves the end-effector ±0.5 m in Y. Make sure the workspace is clear before running. There is a commented-out "risky_target" block that jumps 0.5 m in one step — leave it disabled.

## What it validates
- Arm connects (`controller.connect_event.wait()` returns) and reports a valid `position` 4x4 pose.
- End-effector moves smoothly along Y and reverses direction at the ±0.5 m bounds.
- `controller.move()` accepts a 4x4 pose target and the arm tracks it.

## Related
- [`paradex/io/robot_controller/xarm_controller.py`](../../../paradex/io/robot_controller/xarm_controller.py) — `XArmController` (`get_data`, `move`, `connect_event`, `end`).
- [`paradex/utils/system.py`](../../../paradex/utils/system.py) — `network_info` (XArm connection params).
- For the higher-level `get_arm("xarm")` interface, see `src/validate/robot/xarm_base_wiggle.py`.
