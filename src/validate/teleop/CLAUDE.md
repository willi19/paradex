# CLAUDE.md — src/validate/teleop

## Purpose
Connectivity/visualization checks for teleop motion-capture input: XSens body suit or Oculus hand tracking streamed into a `HandVisualizer` skeleton.

## Files
- `io_connection.py` — `--device {xsens,occulus}`. XSens path: `XSensReceiver(**network_info['xsens']['param'])`. Oculus path: `OculusReceiver()`. Builds `skeleton_info` (child→parent), creates `HandVisualizer`, registers `q`→stop via `listen_keyboard`. Calls `receiver.start("tmp/hand_pose")`, then loops `receiver.get_data()` → recenter on first-frame wrist midpoint → `visualizer.update_sphere_positions(hand_data)` at ~50 Hz. On exit: `receiver.stop()`, `visualizer.stop()`, `receiver.end()`.
- `xsens_body.py` — near-identical, but XSens path uses `XSensReceiver()` (no params), there is **no** `receiver.start()/stop()` (no recording), and teardown uses `visualizer.stop()` + `receiver.quit()`.

## paradex modules used
- `paradex.io.teleop`: `XSensReceiver`, `xsens_joint_name`, `xsens_joint_parent_name`; `OculusReceiver`, `occulus_hand_joint_name`, `occulus_hand_joint_parent_name`.
- `paradex.visualization.skeleton.hand.HandVisualizer`.
- `paradex.utils.keyboard_listener.listen_keyboard`.
- `paradex.utils.system.network_info` (io_connection.py only).

## Data flow & IO
- `get_data()` returns `{'Left': {...} | None, 'Right': {...} | None}`; each joint value is a 4x4 pose, translation at `[:3, 3]`.
- Recentering: subtract first valid frame's wrist-midpoint `(Right['wrist'][:3,3] + Left['wrist'][:3,3]) / 2` from every joint translation.
- `io_connection.py` records to `tmp/hand_pose`; `xsens_body.py` records nothing.

## When working here
- Validation scripts; keep runnable as `python src/validate/teleop/<x>.py --device xsens|occulus`.
- Do NOT edit the `.py` for doc tasks.

## Gotchas
- `occulus` (double-misspelled) is the literal choice value AND the module symbol prefix — match it exactly, do not "fix".
- XSens receiver is constructed **with** params in `io_connection.py` but **without** params in `xsens_body.py`.
- Teardown differs: `io_connection.py` → `receiver.end()`; `xsens_body.py` → `receiver.quit()`.
- Loop skips frames where either hand is `None` (`continue`), so the visualizer only updates when both hands are present.
- XSens is a teleop motion-capture suit (not a camera/robot); these scripts do not touch any robot hardware.
