# CLAUDE.md — src/validate/robot

## Purpose
Hardware smoke-tests for robot hands (Allegro via ROS2, Inspire via USB Modbus and Modbus TCP/IP) and the XArm arm (SDK), plus a tactile/force GUI and a URDF→image overlay tuning tool.

## Files
- `allegro.py` — `get_hand("allegro")` → `hand.start("allegro_order_debug")`; loops 16 joints, ramps each `qpos[i]` 0→1→0 over 2s via `hand.move(qpos)`, then `hand.end()`. Bottom comments document joint index → finger mapping.
- `inspire.py` — `InspireController(**network_info['inspire_usb']['param'])` (USB Modbus). The motion block is commented out; active code is a `plt.ion()` loop reading `ic.get_data()['tactile_data']` and updating a 5x4 heatmap grid keyed by `SENSOR_LAYOUT`.
- `inspire_left.py` — `InspireControllerIP(**network_info['inspire'])` (Modbus TCP/IP). Runs a fixed pose list, then a 8s per-finger sinusoid, then HOME. Angle convention: 0=closed, 1000=open. Uses `ic.move()`, `ic.get_qpos()`, `ic.end()`.
- `inspire_left_gui.py` — `InspireControllerIP(**network_info['inspire'], tactile=True)`. matplotlib GridSpec UI: sliders → `target` → `ic.move()`; polls `ic.get_force()` (median-filtered bars + raw text), `ic.get_tactile()` (heatmaps via `TACTILE_LAYOUT`), `ic.get_qpos()` (status). "CALIBRATE FORCE" button → `ic.calibrate_force(timeout=15)`. ~20 Hz loop.
- `inspire_left_overlay.py` — biggest script. Captures (or `--load`s) a snapshot: remote multi-cam images + `arm.get_data()["qpos"]` (XArm) + `hand.get_qpos()` (inspire_left), plus `save_current_camparam`/`save_current_C2R`. Strips the `<mimic>` tag off `left_thumb_3/4` joints so they are independently sliderable, loads URDF with `RobotModule`, then per-frame: trackbars → `cfg`, `build_hand_groups` transforms meshes into camera-world via `c2r @ T_extra @ T_world_geom`, `ImageDict.project_mesh` overlays each finger group in its own BGR color, crops around hand centroid, merges to a grid.
- `xarm_base_wiggle.py` — `get_arm("xarm")`; `arm.connect_event.wait()`; reads home qpos, then 60 Hz loop sets `target[0] = home + amp*sin(...)`, `arm.move(target, is_servo=True)`. On exit returns home with `is_servo=False`, `arm.end(False)`.

## paradex modules used
- `paradex.io.robot_controller`: `get_arm`, `get_hand`.
- `paradex.io.robot_controller.inspire_controller`: `InspireController`, `SENSOR_LAYOUT` (USB).
- `paradex.io.robot_controller.inspire_controller_ip`: `InspireControllerIP`, `TACTILE_LAYOUT` (TCP/IP).
- `paradex.utils.system.network_info` — connection params (`inspire`, `inspire_usb`, `xarm`).
- `paradex.visualization.robot.RobotModule`, `paradex.image.image_dict.ImageDict`.
- `paradex.calibration.utils` — `load_camparam`, `load_current_camparam`, `load_c2r`, `load_current_C2R`, `save_current_camparam`, `save_current_C2R`, `get_cammtx`.
- `paradex.io.camera_system.remote_camera_controller.remote_camera_controller` (overlay capture).
- `paradex.utils.path` — `home_path`, `shared_dir`, `rsc_path`.

## Data flow & IO
- Inspire angle scale: 0=closed … 1000=open (six DOFs: little, ring, middle, index, thumb_bend, thumb_rot).
- `inspire_left_overlay.py` snapshot dir (default `~/shared_data/inspire_left_debug/<ts>/`): `arm_qpos.npy`, `hand_motor.npy`, `cam_param/`, `C2R.npy`, `images/<serial>.png`, `undistort/`, and on save `tuned.json` (cfg + wrist offset in mm). Capture PCs write under `~/<rel_path>/images`; main PC sees them via the shared mount.
- Motor→joint mapping in overlay: `MOTOR_ORDER` = [little, ring, middle, index, thumb_2(bend), thumb_1(rot)]; joint rad = `LEFT_LIMIT[jn] * (1 - motor/1000)`.

## When working here
- These are validation/debug scripts, not library code — keep them runnable as `python src/validate/robot/<x>.py`.
- Do NOT edit the `.py` files for documentation tasks.

## Gotchas
- `inspire.py` uses the **USB** controller (`inspire_usb`); `inspire_left*.py` use the **IP** controller (`inspire`). Different classes, different `network_info` keys.
- `inspire_left_overlay.py` hardcodes `URDF_SRC = /home/robot/shared_data/AutoDex/.../xarm_inspire_left.urdf` and writes a temp `.tmp_inspire_left_overlay.urdf` **next to the original** (so meshes resolve), removing it on exit. Path is machine-specific.
- Overlay strips only `left_thumb_3/4` mimic tags; other finger mimics stay. thumb_3/4 init from thumb_2 × 0.60 / 0.80.
- `allegro.py` end-of-file comments are an incomplete joint map (entries 0-4 only).
- `xarm_base_wiggle.py` relies on `arm.connect_event` and `arm.get_data()["qpos"]`; XArmController has no clean `.end()` in the overlay capture path (process exit cleans up).
