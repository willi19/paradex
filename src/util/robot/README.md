# Robot Utilities

Helpers for building combined arm+hand URDFs, inspecting collision geometry, and visualizing robot state in the web viewer.

## Scripts
| File | Purpose |
|------|---------|
| `merge_urdf.py` | Combines a separate arm URDF and hand URDF into one `<arm>_<hand>.urdf` via a xacro template, placing the hand at the correct wrist offset. |
| `visualize.py` | Open3D inspection of a single link's visual meshes plus its CuRobo collision spheres (currently hard-coded to `link4` of `xarm_allegro`). |
| `get_bounding_sphere.py` | Empty placeholder (0 bytes) — intended for fitting collision spheres to meshes. |
| `replay.py` | Streams the live arm's current qpos into the Viser viewer as a one-frame trajectory. |
| `replay_sim.py` | Empty placeholder (0 bytes). |

## Usage
```bash
# Build a combined URDF (writes rsc/robot/<arm>_<hand>.urdf)
python src/util/robot/merge_urdf.py --arm xarm --hand allegro

# Connect to the live arm and show its current pose in Viser
python src/util/robot/replay.py --arm xarm --hand allegro

# Inspect link meshes + collision spheres (edit link name / robot in source)
python src/util/robot/visualize.py
```

`visualize.py` controls: mouse to rotate/zoom, ESC or close window to exit. `replay.py` serves the Viser web viewer (open the printed URL).

> `get_bounding_sphere.py` and `replay_sim.py` are empty stubs — nothing to run.

## Inputs & Outputs
- `merge_urdf.py`: reads arm/hand URDFs (`get_robot_urdf_path`) and `rsc/robot/robot_combined.urdf.xacro`; uses `DEVICE2WRIST` transforms to compute the arm->wrist xyz/rpy; writes `rsc/robot/<arm>_<hand>.urdf`.
- `visualize.py`: reads the combined URDF (`get_robot_urdf_path(arm, hand)`), its link meshes, and `rsc/robot/spheres/<arm>_<hand>.yml` collision spheres. No file output (display only).
- `replay.py`: reads live arm qpos over the network; loads combined URDF into Viser. No file output.

## Related
- [`paradex/robot/robot_wrapper.py`](../../../paradex/robot/robot_wrapper.py) — `RobotWrapper` (FK, link poses, end links).
- [`paradex/robot/urdf.py`](../../../paradex/robot/urdf.py) — `generate_urdf` (xacro expansion).
- [`paradex/robot/curobo.py`](../../../paradex/robot/curobo.py) — CuRobo helpers (`to_quat`), collision spheres.
- [`paradex/io/robot_controller`](../../../paradex/io/robot_controller) — `get_arm`.
- [`paradex/visualization/visualizer/viser.py`](../../../paradex/visualization/visualizer/viser.py) — `ViserViewer` (`add_robot`, `add_traj`, `start_viewer`).
- `paradex.geometry.coordinate.DEVICE2WRIST` — device-to-wrist mounting transforms.
