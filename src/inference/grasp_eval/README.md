# grasp_eval

Grasp evaluation pipelines for the pringles can. Estimates the object's 6D pose from
the camera rig and evaluates a grasp (from DexGraspNet) by visualizing it or executing
it on the physical robot.

## Contents
| Dir | Purpose |
|-----|---------|
| [`real/`](real/) | Run grasp evaluation on the real XArm + Allegro robot using DexGraspNet results. |

## Usage
See [`real/README.md`](real/README.md). Typical order is `object6d.py` (capture + pose +
launch GUI to execute) and `visualize.py` (Viser preview of the grasp).

## Inputs & Outputs
- Grasps: `dexgraspnet/results/pringles/<index>/{qpos.npy, wrist_6d.npy}`.
- Marker offsets: `~/shared_data/object/marker_offset/pringles/0/marker_offset.npy`.
- Object mesh: `rsc/object/pringles/pringles.obj`.
- Captures images under `~/shared_data/inference/grasp_eval/<timestamp>/`.

## Related
- Sibling grasp sources: [`../bodex/`](../bodex/) and [`../grasp_w_gui/`](../grasp_w_gui/) use BODex grasps instead of DexGraspNet.
- [`paradex/robot/robot_wrapper.py`](../../../paradex/robot/robot_wrapper.py), [`paradex/io/robot_controller/`](../../../paradex/io/robot_controller/)
