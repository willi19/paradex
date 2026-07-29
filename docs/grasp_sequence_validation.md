# Grasp sequence validation

`paradex.grasp` checks the geometric and kinematic consistency of synchronized
robot/hand and object trajectories. All transforms use `world_T_local`, all
distances are metres, and both trajectories must share the same clock.

```python
import numpy as np
import trimesh

from paradex.grasp import (
    ArticulatedMeshModel,
    GraspPhase,
    GraspSequenceValidator,
    GraspValidationConfig,
    PoseTrajectory,
    SampledTrajectory,
)
from paradex.visualization.robot import RobotModule

robot_module = RobotModule("rsc/robot/xarm_allegro_v5.urdf")

# Use collision_geometry=True when the URDF actually contains collision meshes.
# Parent-child pairs are ignored automatically.
robot = ArticulatedMeshModel.from_robot_module(
    robot_module,
    hand_links={
        name
        for name in robot_module.urdf.link_map
        if name == "palm_link" or name.startswith("link_")
    },
    collision_geometry=False,
    # Add intentional non-parent/child contact pairs here when needed.
    disabled_self_collision_pairs=(),
)

times = np.load("times.npy")                    # (N,), seconds
q = np.load("robot_hand_q.npy")                 # (N, D)
world_T_object = np.load("object_poses.npy")     # (N, 4, 4)
object_mesh = trimesh.load("object.obj", force="mesh")

robot_trajectory = SampledTrajectory(times, q)
object_trajectory = PoseTrajectory(times, world_T_object)

grasp = GraspPhase(
    start_time=2.1,
    end_time=5.8,
    reference_link="palm_link",
    contact_links=(
        "link_3_0",
        "link_7_0",
        "link_11_0",
        "link_15_0",
    ),
    min_contact_links=2,
)

validator = GraspSequenceValidator(
    robot,
    object_mesh,
    GraspValidationConfig(
        sample_dt=0.02,
        contact_distance=0.003,
        forbidden_clearance=0.001,
        allowed_penetration=0.002,
        collision_penetration_tolerance=0.0005,
        max_relative_translation_error=0.01,
        max_relative_rotation_error=np.deg2rad(10),
    ),
)
report = validator.validate(
    robot_trajectory,
    object_trajectory,
    [grasp],
)

print("valid:", report.valid)
for violation in report.violations:
    print(violation.time, violation.code.value, violation.message)
```

For analytic trajectories, replace `SampledTrajectory` with:

```python
from paradex.grasp import CallableTrajectory

robot_trajectory = CallableTrajectory(
    evaluator=lambda t: q_of_t(t),
    start_time=0.0,
    end_time=8.0,
)
```

`PoseTrajectory` also accepts `(N, 6)` values
`[x, y, z, rotation-vector]`, or `(N, 7)` values
`[x, y, z, qx, qy, qz, qw]`.

## What is checked

- robot/hand joint position and velocity limits when supplied;
- robot/hand self-collision, excluding configured link pairs;
- object collision with links that are not allowed to contact it;
- required hand-object contact count during each grasp phase;
- excessive hand-object penetration;
- constancy of `reference_T_object` during a grasp phase.

The mesh test uses bounded deterministic surface sampling, so lowering
`sample_dt` and increasing `max_geometry_samples` makes the check stricter at a
runtime cost. It is not a continuous collision proof. Mesh and trajectory data
alone also cannot establish force closure or dynamic stability; those require
friction, mass/inertia, contact force, and actuator models.
