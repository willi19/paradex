import numpy as np
import trimesh

from paradex.grasp.validation import (
    ArticulatedMeshModel,
    CallableTrajectory,
    GraspPhase,
    GraspSequenceValidator,
    GraspValidationConfig,
    LinkGeometry,
    PoseTrajectory,
    SampledTrajectory,
    ViolationCode,
)


def _translation(x=0.0, y=0.0, z=0.0):
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


def _two_finger_model(extra_links=(), **kwargs):
    finger_mesh = trimesh.creation.box(extents=(0.08, 0.02, 0.08))
    links = [
        LinkGeometry("left_finger", finger_mesh, group="hand"),
        LinkGeometry("right_finger", finger_mesh.copy(), group="hand"),
        *extra_links,
    ]

    def forward_kinematics(q):
        x = float(q[0])
        return {
            "palm": _translation(x=x),
            "left_finger": _translation(x=x, y=-0.06),
            "right_finger": _translation(x=x, y=0.06),
            **{
                link.name: _translation(x=x)
                for link in extra_links
            },
        }

    return ArticulatedMeshModel(
        links,
        forward_kinematics,
        disabled_self_collision_pairs=[("left_finger", "right_finger")],
        **kwargs,
    )


def _object_trajectory(x_positions):
    poses = np.repeat(np.eye(4)[None, ...], len(x_positions), axis=0)
    poses[:, 0, 3] = x_positions
    return PoseTrajectory([0.0, 1.0], poses)


def _phase():
    return GraspPhase(
        start_time=0.0,
        end_time=1.0,
        reference_link="palm",
        contact_links=("left_finger", "right_finger"),
        min_contact_links=2,
    )


def _config():
    return GraspValidationConfig(
        sample_dt=0.5,
        contact_distance=1e-6,
        forbidden_clearance=0.0,
        allowed_penetration=1e-6,
        max_relative_translation_error=0.01,
        max_relative_rotation_error=np.deg2rad(2.0),
        check_self_collision=False,
    )


def test_valid_grasp_keeps_two_contacts_and_relative_pose():
    robot_trajectory = SampledTrajectory([0.0, 1.0], [[0.0], [0.2]])
    object_trajectory = _object_trajectory([0.0, 0.2])
    validator = GraspSequenceValidator(
        _two_finger_model(),
        trimesh.creation.box(extents=(0.06, 0.1, 0.06)),
        _config(),
    )

    report = validator.validate(robot_trajectory, object_trajectory, [_phase()])

    assert report.valid
    assert len(report.sample_times) == 3
    assert all(
        contacts == ("left_finger", "right_finger")
        for contacts in report.contacts_by_time.values()
    )
    assert report.maximum_relative_translation_error < 1e-12


def test_callable_robot_parameter_trajectory_is_supported():
    robot_trajectory = CallableTrajectory(
        lambda time: np.array([0.2 * time]),
        start_time=0.0,
        end_time=1.0,
    )
    validator = GraspSequenceValidator(
        _two_finger_model(),
        trimesh.creation.box(extents=(0.06, 0.1, 0.06)),
        _config(),
    )

    report = validator.validate(
        robot_trajectory,
        _object_trajectory([0.0, 0.2]),
        [_phase()],
    )

    assert report.valid


def test_stationary_object_is_rejected_when_grasp_frame_moves():
    robot_trajectory = SampledTrajectory([0.0, 1.0], [[0.0], [0.2]])
    object_trajectory = _object_trajectory([0.0, 0.0])
    validator = GraspSequenceValidator(
        _two_finger_model(),
        trimesh.creation.box(extents=(0.06, 0.1, 0.06)),
        _config(),
    )

    report = validator.validate(robot_trajectory, object_trajectory, [_phase()])

    assert not report.valid
    assert report.violations_of(ViolationCode.RELATIVE_POSE_DRIFT)
    assert report.violations_of(ViolationCode.INSUFFICIENT_CONTACT)


def test_forbidden_arm_object_collision_is_reported():
    arm = LinkGeometry(
        "arm",
        trimesh.creation.box(extents=(0.04, 0.04, 0.04)),
        group="robot",
    )
    robot = _two_finger_model(extra_links=(arm,))
    validator = GraspSequenceValidator(
        robot,
        trimesh.creation.box(extents=(0.06, 0.1, 0.06)),
        _config(),
    )
    stationary = SampledTrajectory([0.0, 1.0], [[0.0], [0.0]])

    report = validator.validate(
        stationary,
        _object_trajectory([0.0, 0.0]),
        [],
    )

    collision_links = {
        violation.links[0]
        for violation in report.violations_of(
            ViolationCode.FORBIDDEN_OBJECT_COLLISION
        )
    }
    assert "arm" in collision_links


def test_joint_position_and_velocity_limits_are_checked():
    robot = _two_finger_model(
        joint_lower=[-0.1],
        joint_upper=[0.1],
        velocity_limits=[0.15],
    )
    validator = GraspSequenceValidator(
        robot,
        trimesh.creation.box(extents=(0.06, 0.1, 0.06)),
        _config(),
    )

    report = validator.validate(
        SampledTrajectory([0.0, 1.0], [[0.0], [0.2]]),
        _object_trajectory([0.0, 0.2]),
        [_phase()],
    )

    assert report.violations_of(ViolationCode.JOINT_LIMIT)
    assert report.violations_of(ViolationCode.JOINT_VELOCITY)


def test_coincident_link_meshes_are_reported_as_self_collision():
    link_mesh = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    robot = ArticulatedMeshModel(
        [
            LinkGeometry("link_a", link_mesh),
            LinkGeometry("link_b", link_mesh.copy()),
        ],
        lambda _q: {
            "link_a": np.eye(4),
            "link_b": np.eye(4),
        },
    )
    config = GraspValidationConfig(
        sample_dt=1.0,
        forbidden_clearance=0.0,
        collision_penetration_tolerance=0.0,
        check_self_collision=True,
    )
    validator = GraspSequenceValidator(
        robot,
        trimesh.creation.box(extents=(0.01, 0.01, 0.01)),
        config,
    )
    object_poses = np.repeat(_translation(x=1.0)[None, ...], 2, axis=0)

    report = validator.validate(
        SampledTrajectory([0.0, 1.0], [[0.0], [0.0]]),
        PoseTrajectory([0.0, 1.0], object_poses),
        [],
    )

    assert report.violations_of(ViolationCode.SELF_COLLISION)


def test_pose_trajectory_interpolates_translation_and_rotation():
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, np.pi],
        ]
    )
    trajectory = PoseTrajectory([0.0, 1.0], poses)

    middle = trajectory.at(0.5)

    np.testing.assert_allclose(middle[:3, 3], [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        middle[:3, :3],
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-12,
    )
