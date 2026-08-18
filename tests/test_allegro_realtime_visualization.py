import numpy as np

from paradex.visualization.allegro_realtime import (
    ALLEGRO_V5_TACTILE_VERTEX_IDS,
    ALLEGRO_V5_TIP_LINKS,
    ALLEGRO_V5_JOINT_NAMES,
    DEFAULT_ALLEGRO_V5_URDF,
    allegro_tactile_finger_levels,
    centered_robot_offset,
    fingertip_surface_arrow_frame,
    named_allegro_qpos,
    tactile_arrow_length,
)
from paradex.visualization.robot import RobotModule


def test_allegro_tactile_levels_split_four_contiguous_finger_blocks():
    tactile = np.array(
        [1, -9, 3, 4, 10, 11, 12, 13, 20, -29, 22, 23, 30, 31, 32, -39]
    )

    assert allegro_tactile_finger_levels(tactile) == {
        "index": 9.0,
        "middle": 13.0,
        "ring": 29.0,
        "thumb": 39.0,
    }


def test_allegro_tactile_levels_reject_malformed_packets():
    assert allegro_tactile_finger_levels(None) is None
    assert allegro_tactile_finger_levels([1, 2, 3]) is None
    assert allegro_tactile_finger_levels([1, 2, 3, np.nan]) is None


def test_named_allegro_qpos_reorders_ros_feedback_by_joint_name():
    names = tuple(reversed(ALLEGRO_V5_JOINT_NAMES))
    values = np.arange(16, dtype=float)
    expected_by_name = dict(zip(names, values))

    actual = named_allegro_qpos(values, names)

    assert actual == {
        name: expected_by_name[name]
        for name in ALLEGRO_V5_JOINT_NAMES
    }


def test_named_allegro_qpos_rejects_missing_or_nonfinite_feedback():
    assert named_allegro_qpos(np.zeros(15), ALLEGRO_V5_JOINT_NAMES[:-1]) is None
    invalid = np.zeros(16)
    invalid[4] = np.inf
    assert named_allegro_qpos(invalid, ALLEGRO_V5_JOINT_NAMES) is None


def test_tactile_arrow_length_has_noise_floor_and_bounded_scale():
    kwargs = {
        "threshold": 200.0,
        "display_max": 5000.0,
        "max_length": 0.06,
    }
    assert tactile_arrow_length(199.0, **kwargs) == 0.0
    assert tactile_arrow_length(200.0, **kwargs) == 0.0
    assert np.isclose(tactile_arrow_length(2600.0, **kwargs), 0.03)
    assert tactile_arrow_length(6000.0, **kwargs) == 0.06


def test_v5_mesh_center_is_translated_to_the_world_origin():
    robot = RobotModule(DEFAULT_ALLEGRO_V5_URDF)
    mesh_center = np.asarray(robot.get_robot_mesh().bounding_box.centroid)

    np.testing.assert_allclose(mesh_center + centered_robot_offset(robot), 0.0)


def test_v5_tactile_arrows_are_anchored_on_fingertip_mesh_surfaces():
    robot = RobotModule(DEFAULT_ALLEGRO_V5_URDF)
    assert ALLEGRO_V5_TACTILE_VERTEX_IDS == (1783, 1601, 2222, 1588)

    for link_name in ALLEGRO_V5_TIP_LINKS.values():
        anchor, normal = fingertip_surface_arrow_frame(robot, link_name)
        assert np.all(np.isfinite(anchor))
        assert np.all(np.isfinite(normal))
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1.0e-6)
