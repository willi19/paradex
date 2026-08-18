import numpy as np

from paradex.visualization.allegro_realtime import (
    ALLEGRO_V5_TACTILE_VERTEX_IDS,
    ALLEGRO_V5_TIP_LINKS,
    ALLEGRO_V5_VISUAL_LINKS,
    DEFAULT_ALLEGRO_V5_URDF,
    allegro_tactile_finger_levels,
    canonical_allegro_mesh,
    centered_robot_offset,
    fingertip_surface_arrow_frame,
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


def test_tactile_arrow_length_shows_every_positive_value_and_is_bounded():
    kwargs = {
        "display_max": 1000.0,
        "max_length": 0.2,
    }
    assert tactile_arrow_length(0.0, **kwargs) == 0.0
    assert tactile_arrow_length(-1.0, **kwargs) == 0.0
    assert tactile_arrow_length(1.0, **kwargs) > 0.0
    assert np.isclose(tactile_arrow_length(500.0, **kwargs), 0.1)
    assert tactile_arrow_length(2000.0, **kwargs) == 0.2


def test_v5_mesh_center_is_translated_to_the_world_origin():
    robot = RobotModule(DEFAULT_ALLEGRO_V5_URDF)
    mesh_center = np.asarray(canonical_allegro_mesh(robot).bounding_box.centroid)

    np.testing.assert_allclose(mesh_center + centered_robot_offset(robot), 0.0)


def test_v5_visual_meshes_do_not_match_the_inspire_hand_only_filter():
    robot = RobotModule(DEFAULT_ALLEGRO_V5_URDF)
    mesh_names = tuple(robot.scene.geometry)

    assert mesh_names
    assert not any(
        "/left_hand_" in name or "/right_hand_" in name
        for name in mesh_names
    )


def test_v5_visual_allowlist_contains_every_allegro_mesh_parent():
    robot = RobotModule(DEFAULT_ALLEGRO_V5_URDF)
    mesh_parent_links = {
        robot.scene.graph.transforms.parents[name]
        for name in robot.scene.geometry
    }

    assert mesh_parent_links == ALLEGRO_V5_VISUAL_LINKS


def test_v5_tactile_arrows_are_anchored_on_fingertip_mesh_surfaces():
    robot = RobotModule(DEFAULT_ALLEGRO_V5_URDF)
    assert ALLEGRO_V5_TACTILE_VERTEX_IDS == (1783, 1601, 2222, 1588)

    for link_name in ALLEGRO_V5_TIP_LINKS.values():
        anchor, normal = fingertip_surface_arrow_frame(robot, link_name)
        assert np.all(np.isfinite(anchor))
        assert np.all(np.isfinite(normal))
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1.0e-6)
