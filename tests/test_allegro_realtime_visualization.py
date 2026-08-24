from types import SimpleNamespace
import time

import numpy as np

from paradex.visualization.allegro_realtime import (
    ALLEGRO_V5_TACTILE_VERTEX_IDS,
    ALLEGRO_V5_TIP_LINKS,
    ALLEGRO_V5_VISUAL_LINKS,
    DEFAULT_ALLEGRO_V5_URDF,
    AllegroRealtimeViser,
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
        "max_length": 0.1,
    }
    assert tactile_arrow_length(0.0, **kwargs) == 0.0
    assert tactile_arrow_length(-1.0, **kwargs) == 0.0
    assert tactile_arrow_length(1.0, **kwargs) > 0.0
    assert np.isclose(tactile_arrow_length(500.0, **kwargs), 0.05)
    assert tactile_arrow_length(2000.0, **kwargs) == 0.1


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


def test_feedback_pose_uses_named_v5_feedback_conversion_before_rendering():
    class RecordingViserRobot:
        def __init__(self):
            self.configurations = []

        def update_cfg(self, configuration):
            self.configurations.append(np.asarray(configuration, dtype=np.float64))

    studio = object.__new__(AllegroRealtimeViser)
    studio.viser_robot = RecordingViserRobot()
    studio.render_feedback_pose = True
    studio.urdf_joint_names = (
        "thumb_base", "thumb_proximal", "thumb_medial", "thumb_distal",
        "index_base", "index_proximal", "index_medial", "index_distal",
        "middle_base", "middle_proximal", "middle_medial", "middle_distal",
        "ring_base", "ring_proximal", "ring_medial", "ring_distal",
    )
    studio._sync_arrow_frames_to_feedback_pose = lambda: None

    assert studio._update_feedback_pose(
        {
            "qpos": np.arange(16, dtype=np.float64),
            "joint_names": tuple(f"joint_{index}_0" for index in range(16)),
        }
    )
    # V5 ROS names are index/middle/ring/thumb, but the Allegro URDF is
    # thumb/index/middle/ring: this is the same conversion used by the UI.
    np.testing.assert_allclose(
        studio.viser_robot.configurations[-1],
        np.r_[np.arange(12, 16), np.arange(0, 12)],
    )


def test_feedback_pose_preserves_native_standalone_urdf_joint_names():
    class RecordingViserRobot:
        def __init__(self):
            self.configuration = None

        def update_cfg(self, configuration):
            self.configuration = np.asarray(configuration, dtype=np.float64)

    studio = object.__new__(AllegroRealtimeViser)
    studio.viser_robot = RecordingViserRobot()
    studio.render_feedback_pose = True
    studio.urdf_joint_names = tuple(f"joint_{index}_0" for index in range(16))
    studio._sync_arrow_frames_to_feedback_pose = lambda: None
    feedback_names = tuple(reversed(studio.urdf_joint_names))
    feedback_values = np.arange(16, dtype=np.float64)

    assert studio._update_feedback_pose(
        {"qpos": feedback_values, "joint_names": feedback_names}
    )
    np.testing.assert_allclose(studio.viser_robot.configuration, feedback_values[::-1])


def test_viser_side_panel_render_is_bounded_and_converted_to_bgr():
    class FakeClient:
        def __init__(self):
            self.request = None

        def get_render(self, **kwargs):
            self.request = kwargs
            return np.array([[[10, 20, 30]]], dtype=np.uint8)

    client = FakeClient()
    studio = object.__new__(AllegroRealtimeViser)
    studio._initialize_render_state()
    studio.viewer = SimpleNamespace(
        server=SimpleNamespace(get_clients=lambda: {0: client})
    )

    image = studio.render_bgr(height=1230, width=1298)
    deadline = time.monotonic() + 1.0
    while image is None and time.monotonic() < deadline:
        time.sleep(0.01)
        image = studio.render_bgr(height=1230, width=1298)

    assert client.request == {
        "height": 480,
        "width": 507,
        "transport_format": "jpeg",
    }
    assert image.tolist() == [[[30, 20, 10]]]
