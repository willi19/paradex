import json
import numpy as np
import pytest
import time
from threading import Event, Lock
from types import SimpleNamespace

from paradex.retargetor.allegro_alignment import (
    ALLEGRO_FEEDBACK_JOINT_NAMES,
    ALLEGRO_RETARGETER_JOINT_NAMES,
    ALLEGRO_URDF_JOINT_NAMES,
    ALLEGRO_V5_DRIVER_JOINT_NAMES,
    ALLEGRO_V5_FEEDBACK_JOINT_NAMES,
    feedback_to_urdf_qpos,
    retargeter_action_to_controller_qpos,
    retargeter_action_to_live_controller_qpos,
    retargeter_action_to_urdf_qpos,
    urdf_qpos_from_hand_qpos,
    urdf_qpos_to_retargeter_action,
)
from paradex.retargetor.inspire_f1_alignment import (
    F1_RIGHT_QPOS_UPPER,
    F1_RIGHT_RAW_CLOSE,
    F1_RIGHT_RAW_OPEN,
    F1_RIGHT_WIRE_JOINT_NAMES,
    HandStateContractError,
    master_qpos_to_raw_f1,
    raw_f1_to_master_qpos,
    retargeter_action_to_raw_f1,
    urdf_qpos_from_master_qpos,
)
from paradex.io.teleop.vive.receiver import ViveManusROSReceiver
from paradex.retargetor.hand_regargetor import (
    _align_allegro_v5_action,
    _align_wuji_direct_action,
    _cluster_balanced_distance_gated_target,
    _distance_gated_weights,
    _allegro_v5_manus_ergonomic_feature,
    _allegro_v5_raw_from_manus_ergonomics,
    _load_allegro_v5_direct_anchor_data,
    _load_wuji_direct_anchor_data,
    allegro_v5,
    _wuji_direct_tip_feature,
    clip_wuji_direct_safe_action,
    clip_allegro_v5_safe_action,
)
from paradex.retargetor.unimanual import Retargetor
from src.dataset_acquisition.hri.allegro_retargeter_alignment_ui import (
    AllegroRetargeterAlignmentStudio,
    _alignment_samples,
    _delete_alignment_sample,
    _live_retargeter_kwargs,
    _retargeter_action_to_preview_qpos,
)
from src.dataset_acquisition.hri.wuji_retargeter_alignment_ui import (
    _as_qpos as _as_wuji_qpos,
    _alignment_samples as _wuji_alignment_samples,
    _delete_alignment_sample as _delete_wuji_alignment_sample,
    _resolve_target_source_sample,
    _wuji_joint_names,
)
from src.dataset_acquisition.hri.allegro_pedal_pose_ui import (
    AllegroPedalPoseStudio,
    allegro_tactile_finger_levels,
    ensure_editable_pose_copy,
    hold_contacted_allegro_fingers,
    interpolate_allegro_pose,
)


def test_raw_endpoints_map_to_urdf_endpoints_and_back():
    opened = [F1_RIGHT_RAW_OPEN[name] for name in F1_RIGHT_WIRE_JOINT_NAMES]
    closed = [F1_RIGHT_RAW_CLOSE[name] for name in F1_RIGHT_WIRE_JOINT_NAMES]

    open_qpos = raw_f1_to_master_qpos(opened)
    closed_qpos = raw_f1_to_master_qpos(closed)

    assert all(value == 0.0 for value in open_qpos.values())
    assert closed_qpos == F1_RIGHT_QPOS_UPPER
    assert master_qpos_to_raw_f1(closed_qpos) == F1_RIGHT_RAW_CLOSE


def test_named_state_can_be_reordered_without_changing_pose():
    values = np.array([700.0, 1200.0, 1100.0, 1000.0, 900.0, 850.0])
    reordered_names = tuple(reversed(F1_RIGHT_WIRE_JOINT_NAMES))
    expected = raw_f1_to_master_qpos(values)
    actual = raw_f1_to_master_qpos(values[::-1], reordered_names)
    assert actual == expected


def test_full_urdf_vector_only_sets_f1_masters():
    masters = raw_f1_to_master_qpos([1000.0] * 6)
    names = ("joint1", *F1_RIGHT_WIRE_JOINT_NAMES)
    qpos = urdf_qpos_from_master_qpos(masters, names)
    assert qpos[0] == 0.0
    assert np.allclose(qpos[1:], list(masters.values()))


def test_state_must_include_all_six_wire_joints():
    with pytest.raises(HandStateContractError, match="missing wire joints"):
        raw_f1_to_master_qpos([1.0], ["right_thumb_1_joint"])


def test_allegro_feedback_and_retargeter_orders_are_explicit():
    feedback = np.arange(16, dtype=np.float64) / 10.0
    urdf_hand_qpos = feedback_to_urdf_qpos(feedback, ALLEGRO_FEEDBACK_JOINT_NAMES)

    assert list(urdf_hand_qpos) == list(ALLEGRO_URDF_JOINT_NAMES)
    assert urdf_hand_qpos["thumb_base"] == 0.0
    assert urdf_hand_qpos["index_base"] == 0.4
    assert urdf_qpos_to_retargeter_action(urdf_hand_qpos) == [
        urdf_hand_qpos[name] for name in ALLEGRO_RETARGETER_JOINT_NAMES
    ]


def test_allegro_v5_driver_feedback_reorders_finger_blocks_to_urdf_pose():
    # The incoming v5 vector is index, middle, ring, thumb.  Its joint names
    # are in numeric driver order, not the semantic order used for the URDF.
    feedback = np.arange(16, dtype=np.float64) / 10.0
    v5 = feedback_to_urdf_qpos(feedback, ALLEGRO_V5_DRIVER_JOINT_NAMES)
    assert v5["thumb_base"] == 1.2
    assert v5["index_base"] == 0.0
    assert v5["middle_base"] == 0.4
    assert v5["ring_base"] == 0.8
    assert ALLEGRO_V5_FEEDBACK_JOINT_NAMES[:4] == (
        "joint_12_0", "joint_13_0", "joint_14_0", "joint_15_0"
    )


def test_allegro_full_urdf_vector_only_sets_hand_joints():
    hand_qpos = feedback_to_urdf_qpos(np.arange(16), ALLEGRO_FEEDBACK_JOINT_NAMES)
    names = ("joint1", *ALLEGRO_URDF_JOINT_NAMES)
    qpos = urdf_qpos_from_hand_qpos(hand_qpos, names)
    assert qpos[0] == 0.0
    assert np.allclose(qpos[1:], list(hand_qpos.values()))


def test_inspire_f1_retargeter_action_is_named_before_wire_publish():
    raw_target = retargeter_action_to_raw_f1([10, 20, 30, 40, 50, 60])
    assert raw_target == {
        "right_thumb_1_joint": 50.0,
        "right_thumb_2_joint": 60.0,
        "right_index_1_joint": 40.0,
        "right_middle_1_joint": 30.0,
        "right_ring_1_joint": 20.0,
        "right_little_1_joint": 10.0,
    }


def test_allegro_retargeter_action_is_converted_to_controller_order():
    action = np.arange(16, dtype=np.float64)
    controller_qpos = retargeter_action_to_controller_qpos(action)
    assert controller_qpos == [12.0, 13.0, 14.0, 15.0, 0.0, 1.0, 2.0, 3.0,
                               4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


def test_allegro_action_round_trips_through_semantic_urdf_names():
    action = np.arange(16, dtype=np.float64)
    qpos = retargeter_action_to_urdf_qpos(action)
    assert urdf_qpos_to_retargeter_action(qpos) == list(action)


def test_allegro_v5_uses_capture_session_retargeter_order_without_reordering():
    action = np.arange(16, dtype=np.float64)
    assert retargeter_action_to_live_controller_qpos(action, "allegro_v5") == list(action)
    assert retargeter_action_to_live_controller_qpos(action, "allegro") == [
        12.0, 13.0, 14.0, 15.0, 0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
    ]


def test_allegro_pedal_pose_interpolation_is_semantic_and_clamped():
    pose_a = {name: float(index) for index, name in enumerate(ALLEGRO_URDF_JOINT_NAMES)}
    pose_b = {name: float(index + 16) for index, name in enumerate(ALLEGRO_URDF_JOINT_NAMES)}
    midpoint = interpolate_allegro_pose(pose_a, pose_b, 0.5)
    assert midpoint["thumb_base"] == 8.0
    assert midpoint["ring_distal"] == 23.0
    assert interpolate_allegro_pose(pose_a, pose_b, -1.0) == pose_a
    assert interpolate_allegro_pose(pose_a, pose_b, 2.0) == pose_b


def test_allegro_tactile_levels_use_one_contiguous_block_per_finger():
    tactile = np.array([
        0, 4, 1, 0,       # index
        -7, 0, 0, 2,      # middle
        3, -12, 0, 0,     # ring
        0, 0, 9, 1,       # thumb
    ])
    assert allegro_tactile_finger_levels(tactile) == {
        "index": 4.0,
        "middle": 7.0,
        "ring": 12.0,
        "thumb": 9.0,
    }
    assert allegro_tactile_finger_levels(np.array([])) is None
    assert allegro_tactile_finger_levels(None) is None


def test_allegro_tactile_hold_only_replaces_contacted_finger_blocks():
    desired = np.arange(16, dtype=float) + 10.0
    feedback = -np.arange(16, dtype=float)
    command = hold_contacted_allegro_fingers(
        desired,
        feedback,
        {"index": True, "middle": False, "ring": True, "thumb": False},
    )
    assert np.array_equal(command[:4], feedback[:4])
    assert np.array_equal(command[4:8], desired[4:8])
    assert np.array_equal(command[8:12], feedback[8:12])
    assert np.array_equal(command[12:16], desired[12:16])


def test_allegro_tactile_raw_values_update_four_read_only_sliders():
    studio = object.__new__(AllegroPedalPoseStudio)
    studio.tactile_raw_status = SimpleNamespace(value="")
    studio.tactile_value_sliders = {
        finger: SimpleNamespace(value=0)
        for finger in ("index", "middle", "ring", "thumb")
    }
    studio._update_tactile_raw_status(
        {"tactile": np.array([12, 200, 5100, -3]), "tactile_time": time.perf_counter()}
    )
    assert [studio.tactile_value_sliders[finger].value for finger in ("index", "middle", "ring", "thumb")] == [12, 200, 5000, 0]
    assert "age=" in studio.tactile_raw_status.value


def test_allegro_tactile_stop_latches_contacted_finger_and_fails_closed_when_stale():
    # Construct just the tactile-control portion: no ROS/Viser process is
    # needed to lock down command behaviour.
    studio = object.__new__(AllegroPedalPoseStudio)
    studio.tactile_contact_stop = True
    studio.moving_toward_b = True
    studio.tactile_threshold = 200.0
    studio.parameter = 0.6
    studio.contact_latched = {
        "index": False, "middle": False, "ring": False, "thumb": False,
    }
    studio.contact_latch_parameter = {
        "index": None, "middle": None, "ring": None, "thumb": None,
    }
    studio.contact_hold_action = {
        "index": None, "middle": None, "ring": None, "thumb": None,
    }
    studio.contact_latch_level = {
        "index": None, "middle": None, "ring": None, "thumb": None,
    }
    studio.contact_above_threshold_count = {
        "index": 0, "middle": 0, "ring": 0, "thumb": 0,
    }
    studio.tactile_status = SimpleNamespace(value="")
    desired = np.ones(16)
    feedback = np.zeros(16)
    fresh_feedback = {
        "qpos": feedback,
        "tactile": np.array([201, 0, 0, 0]),
        "tactile_time": time.perf_counter(),
    }
    # One or two high samples arm the contact but cannot stop a finger yet.
    assert np.array_equal(
        studio._apply_tactile_contact_stop(desired, fresh_feedback), desired
    )
    assert np.array_equal(
        studio._apply_tactile_contact_stop(desired, fresh_feedback), desired
    )
    command = studio._apply_tactile_contact_stop(desired, fresh_feedback)
    assert np.array_equal(command[:4], feedback[:4])
    assert np.array_equal(command[4:], desired[4:])
    assert studio.contact_latched["index"]
    assert studio.contact_latch_parameter["index"] == 0.6
    assert studio.contact_latch_level["index"] == 201.0
    assert np.array_equal(studio.contact_hold_action["index"], feedback[:4])

    # Later measured deflection must not move the latched command target.
    deflected_feedback = dict(fresh_feedback, qpos=np.full(16, -0.5))
    held_command = studio._apply_tactile_contact_stop(desired, deflected_feedback)
    assert np.array_equal(held_command[:4], feedback[:4])

    # Reopening keeps the stopped finger fixed until the global parameter has
    # returned to where that finger made contact, then releases it.
    studio._set_parameter(0.7)
    studio._set_parameter(0.61)
    assert studio.contact_latched["index"]
    studio._set_parameter(0.6)
    assert not studio.contact_latched["index"]
    assert studio.contact_latch_parameter["index"] is None
    assert studio.contact_hold_action["index"] is None

    studio._set_parameter(0.7)
    stale_feedback = dict(fresh_feedback, tactile_time=time.perf_counter() - 1.0)
    assert np.array_equal(
        studio._apply_tactile_contact_stop(desired, stale_feedback), feedback
    )


def test_allegro_pedal_endpoint_copy_preserves_source_and_is_reusable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "metadata.json").write_text('{"source": true}', encoding="utf-8")
    (source / "target_robot.json").write_text('{"value": 1}', encoding="utf-8")
    destination = tmp_path / "editable" / "000000"

    result = ensure_editable_pose_copy(destination, source)

    assert result == destination.resolve()
    assert (source / "target_robot.json").read_text(encoding="utf-8") == '{"value": 1}'
    target = json.loads((destination / "target_robot.json").read_text(encoding="utf-8"))
    assert target["value"] == 1
    assert target["pedal_pose_source"] == str(source.resolve())
    assert ensure_editable_pose_copy(destination, source) == destination.resolve()


def test_allegro_v5_direct_alignment_matches_all_seven_edited_targets():
    anchors = _load_allegro_v5_direct_anchor_data()
    assert anchors["raw_features"].shape == (7, 16)
    assert anchors["manus_tip_features"].shape == (7, 15)
    assert anchors["manus_ergonomic_features"].shape == (7, 6)
    for raw, tip_feature, ergonomic_feature, expected in zip(
        anchors["raw_features"],
        anchors["manus_tip_features"],
        anchors["manus_ergonomic_features"],
        anchors["targets"],
    ):
        np.testing.assert_allclose(
            _align_allegro_v5_action(raw, tip_feature, ergonomic_feature),
            expected,
            atol=2e-6,
        )


def test_wuji_left_direct_alignment_matches_both_operator_edited_targets():
    anchors = _load_wuji_direct_anchor_data(is_right=False)
    assert anchors["raw_features"].shape == (3, 20)
    assert anchors["manus_tip_features"].shape == (3, 15)
    for raw, tip_feature, expected in zip(
        anchors["raw_features"], anchors["manus_tip_features"], anchors["targets"]
    ):
        keypoints = np.zeros((21, 3), dtype=np.float64)
        keypoints[9, 2] = 1.0
        keypoints[[4, 8, 12, 16, 20]] = tip_feature.reshape(5, 3)
        np.testing.assert_allclose(
            _align_wuji_direct_action(raw, keypoints, is_right=False),
            expected,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            _wuji_direct_tip_feature(keypoints * 1.15), tip_feature, atol=2e-6
        )


def test_wuji_duplicate_finger_pose_count_does_not_increase_its_influence():
    # Five equivalent captures at pose 0 and one at pose 1 must contribute as
    # two pose clusters, not a 5:1 sample-count vote.
    features = np.array([[0.0]] * 5 + [[1.0]])
    targets = np.array([[0.0]] * 5 + [[10.0]])
    target, influence = _cluster_balanced_distance_gated_target(
        features,
        np.array([0.5]),
        targets,
        sigma=10.0,
        influence_radius=10.0,
        epsilon=1e-8,
        distance_power=1.0,
        duplicate_cluster_radius=0.01,
    )
    assert influence > 0.0
    np.testing.assert_allclose(target, [5.0], atol=1e-6)


def test_wuji_left_direct_alignment_is_finger_local_and_physically_clamped():
    anchors = _load_wuji_direct_anchor_data(is_right=False)
    raw = anchors["raw_features"][0].copy()
    tips = anchors["manus_tip_features"][0].copy()
    # Only the index input comes from the bent capture. The target's index
    # block must follow it while the manually fixed thumb block stays open.
    raw[4:8] = anchors["raw_features"][1, 4:8]
    tips[3:6] = anchors["manus_tip_features"][1, 3:6]
    keypoints = np.zeros((21, 3), dtype=np.float64)
    keypoints[9, 2] = 1.0
    keypoints[[4, 8, 12, 16, 20]] = tips.reshape(5, 3)
    result = _align_wuji_direct_action(raw, keypoints, is_right=False)
    np.testing.assert_allclose(result[:4], anchors["targets"][0, :4], atol=2e-6)
    np.testing.assert_allclose(result[4:8], anchors["targets"][1, 4:8], atol=2e-6)
    lower, upper = anchors["safe_lower"], anchors["safe_upper"]
    assert np.all(clip_wuji_direct_safe_action(np.full(20, -1.0e6), is_right=False) == lower)
    assert np.all(clip_wuji_direct_safe_action(np.full(20, 1.0e6), is_right=False) == upper)


def test_wuji_left_direct_alignment_blends_smoothly_between_edited_anchors():
    anchors = _load_wuji_direct_anchor_data(is_right=False)
    previous = None
    max_step = 0.0
    for ratio in np.linspace(0.0, 1.0, 101):
        raw = (
            (1.0 - ratio) * anchors["raw_features"][0]
            + ratio * anchors["raw_features"][1]
        )
        tips = (
            (1.0 - ratio) * anchors["manus_tip_features"][0]
            + ratio * anchors["manus_tip_features"][1]
        )
        keypoints = np.zeros((21, 3), dtype=np.float64)
        keypoints[9, 2] = 1.0
        keypoints[[4, 8, 12, 16, 20]] = tips.reshape(5, 3)
        action = _align_wuji_direct_action(raw, keypoints, is_right=False)
        assert np.all(action >= anchors["safe_lower"])
        assert np.all(action <= anchors["safe_upper"])
        if previous is not None:
            max_step = max(max_step, np.max(np.abs(action - previous)))
        previous = action
    # The latest first/second edited targets differ more than the original
    # pair, while the extra third anchor also participates in the blend.
    # Keep a 1% input step comfortably below a visible 0.07-rad jump.
    assert max_step < 0.07


def test_allegro_v5_ergonomics_uses_named_angles_and_ignores_field_order():
    ergonomics = {
        "ThumbMCPStretch": 20.0,
        "ThumbMCPSpread": 10.0,
        "IndexMCPStretch": 1.0,
        "IndexPIPStretch": 2.0,
        "IndexDIPStretch": 3.0,
        "MiddleMCPStretch": 4.0,
        "MiddlePIPStretch": 5.0,
        "MiddleDIPStretch": 6.0,
        "RingMCPStretch": 7.0,
        "RingPIPStretch": 8.0,
        "RingDIPStretch": 9.0,
        "PinkyMCPStretch": 10.0,
        "PinkyPIPStretch": 11.0,
        "PinkyDIPStretch": 12.0,
    }
    expected = np.array([20.0, 10.0, 6.0, 15.0, 24.0, 33.0])
    np.testing.assert_allclose(_allegro_v5_manus_ergonomic_feature(ergonomics), expected)
    assert _allegro_v5_manus_ergonomic_feature({}) is None


def test_allegro_v5_raw_uses_full_manus_joint_angles_for_vive():
    ergonomics = {
        "ThumbMCPStretch": 42.0,
        "ThumbMCPSpread": 31.0,
        "ThumbPIPStretch": 30.0,
        "ThumbDIPStretch": 60.0,
        "IndexSpread": -6.0,
        "IndexMCPStretch": 20.0,
        "IndexPIPStretch": 30.0,
        "IndexDIPStretch": 40.0,
        "MiddleSpread": -4.0,
        "MiddleMCPStretch": 25.0,
        "MiddlePIPStretch": 35.0,
        "MiddleDIPStretch": 45.0,
        "RingSpread": -2.0,
        "RingMCPStretch": 30.0,
        "RingPIPStretch": 40.0,
        "RingDIPStretch": 50.0,
    }
    expected = np.array(
        [
            np.deg2rad(-6.0),
            (np.deg2rad(20.0) - 0.35) * 1.5,
            np.deg2rad(30.0),
            np.deg2rad(40.0),
            np.deg2rad(-4.0),
            (np.deg2rad(25.0) - 0.35) * 1.5,
            np.deg2rad(35.0),
            np.deg2rad(45.0),
            np.deg2rad(-2.0),
            (np.deg2rad(30.0) - 0.35) * 1.5,
            np.deg2rad(40.0),
            np.deg2rad(50.0),
            (1.0 - np.clip(42.0 / (53.6 - (-13.0)), 0.0, 1.0))
            * 0.8
            * 1.78,
            -np.deg2rad(31.0) - 1.57,
            1.2 * np.sin(np.deg2rad(30.0)),
            1.2 * np.sin(np.deg2rad(60.0)),
        ]
    )
    np.testing.assert_allclose(
        _allegro_v5_raw_from_manus_ergonomics(dict(reversed(ergonomics.items()))),
        expected,
    )
    # A VIVE/MANUS packet can now drive the raw path without a 4x4 frame.
    np.testing.assert_allclose(allegro_v5({}, ergonomics=ergonomics), expected)

    low_stretch = dict(ergonomics, ThumbMCPStretch=0.0)
    high_stretch = dict(ergonomics, ThumbMCPStretch=53.6 - (-13.0))
    low_action = _allegro_v5_raw_from_manus_ergonomics(low_stretch)[12]
    high_action = _allegro_v5_raw_from_manus_ergonomics(high_stretch)[12]
    np.testing.assert_allclose(low_action, 0.8 * 1.78)
    np.testing.assert_allclose(high_action, 0.0)
    assert low_action > high_action


def test_allegro_v5_raw_rejects_partial_manus_ergonomics():
    with pytest.raises(ValueError, match="missing Allegro v5 raw inputs"):
        _allegro_v5_raw_from_manus_ergonomics({"IndexMCPStretch": 10.0})


def test_allegro_v5_invalid_ergonomics_preserves_existing_transform_result():
    anchors = _load_allegro_v5_direct_anchor_data()
    raw = anchors["raw_features"][6]
    tip_feature = anchors["manus_tip_features"][6]
    expected = _align_allegro_v5_action(raw, tip_feature)
    actual = _align_allegro_v5_action(raw, tip_feature, np.array([np.nan] * 6))
    np.testing.assert_allclose(actual, expected)


def test_allegro_v5_interpolation_favors_nearest_anchor_without_hard_switching():
    features = np.array([[0.0], [1.0], [2.0]])
    query = np.array([0.4])
    linear_weights, _ = _distance_gated_weights(
        features, query, sigma=100.0, epsilon=1e-6, distance_power=1.0
    )
    smooth_nearest_weights, _ = _distance_gated_weights(
        features, query, sigma=100.0, epsilon=1e-6, distance_power=1.5
    )
    # The nearest anchor is 0.0; increasing Shepard's exponent must make its
    # share larger relative to either more distant anchor.
    assert smooth_nearest_weights[0] > linear_weights[0]
    assert (
        smooth_nearest_weights[0] / smooth_nearest_weights[1]
        > linear_weights[0] / linear_weights[1]
    )
    assert (
        smooth_nearest_weights[0] / smooth_nearest_weights[2]
        > linear_weights[0] / linear_weights[2]
    )


def test_allegro_v5_interpolates_each_finger_from_its_own_nearest_anchor():
    anchors = _load_allegro_v5_direct_anchor_data()
    hybrid_tip_feature = anchors["manus_tip_features"][0].copy()
    hybrid_raw = anchors["raw_features"][0].copy()
    # Index tip comes from anchor 1; thumb tip comes from anchor 4.  The
    # remaining finger tips stay at anchor 0.
    hybrid_tip_feature[3:6] = anchors["manus_tip_features"][1, 3:6]
    hybrid_tip_feature[:3] = anchors["manus_tip_features"][4, :3]
    hybrid_raw[:4] = anchors["raw_features"][1, :4]
    hybrid_raw[12:] = anchors["raw_features"][4, 12:]

    action = _align_allegro_v5_action(hybrid_raw, hybrid_tip_feature)

    np.testing.assert_allclose(action[:4], anchors["targets"][1, :4], atol=2e-6)
    np.testing.assert_allclose(action[12:], anchors["targets"][4, 12:], atol=2e-6)


def test_allegro_v5_uses_only_seven_target_anchors_outside_local_radius():
    anchors = _load_allegro_v5_direct_anchor_data()
    raw = anchors["raw_features"][0].copy()
    tips = anchors["manus_tip_features"][0].copy() + 10.0

    action = _align_allegro_v5_action(raw, tips)

    for action_slice, tip_slice in (
        (slice(0, 4), slice(3, 6)),
        (slice(4, 8), slice(6, 9)),
        (slice(8, 12), slice(9, 12)),
        (slice(12, 16), slice(0, 3)),
    ):
        features = np.concatenate(
            (
                anchors["manus_tip_features"][:, tip_slice],
                anchors["raw_features"][:, action_slice] * 0.1,
            ),
            axis=1,
        )
        query = np.concatenate((tips[tip_slice], raw[action_slice] * 0.1))
        weights, _ = _distance_gated_weights(
            features,
            query,
            sigma=anchors["sigma"],
            epsilon=anchors["inverse_distance_epsilon"],
            distance_power=anchors["inverse_distance_power"],
        )
        expected = weights @ anchors["targets"][:, action_slice]
        np.testing.assert_allclose(action[action_slice], expected, atol=2e-6)


def test_retargetor_passes_right_manus_ergonomics_to_allegro_v5():
    class Recorder:
        def __init__(self):
            self.frame = None
            self.ergonomics = None

        def __call__(self, frame, *, ergonomics=None):
            self.frame = frame
            self.ergonomics = ergonomics
            return np.zeros(16)

    recorder = Recorder()
    frame = {"wrist": np.eye(4)}
    ergonomics = {"ThumbMCPStretch": 20.0}
    retargetor = Retargetor(arm_name=None, hand_name="allegro_v5", hand_side="Right")
    retargetor.hand_retargetor = recorder
    retargetor.start(np.eye(4))
    _, action = retargetor.get_action({"Right": frame, "ergonomics": {"Right": ergonomics}})
    assert recorder.frame is frame
    assert recorder.ergonomics is ergonomics
    np.testing.assert_array_equal(action, np.zeros(16))


def test_alignment_ui_matches_capture_v5_ergonomics_input_contract():
    ergonomics = {"ThumbMCPStretch": 20.0}
    assert _live_retargeter_kwargs("allegro_v5", "direct", ergonomics) == {
        "ergonomics": ergonomics
    }
    assert _live_retargeter_kwargs("allegro_v5_wonik", "direct", ergonomics) == {
        "ergonomics": ergonomics
    }
    assert _live_retargeter_kwargs("allegro_v5", "anyteleop", ergonomics) == {}
    assert _live_retargeter_kwargs("allegro", "direct", ergonomics) == {}


def test_alignment_ui_observe_only_reads_feedback_without_moving_hand():
    class Hand:
        def __init__(self):
            self.moves = []

        def get_data(self):
            return {"is_connected": True, "qpos": np.zeros(16)}

        def move(self, target):
            self.moves.append(target)

    studio = object.__new__(AllegroRetargeterAlignmentStudio)
    studio.live = True
    studio.observe_only = True
    studio.hand = Hand()
    studio.command_target_status = SimpleNamespace(value=None)
    statuses = []
    studio._show_allegro_connection = lambda _feedback: True
    studio._set_command_status = statuses.append

    studio._send_right_hand_command()

    assert studio.hand.moves == []
    assert statuses == ["OBSERVING ALLEGRO FEEDBACK (NO COMMANDS)"]
    assert studio.command_target_status.value == "Observation only: no command target"


def test_alignment_ui_open_mode_can_observe_hand_without_manus():
    class Hand:
        def __init__(self):
            self.moves = []

        def get_data(self):
            return {"is_connected": True, "qpos": np.zeros(16)}

        def move(self, target):
            self.moves.append(target)

    studio = object.__new__(AllegroRetargeterAlignmentStudio)
    studio.live = False
    studio.hand_control = True
    studio.observe_only = True
    studio.hand = Hand()
    studio.command_target_status = SimpleNamespace(value=None)
    statuses = []
    studio._show_allegro_connection = lambda _feedback: True
    studio._set_command_status = statuses.append

    studio._send_right_hand_command()

    assert studio.hand.moves == []
    assert statuses == ["OBSERVING ALLEGRO FEEDBACK (NO COMMANDS)"]


def test_alignment_ui_loaded_target_is_not_overwritten_by_live_feedback(tmp_path):
    target_qpos = {
        name: float(index) / 10.0
        for index, name in enumerate(ALLEGRO_URDF_JOINT_NAMES)
    }
    (tmp_path / "target_robot.json").write_text(
        json.dumps({"edited": True, "urdf_hand_qpos_rad": target_qpos}),
        encoding="utf-8",
    )
    studio = object.__new__(AllegroRetargeterAlignmentStudio)
    studio.source_sample = tmp_path
    studio.captured_hand_qpos = None
    studio.lock = Lock()
    studio.slider_target_active = False
    applied = []
    statuses = []
    studio.set_hand_qpos = lambda values: applied.append(dict(values))
    studio._set_command_status = statuses.append

    studio.load_saved_target()

    assert studio.slider_target_active is True
    assert applied == [target_qpos]
    assert statuses == ["EDITED TARGET POSE"]


def test_alignment_ui_applies_slider_target_in_the_button_callback():
    studio = object.__new__(AllegroRetargeterAlignmentStudio)
    studio.hand = object()
    studio.observe_only = False
    studio.simulate = False
    calls = []
    studio.hold_slider_target = lambda **_kwargs: calls.append("hold")
    studio._send_right_hand_command = lambda: calls.append("send")

    studio.apply_slider_target_now()

    assert calls == ["hold", "send"]


def test_alignment_ui_continuous_slider_button_toggles_follow_mode():
    studio = object.__new__(AllegroRetargeterAlignmentStudio)
    studio.live = False
    studio.hand_control = True
    studio.observe_only = False
    studio.simulate = False
    studio.slider_follow_enabled = Event()
    calls = []
    statuses = []
    studio.apply_slider_target_now = lambda: calls.append("apply")
    studio._set_command_status = statuses.append

    studio.toggle_continuous_slider_control()
    assert studio.slider_follow_enabled.is_set()
    assert calls == ["apply"]
    assert statuses == ["CONTINUOUS SLIDER CONTROL ON"]

    studio.toggle_continuous_slider_control()
    assert not studio.slider_follow_enabled.is_set()
    assert statuses[-1] == "CONTINUOUS SLIDER CONTROL OFF: HOLDING LAST TARGET"


def test_alignment_ui_simulation_preview_uses_v5_urdf_joint_order():
    action = np.arange(16, dtype=np.float64) / 10.0
    expected = retargeter_action_to_urdf_qpos(action)
    assert _retargeter_action_to_preview_qpos("allegro_v5", action) == expected


def test_wuji_alignment_uses_controller_ordered_twenty_joint_contract():
    names = _wuji_joint_names("right")
    assert names[0] == "right_finger1_joint1"
    assert names[-1] == "right_finger5_joint4"
    np.testing.assert_array_equal(_as_wuji_qpos(np.arange(20)), np.arange(20))
    with pytest.raises(ValueError):
        _as_wuji_qpos(np.arange(19))


def test_wuji_alignment_capture_browser_orders_and_deletes_selected_sample(tmp_path):
    session_root = tmp_path / "wuji_alignment"
    session_root.mkdir()
    for name in ("000010", "000002", "000000", "notes"):
        (session_root / name).mkdir()

    assert [path.name for path in _wuji_alignment_samples(session_root)] == [
        "000000",
        "000002",
        "000010",
    ]
    _delete_wuji_alignment_sample(session_root / "000002", session_root)
    assert [path.name for path in _wuji_alignment_samples(session_root)] == [
        "000000",
        "000010",
    ]
    assert (session_root / "notes").is_dir()
    with pytest.raises(ValueError, match="Refusing to delete"):
        _delete_wuji_alignment_sample(tmp_path / "outside", session_root)


def test_wuji_open_target_source_is_limited_to_a_numbered_sibling_capture(tmp_path):
    session_root = tmp_path / "wuji_alignment"
    source = session_root / "000002"
    target = session_root / "000010"
    source.mkdir(parents=True)
    target.mkdir()
    assert _resolve_target_source_sample(source, "000010") == target.resolve()
    assert _resolve_target_source_sample(source, None) is None
    with pytest.raises(ValueError, match="numbered capture"):
        _resolve_target_source_sample(source, "latest")
    with pytest.raises(ValueError, match="does not exist"):
        _resolve_target_source_sample(source, "000999")


def test_allegro_v5_direct_alignment_blends_smoothly_between_anchors():
    anchors = _load_allegro_v5_direct_anchor_data()
    previous = None
    max_step = 0.0
    for ratio in np.linspace(0.0, 1.0, 101):
        raw = (
            (1.0 - ratio) * anchors["raw_features"][0]
            + ratio * anchors["raw_features"][1]
        )
        tip_feature = (
            (1.0 - ratio) * anchors["manus_tip_features"][0]
            + ratio * anchors["manus_tip_features"][1]
        )
        action = _align_allegro_v5_action(raw, tip_feature)
        assert np.all(action >= anchors["safe_lower"])
        assert np.all(action <= anchors["safe_upper"])
        if previous is not None:
            max_step = max(max_step, np.max(np.abs(action - previous)))
        previous = action
    # A 1% input interpolation step must not recreate the old RBF's 0.335-rad
    # near-anchor jump.
    assert max_step < 0.07


def test_allegro_v5_alignment_clamps_to_operator_approved_safety_bar():
    action = clip_allegro_v5_safe_action(np.full(16, 1.0e6))
    # These upper bars come from the seven edited targets and are validated
    # against the Allegro v5 URDF's physical hard limits.
    assert action[3] == pytest.approx(1.068)
    assert action[7] == pytest.approx(1.618)
    assert action[15] == pytest.approx(1.116)


def test_allegro_preview_updates_the_viser_mesh_handle_not_only_urdf_state():
    class ViserRobot:
        def __init__(self):
            self.configuration = None

        def update_cfg(self, configuration):
            self.configuration = np.asarray(configuration, dtype=np.float64)

    studio = object.__new__(AllegroRetargeterAlignmentStudio)
    studio.lock = Lock()
    studio.hand_qpos = {
        name: float(index) for index, name in enumerate(ALLEGRO_URDF_JOINT_NAMES)
    }
    studio.urdf_joint_names = ALLEGRO_URDF_JOINT_NAMES
    studio.viser_robot = ViserRobot()

    studio._update_robot()

    np.testing.assert_array_equal(studio.viser_robot.configuration, np.arange(16))


def test_live_feedback_updates_mesh_target_when_no_slider_target_is_active():
    studio = object.__new__(AllegroRetargeterAlignmentStudio)
    studio.lock = Lock()
    studio.actual_hand_qpos = None
    studio.slider_target_active = False
    studio.actual_feedback_status = type("Status", (), {"value": None})()
    applied_qposes = []
    studio.set_hand_qpos = lambda qpos: applied_qposes.append(qpos)
    feedback = np.arange(16, dtype=np.float64) / 10.0

    studio._sync_preview_from_feedback(feedback, ALLEGRO_V5_DRIVER_JOINT_NAMES)
    expected = feedback_to_urdf_qpos(
        feedback, ALLEGRO_V5_DRIVER_JOINT_NAMES
    )
    assert studio.actual_hand_qpos == expected
    assert studio.actual_feedback_status.value is not None
    assert applied_qposes == [expected]


def test_live_feedback_preserves_slider_target_during_slider_hold():
    studio = object.__new__(AllegroRetargeterAlignmentStudio)
    studio.lock = Lock()
    studio.actual_hand_qpos = None
    studio.slider_target_active = True
    studio.actual_feedback_status = type("Status", (), {"value": None})()
    studio.set_hand_qpos = lambda _qpos: pytest.fail("feedback overwrote slider target")

    studio._sync_preview_from_feedback(
        np.arange(16, dtype=np.float64) / 10.0,
        ALLEGRO_V5_DRIVER_JOINT_NAMES,
    )


def test_alignment_capture_browser_orders_numeric_samples_and_deletes_only_selected(tmp_path):
    session_root = tmp_path / "allegro_alignment"
    session_root.mkdir()
    for name in ("000010", "000002", "000000", "notes"):
        (session_root / name).mkdir()

    assert [path.name for path in _alignment_samples(session_root)] == [
        "000000",
        "000002",
        "000010",
    ]

    _delete_alignment_sample(session_root / "000002", session_root)

    assert [path.name for path in _alignment_samples(session_root)] == ["000000", "000010"]
    assert (session_root / "notes").is_dir()
    with pytest.raises(ValueError, match="Refusing to delete"):
        _delete_alignment_sample(tmp_path / "outside", session_root)


def _right_only_vive_receiver():
    receiver = object.__new__(ViveManusROSReceiver)
    receiver.lock = Lock()
    receiver.save_event = Event()
    receiver.data = None
    receiver.hand_side = "right"
    receiver.require_left_control = False
    receiver.use_vive = True
    receiver.max_age_s = 1.0
    now = time.monotonic()
    receiver._vive_right = np.eye(4)
    receiver._vive_right_time = now
    receiver._vive_left = None
    receiver._vive_left_time = None
    receiver._manus_frames = {"Right": {"wrist": np.eye(4)}, "Left": None}
    receiver._manus_ergonomics = {"Right": {}, "Left": None}
    receiver._manus_times = {"Right": now, "Left": None}
    return receiver


def test_right_only_vive_mode_does_not_wait_for_left_manus():
    receiver = _right_only_vive_receiver()
    result = receiver.get_data()
    assert result["Left"] is None
    assert result["Right"] is not None


def test_legacy_left_control_mode_still_waits_for_left_manus():
    receiver = _right_only_vive_receiver()
    receiver.require_left_control = True
    result = receiver.get_data()
    assert result["Right"] is None


def test_manus_only_mode_does_not_require_vive_or_left_manus():
    receiver = _right_only_vive_receiver()
    receiver.use_vive = False
    receiver._vive_right = None
    receiver._vive_right_time = None
    result = receiver.get_data()
    assert result["Left"] is None
    assert result["Right"] is not None


def test_vive_only_mode_exposes_wrist_without_any_manus_frame():
    receiver = _right_only_vive_receiver()
    receiver.use_manus = False
    receiver._manus_frames = {"Right": None, "Left": None}
    receiver._manus_times = {"Right": None, "Left": None}

    result = receiver.get_data()

    assert result["Left"] is None
    np.testing.assert_array_equal(result["Right"]["wrist"], np.eye(4))
    assert result["ergonomics"] == {}
