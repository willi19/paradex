import json
from types import SimpleNamespace
import time
from threading import Event, Lock

import numpy as np
import pytest

from paradex.io.teleop.vive.receiver import (
    _VIVE_TRACKER_TO_WRIST_ROTATION,
    _VIVE_TRANSLATION_AXIS_TRANSFORM,
    _canonical_joint_name,
    apply_vive_tracker_mount_rotation,
    manus_ergonomics_to_dict,
    manus_message_to_frame,
    pose_to_matrix,
    reparent_frame,
    ViveManusROSReceiver,
    _pause_state_from_ergonomics,
)
from paradex.retargetor.hand_regargetor import inspire_f1
from paradex.retargetor.state import HandStateExtractor
from paradex.retargetor.unimanual import Retargetor


def make_pose(position=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        position=SimpleNamespace(
            x=position[0],
            y=position[1],
            z=position[2],
        ),
        orientation=SimpleNamespace(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3],
        ),
    )


def make_node(node_id, parent_id, chain, joint, position):
    return SimpleNamespace(
        node_id=node_id,
        parent_node_id=parent_id,
        chain_type=chain,
        joint_type=joint,
        pose=make_pose(position),
    )


def make_manus_message(side="Right", invalid_orientations=False, flex=0.0):
    quaternion = (
        (np.nan, np.nan, np.nan, np.nan)
        if invalid_orientations
        else (0.0, 0.0, 0.0, 1.0)
    )
    nodes = [make_node(0, 0, "Hand", "Invalid", (1.0, 2.0, 3.0))]
    node_id = 1
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        joints = (
            ("MCP", "PIP", "DIP", "TIP")
            if finger == "Thumb"
            else ("MCP", "PIP", "IP", "DIP", "TIP")
        )
        parent_id = 0
        for joint_index, joint in enumerate(joints):
            nodes.append(
                make_node(
                    node_id,
                    parent_id,
                    finger,
                    joint,
                    (1.0 + node_id * 0.01, 2.0 + joint_index * 0.01, 3.0),
                )
            )
            parent_id = node_id
            node_id += 1
    for node in nodes:
        node.pose.orientation = SimpleNamespace(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3],
        )
    ergonomics = []
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        ergonomics.extend(
            [
                SimpleNamespace(type=f"{finger}MCPStretch", value=flex),
                SimpleNamespace(type=f"{finger}PIPStretch", value=flex),
            ]
        )
    return SimpleNamespace(
        side=side,
        glove_id=42,
        raw_nodes=nodes,
        ergonomics=ergonomics,
    )


def test_pose_to_matrix_normalizes_quaternion():
    transform = pose_to_matrix(
        make_pose(position=(1.0, 2.0, 3.0), quaternion=(0.0, 0.0, 0.0, 2.0))
    )
    np.testing.assert_allclose(transform, np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ))


def _axis_rotation(axis, angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    if axis == "x":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
        )
    if axis == "y":
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
        )
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def test_vive_mount_rotation_preserves_existing_right_hand_correction():
    initial = np.eye(4)
    initial[:3, 3] = [0.4, -0.2, 1.1]
    corrected_initial = apply_vive_tracker_mount_rotation(initial)

    np.testing.assert_allclose(
        corrected_initial[:3, 3],
        _VIVE_TRANSLATION_AXIS_TRANSFORM @ initial[:3, 3],
    )
    np.testing.assert_allclose(
        corrected_initial[:3, :3],
        _VIVE_TRACKER_TO_WRIST_ROTATION,
    )
    for axis in ("x", "y", "z"):
        current = initial.copy()
        current[:3, :3] = _axis_rotation(axis, 0.2)
        corrected_current = apply_vive_tracker_mount_rotation(current)
        np.testing.assert_allclose(
            corrected_current[:3, :3],
            current[:3, :3] @ _VIVE_TRACKER_TO_WRIST_ROTATION,
            atol=1.0e-8,
        )


def test_canonical_joint_mapping_matches_live_manus_names():
    assert _canonical_joint_name("Thumb", "MCP") == "thumb_metacarpal"
    assert _canonical_joint_name("Thumb", "DIP") == "thumb_distal"
    assert _canonical_joint_name("Index", "IP") == "index_intermediate"
    assert _canonical_joint_name("Pinky", "TIP") == "pinky_tip"
    assert _canonical_joint_name("Hand", "Invalid") is None


def test_manus_message_converts_to_xsens_named_contract():
    frame = manus_message_to_frame(make_manus_message())
    assert frame["wrist"].shape == (4, 4)
    for finger in ("index", "middle", "ring", "pinky"):
        for joint in ("metacarpal", "proximal", "intermediate", "distal"):
            assert f"{finger}_{joint}" in frame
    for joint in ("metacarpal", "proximal", "distal"):
        assert f"thumb_{joint}" in frame


def test_manus_nan_orientations_are_reconstructed_from_joint_positions():
    frame = manus_message_to_frame(
        make_manus_message(invalid_orientations=True)
    )

    for transform in frame.values():
        assert np.all(np.isfinite(transform))
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3],
            np.eye(3),
            atol=1.0e-6,
        )


def test_reparent_frame_uses_vive_wrist_and_preserves_manus_relative_pose():
    source = manus_message_to_frame(make_manus_message())
    vive_wrist = np.eye(4)
    vive_wrist[:3, 3] = [10.0, 20.0, 30.0]

    result = reparent_frame(source, vive_wrist)

    np.testing.assert_allclose(result["wrist"], vive_wrist)
    expected_relative = (
        np.linalg.inv(source["wrist"]) @ source["index_distal"]
    )
    actual_relative = np.linalg.inv(result["wrist"]) @ result["index_distal"]
    np.testing.assert_allclose(actual_relative, expected_relative)


def test_frames_feed_existing_right_hand_and_left_state_retargetors():
    right_manus = manus_message_to_frame(make_manus_message("Right"))
    left_manus = manus_message_to_frame(make_manus_message("Left"))
    right_frame = reparent_frame(right_manus, np.eye(4))

    hand_action = inspire_f1(right_frame, is_right=True)
    state = HandStateExtractor().get_state(left_manus)

    assert hand_action.shape == (6,)
    assert state in (0, 1, 2, 3)


def make_receiver_without_ros(sample_age=0.0, hand_side="right"):
    receiver = ViveManusROSReceiver.__new__(ViveManusROSReceiver)
    receiver.hand_side = hand_side
    receiver.vive_right_topic = "/vive_trackers/right_hand/pose"
    receiver.vive_left_topic = "/vive_trackers/left_hand/pose"
    receiver.manus_topics = ("/manus_glove_0", "/manus_glove_1")
    receiver.lock = Lock()
    receiver.save_event = Event()
    receiver.data = None
    receiver.max_age_s = 0.25
    receiver._vive_right = np.eye(4)
    receiver._vive_left = np.eye(4)
    receiver._vive_left[:3, 3] = [-0.4, 0.2, 1.1]
    receiver._manus_frames = {
        "Left": manus_message_to_frame(make_manus_message("Left")),
        "Right": manus_message_to_frame(make_manus_message("Right")),
    }
    receiver._manus_ergonomics = {
        "Left": manus_ergonomics_to_dict(
            make_manus_message("Left").ergonomics
        ),
        "Right": manus_ergonomics_to_dict(
            make_manus_message("Right").ergonomics
        ),
    }
    timestamp = time.monotonic() - sample_age
    receiver._vive_right_time = timestamp
    receiver._vive_left_time = timestamp
    receiver._manus_times = {"Left": timestamp, "Right": timestamp}
    receiver._vive_frame_id = "world"
    receiver._vive_left_frame_id = "world"
    receiver._manus_glove_ids = {"Left": 41, "Right": 42}
    receiver.save_path = None
    return receiver


def test_get_data_fuses_fresh_right_vive_and_both_manus_hands():
    data = make_receiver_without_ros().get_data()

    assert data["Left"] is not None
    assert data["Right"] is not None
    assert data["ergonomics"]["Right"] is not None
    np.testing.assert_allclose(data["Right"]["wrist"], np.eye(4))


def test_get_data_fuses_each_manus_hand_with_its_bimanual_vive_tracker():
    receiver = make_receiver_without_ros(hand_side="bimanual")
    source_left = receiver._manus_frames["Left"]
    source_right = receiver._manus_frames["Right"]

    data = receiver.get_data()

    np.testing.assert_allclose(data["Left"]["wrist"], receiver._vive_left)
    np.testing.assert_allclose(data["Right"]["wrist"], receiver._vive_right)
    for side, source in (("Left", source_left), ("Right", source_right)):
        expected_relative = (
            np.linalg.inv(source["wrist"]) @ source["index_distal"]
        )
        actual_relative = (
            np.linalg.inv(data[side]["wrist"]) @ data[side]["index_distal"]
        )
        np.testing.assert_allclose(actual_relative, expected_relative)


def test_manus_only_left_mode_does_not_require_right_glove_or_vive():
    receiver = make_receiver_without_ros(hand_side="left")
    receiver.use_vive = False
    receiver.require_left_control = False
    receiver._manus_times["Right"] = None
    receiver._manus_frames["Right"] = None

    data = receiver.get_data()

    assert data["Left"] is not None
    assert data["Right"] is None
    assert set(data["ergonomics"]) == {"Left"}


def test_get_data_suppresses_commands_when_any_input_is_stale():
    data = make_receiver_without_ros(sample_age=1.0).get_data()

    assert data["Left"] is None
    assert data["Right"] is None


@pytest.mark.parametrize(
    "source",
    ("vive_left", "vive_right", "manus_left", "manus_right"),
)
def test_bimanual_get_data_suppresses_both_sides_when_one_input_is_stale(source):
    receiver = make_receiver_without_ros(hand_side="bimanual")
    stale_time = time.monotonic() - 1.0
    if source == "vive_left":
        receiver._vive_left_time = stale_time
    elif source == "vive_right":
        receiver._vive_right_time = stale_time
    elif source == "manus_left":
        receiver._manus_times["Left"] = stale_time
    else:
        receiver._manus_times["Right"] = stale_time

    data = receiver.get_data()

    assert data["Left"] is None
    assert data["Right"] is None


def test_unimanual_right_does_not_require_a_left_vive_tracker():
    receiver = make_receiver_without_ros(hand_side="right")
    receiver._vive_left = None
    receiver._vive_left_time = None

    data = receiver.get_data()

    assert data["Left"] is not None
    assert data["Right"] is not None


def test_bimanual_recording_saves_both_vive_timestamps_and_metadata(tmp_path):
    receiver = make_receiver_without_ros(hand_side="bimanual")
    save_path = tmp_path / "teleop"

    receiver.start(save_path)
    receiver.get_data()
    receiver.stop()

    assert np.load(save_path / "vive_left_time.npy").shape == (1,)
    assert np.load(save_path / "vive_right_time.npy").shape == (1,)
    with open(save_path / "metadata.json") as stream:
        metadata = json.load(stream)
    assert metadata["mode"] == "vive_manus_bimanual"
    assert metadata["vive_left_topic"] == "/vive_trackers/left_hand/pose"
    assert metadata["vive_right_topic"] == "/vive_trackers/right_hand/pose"
    assert metadata["vive_left_frame_id"] == "world"


def test_left_manus_ergonomics_produces_pause_and_go_states():
    assert _pause_state_from_ergonomics(
        make_manus_message("Left", flex=60.0).ergonomics
    ) == 0
    assert _pause_state_from_ergonomics(
        make_manus_message("Left", flex=10.0).ergonomics
    ) == 1


def test_vive_arm_only_mode_does_not_require_a_robot_hand():
    retargetor = Retargetor(
        arm_name="xarm",
        hand_name=None,
        hand_side="Right",
    )
    retargetor.start(np.eye(4))
    data = {"Right": {"wrist": np.eye(4)}}
    retargetor.get_action(data)
    data["Right"]["wrist"][:3, 3] = [0.1, -0.2, 0.3]

    wrist_action, hand_action = retargetor.get_action(data)

    assert hand_action is None
    np.testing.assert_allclose(wrist_action[:3, 3], [0.1, -0.2, 0.3])
