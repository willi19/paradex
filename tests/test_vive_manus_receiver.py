from types import SimpleNamespace
import time
from threading import Event, Lock

import numpy as np

from paradex.io.teleop.vive.receiver import (
    _canonical_joint_name,
    apply_vive_tracker_mount_rotation,
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


def test_vive_mount_rotation_keeps_up_axis_and_reverses_other_axes():
    initial = np.eye(4)
    initial[:3, 3] = [0.4, -0.2, 1.1]
    corrected_initial = apply_vive_tracker_mount_rotation(initial)

    np.testing.assert_allclose(corrected_initial[:3, 3], initial[:3, 3])
    for axis, expected_sign in (("x", -1.0), ("y", 1.0), ("z", -1.0)):
        current = initial.copy()
        current[:3, :3] = _axis_rotation(axis, 0.2)
        corrected_current = apply_vive_tracker_mount_rotation(current)
        relative = (
            corrected_initial[:3, :3].T
            @ corrected_current[:3, :3]
        )
        np.testing.assert_allclose(
            relative,
            _axis_rotation(axis, expected_sign * 0.2),
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


def make_receiver_without_ros(sample_age=0.0):
    receiver = ViveManusROSReceiver.__new__(ViveManusROSReceiver)
    receiver.lock = Lock()
    receiver.save_event = Event()
    receiver.data = None
    receiver.max_age_s = 0.25
    receiver._vive_right = np.eye(4)
    receiver._manus_frames = {
        "Left": manus_message_to_frame(make_manus_message("Left")),
        "Right": manus_message_to_frame(make_manus_message("Right")),
    }
    timestamp = time.monotonic() - sample_age
    receiver._vive_right_time = timestamp
    receiver._manus_times = {"Left": timestamp, "Right": timestamp}
    return receiver


def test_get_data_fuses_fresh_right_vive_and_both_manus_hands():
    data = make_receiver_without_ros().get_data()

    assert data["Left"] is not None
    assert data["Right"] is not None
    np.testing.assert_allclose(data["Right"]["wrist"], np.eye(4))


def test_get_data_suppresses_commands_when_any_input_is_stale():
    data = make_receiver_without_ros(sample_age=1.0).get_data()

    assert data["Left"] is None
    assert data["Right"] is None


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
