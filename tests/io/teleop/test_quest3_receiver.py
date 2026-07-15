import json
import socket
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from paradex.io.teleop.occulus.receiver import (
    QUEST_JOINT_ORDER,
    QUEST_TO_GLOBAL,
    Quest3Receiver,
)
from paradex.retargetor.unimanual import Retargetor


def _packet(
    wrist_position,
    control_state=0,
    control_tracked=True,
    frame=1,
    missing_joint_indices=(),
):
    wrist_position = np.asarray(wrist_position, dtype=float)
    positions = []
    for index, _ in enumerate(QUEST_JOINT_ORDER[:21]):
        offset = np.array(
            [0.002 * (index % 4), 0.003 * (index % 3), 0.01 * index]
        )
        positions.extend((wrist_position + offset).tolist())
    return {
        "hand": "right",
        "f": frame,
        "t": float(frame),
        "control_state": control_state,
        "control_tracked": control_tracked,
        "control_gesture": "open" if control_state else "fist",
        "m": [0 if index in missing_joint_indices else 1 for index in range(21)],
        "q": [0.0, 0.0, 0.0, 1.0] * 21,
        "jp": positions,
    }


def _send(receiver, packet):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(
            json.dumps(packet).encode("utf-8"),
            ("127.0.0.1", receiver.port),
        )
    finally:
        sock.close()


def _wait_for_wrist(receiver, quest_wrist_position, timeout_s=1.0):
    expected = QUEST_TO_GLOBAL @ np.asarray(quest_wrist_position, dtype=float)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        data = receiver.get_data()
        if data["Right"] is not None:
            actual = data["Right"]["wrist"][:3, 3]
            if np.allclose(actual, expected):
                return data
        time.sleep(0.005)
    raise AssertionError("Quest receiver did not expose the expected wrist frame")


def test_packet_is_adapted_to_paradex_hand_pose_contract():
    receiver = Quest3Receiver(port=0)
    try:
        wrist = np.array([0.2, 1.1, 0.4])
        _send(receiver, _packet(wrist))
        data = _wait_for_wrist(receiver, wrist)

        hand = data["Right"]
        assert hand["wrist"].shape == (4, 4)
        assert hand["wrist"][:3, 3] == pytest.approx([0.4, -0.2, 1.1])
        assert "pinky_distal" in hand
        assert np.linalg.det(hand["index_distal"][:3, :3]) == pytest.approx(1.0)
        assert data["Left"] is None
    finally:
        receiver.end()


def test_untracked_control_keeps_the_last_open_or_fist_state():
    receiver = Quest3Receiver(port=0)
    try:
        wrist = [0.1, 1.0, 0.3]
        _send(receiver, _packet(wrist, control_state=1, control_tracked=True, frame=1))
        _wait_for_wrist(receiver, wrist)
        assert receiver.get_control_state() == 1

        _send(receiver, _packet(wrist, control_state=0, control_tracked=False, frame=2))
        deadline = time.monotonic() + 1.0
        while receiver.get_data()["control_tracked"] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert receiver.get_control_state() == 1

        _send(receiver, _packet(wrist, control_state=0, control_tracked=True, frame=3))
        deadline = time.monotonic() + 1.0
        while receiver.get_control_state() != 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert receiver.get_control_state() == 0
    finally:
        receiver.end()


def test_stale_hand_pose_is_not_returned():
    receiver = Quest3Receiver(port=0, max_age_s=0.02)
    try:
        wrist = [0.1, 1.0, 0.3]
        _send(receiver, _packet(wrist))
        _wait_for_wrist(receiver, wrist)
        time.sleep(0.04)
        assert receiver.get_data()["Right"] is None
    finally:
        receiver.end()


def test_missing_finger_joints_reuse_the_last_pose_in_the_current_wrist_frame():
    receiver = Quest3Receiver(port=0, extrapolation_horizon_s=0.0)
    try:
        initial_wrist = np.array([0.1, 1.0, 0.3])
        _send(receiver, _packet(initial_wrist, frame=1))
        initial = _wait_for_wrist(receiver, initial_wrist)
        initial_joint = initial["Right"]["index_distal"].copy()

        moved_wrist = initial_wrist + [0.05, 0.0, 0.0]
        _send(
            receiver,
            _packet(moved_wrist, frame=2, missing_joint_indices=(8,)),
        )
        moved = _wait_for_wrist(receiver, moved_wrist)

        wrist_delta = (
            moved["Right"]["wrist"]
            @ np.linalg.inv(initial["Right"]["wrist"])
        )
        assert moved["Right"]["index_distal"] == pytest.approx(
            wrist_delta @ initial_joint
        )
    finally:
        receiver.end()


def test_short_tracking_gap_extrapolates_wrist_and_hand_with_bounded_velocity():
    receiver = Quest3Receiver(
        port=0,
        max_age_s=0.25,
        extrapolation_delay_s=0.0,
        extrapolation_horizon_s=0.10,
        max_linear_speed_m_s=1.0,
        max_angular_speed_rad_s=2.0,
    )
    try:
        pose = {"wrist": np.eye(4), "index_distal": np.eye(4)}
        pose["index_distal"][:3, 3] = [0.0, 0.0, 0.2]

        predicted = receiver._extrapolate_hand_pose(
            pose,
            linear_velocity=np.array([0.5, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 1.0]),
            age_s=0.05,
        )
        expected_rotation = R.from_rotvec([0.0, 0.0, 0.05]).as_matrix()
        assert predicted["wrist"][:3, 3] == pytest.approx([0.025, 0.0, 0.0])
        assert predicted["wrist"][:3, :3] == pytest.approx(expected_rotation)
        assert predicted["index_distal"] == pytest.approx(
            predicted["wrist"] @ pose["index_distal"]
        )

        capped = receiver._extrapolate_hand_pose(
            pose,
            linear_velocity=np.array([0.5, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            age_s=0.20,
        )
        assert capped["wrist"][:3, 3] == pytest.approx([0.05, 0.0, 0.0])
    finally:
        receiver.end()


def test_recording_writes_original_quest_packets(tmp_path):
    receiver = Quest3Receiver(port=0)
    try:
        wrist = [0.1, 1.0, 0.3]
        receiver.start(str(tmp_path))
        _send(receiver, _packet(wrist, frame=17))
        _wait_for_wrist(receiver, wrist)
        receiver.stop()

        lines = (tmp_path / "quest_packets.jsonl").read_text().splitlines()
        assert len(lines) == 1
        saved = json.loads(lines[0])
        assert saved["f"] == 17
        assert saved["received_at"] > 0
    finally:
        receiver.end()


def test_pause_resume_rebases_relative_wrist_motion_without_jump():
    receiver = Quest3Receiver(port=0)
    retargetor = Retargetor(
        arm_name="xarm", hand_name="inspire_f1", hand_side="Right"
    )
    try:
        home_pose = np.eye(4)
        retargetor.start(home_pose)

        initial = np.array([0.2, 1.0, 0.4])
        _send(receiver, _packet(initial, control_state=0, frame=1))
        active = _wait_for_wrist(receiver, initial)
        initial_target, hand_action = retargetor.get_action(active)
        assert initial_target == pytest.approx(home_pose)
        assert hand_action.shape == (6,)
        assert np.all((hand_action >= 0) & (hand_action <= 1800))

        retargetor.stop()
        resume_source = initial + [0.3, -0.2, 0.5]
        _send(receiver, _packet(resume_source, control_state=1, frame=2))
        _wait_for_wrist(receiver, resume_source)

        _send(receiver, _packet(resume_source, control_state=0, frame=3))
        resumed = _wait_for_wrist(receiver, resume_source)
        resumed_target, _ = retargetor.get_action(resumed)
        assert resumed_target == pytest.approx(home_pose)

        moved_source = resume_source + [0.0, 0.05, 0.0]
        _send(receiver, _packet(moved_source, control_state=0, frame=4))
        moved = _wait_for_wrist(receiver, moved_source)
        moved_target, _ = retargetor.get_action(moved)
        assert moved_target[:3, 3] == pytest.approx([0.0, 0.0, 0.05])
    finally:
        receiver.end()
