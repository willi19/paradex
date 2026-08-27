from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from paradex.retargetor.experiment import replay_object_relative as core
from paradex.retargetor.experiment import replay_pose_retrieval as retrieval


def _episode(tmp_path: Path, episode_id: int, object_pose: np.ndarray) -> core.Episode:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0.5, 0.4, 0.5]
    return core.Episode(
        root=tmp_path / str(episode_id),
        arm_poses=poses,
        arm_times=np.array([0.0, 1.0, 2.0]),
        hand_commands=np.zeros((3, 16)),
        hand_times=np.array([0.0, 1.0, 2.0]),
        source_object_world=object_pose,
        source_c2r=np.eye(4),
    )


def test_foundpose_initialization_is_a_supported_episode_source_pose(tmp_path):
    pose_path = (
        tmp_path
        / "object_tracking_foundpose_gotrack"
        / "run"
        / "attempt_01"
        / "foundpose_init"
        / "init_pose_world.npy"
    )
    pose_path.parent.mkdir(parents=True)
    np.save(pose_path, np.eye(4))

    assert core._episode_source_pose_path(tmp_path, None) == pose_path


def test_dedicated_foundpose_initialization_is_preferred_over_gotrack_copy(tmp_path):
    tracking_root = tmp_path / "object_tracking_foundpose_gotrack"
    preferred = (
        tracking_root
        / "banana_0825_test4_foundpose_init_01"
        / "attempt_01"
        / "foundpose_init"
        / "init_pose_world.npy"
    )
    gotrack_copy = (
        tracking_root
        / "banana_0825_test4_gotrack_01"
        / "attempt_01"
        / "foundpose_init"
        / "init_pose_world.npy"
    )
    preferred.parent.mkdir(parents=True)
    gotrack_copy.parent.mkdir(parents=True)
    np.save(preferred, np.eye(4))
    np.save(gotrack_copy, np.eye(4))

    assert core._episode_source_pose_path(tmp_path, None) == preferred


def test_gotrack_loader_reads_only_requested_valid_world_poses(tmp_path):
    records_path = (
        tmp_path
        / "object_tracking_foundpose_gotrack"
        / "banana_test_gotrack_01"
        / "attempt_01"
        / "gotrack_tracking"
        / "gotrack_output"
        / "banana"
        / "world_pose_records.json"
    )
    records_path.parent.mkdir(parents=True)
    apex = np.eye(4)
    apex[0, 3] = 0.1
    release = np.eye(4)
    release[1, 3] = 0.2
    records_path.write_text(
        json.dumps(
            [
                {"frame_index": 4, "status": "ok", "pose_world": np.eye(4).tolist()},
                {"frame_index": 5, "status": "ok", "pose_world": apex.tolist()},
                {"frame_index": 8, "status": "ok", "pose_world": release.tolist()},
            ]
        ),
        encoding="utf-8",
    )

    actual_path, poses = retrieval.load_gotrack_world_poses(tmp_path, (5, 8))

    assert actual_path == records_path
    assert set(poses) == {5, 8}
    np.testing.assert_array_equal(poses[5], apex)
    np.testing.assert_array_equal(poses[8], release)


def test_gotrack_loader_rejects_invalid_requested_pose(tmp_path):
    records_path = (
        tmp_path
        / "object_tracking_foundpose_gotrack"
        / "banana_test_gotrack_01"
        / "attempt_01"
        / "gotrack_tracking"
        / "gotrack_output"
        / "banana"
        / "world_pose_records.json"
    )
    records_path.parent.mkdir(parents=True)
    records_path.write_text(
        json.dumps(
            [{"frame_index": 5, "status": "tracking_failed", "pose_world": None}]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="status='tracking_failed'"):
        retrieval.load_gotrack_world_poses(tmp_path, (5,))


def test_execution_refuses_large_camera_arm_sync_error():
    assert retrieval._check_camera_arm_sync(
        label="release",
        camera_timestamp=10.0,
        arm_timestamp=10.02,
        execute=False,
    ) == pytest.approx(0.02)

    with pytest.raises(RuntimeError, match="exceeds 50 ms"):
        retrieval._check_camera_arm_sync(
            label="release",
            camera_timestamp=10.0,
            arm_timestamp=10.051,
            execute=True,
        )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("Y", True), ("n", False), ("", False)],
)
def test_execution_confirmation_accepts_only_y(monkeypatch, answer, expected):
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)

    assert retrieval._execution_confirmed(Namespace(auto_execute=False)) is expected


def test_auto_execute_skips_confirmation(monkeypatch):
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("auto-execute must not prompt"),
    )

    assert retrieval._execution_confirmed(Namespace(auto_execute=True)) is True


def test_object_pose_retrieval_combines_translation_and_rotation(tmp_path):
    current = np.eye(4)
    translation_only = np.eye(4)
    translation_only[0, 3] = 0.04
    rotation_only = np.eye(4)
    angle = 0.2
    rotation_only[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]

    ranked = retrieval.rank_by_object_pose(
        current,
        [_episode(tmp_path, 6, translation_only), _episode(tmp_path, 7, rotation_only)],
        translation_scale_m=0.05,
        rotation_scale_rad=0.5,
    )

    assert [match.episode.root.name for match in ranked] == ["7", "6"]
    assert ranked[0].score == pytest.approx(0.4)
    assert ranked[1].score == pytest.approx(0.8)


def test_retrieved_replay_starts_from_the_annotated_synchronized_frame(
    tmp_path, monkeypatch
):
    current = np.eye(4)
    far_pose = np.eye(4)
    far_pose[0, 3] = 0.2
    selected_episode = _episode(tmp_path, 7, current)
    episodes = [_episode(tmp_path, 6, far_pose), selected_episode]
    live_pose = np.eye(4)
    live_qpos = np.zeros(6)
    live_hand = np.zeros(16)
    transformed_inputs = {}
    composed_inputs = {}

    monkeypatch.setattr(retrieval, "load_candidate_episodes", lambda *_: episodes)
    monkeypatch.setattr(
        retrieval,
        "_read_live_robot_state",
        lambda: (live_pose, live_qpos, live_hand),
    )

    def fake_transform(source, target, arm_poses):
        transformed_inputs["source"] = source
        transformed_inputs["target"] = target
        transformed_inputs["arm_poses"] = arm_poses
        return arm_poses

    monkeypatch.setattr(core, "relative_arm_actions", fake_transform)

    def fake_compose(_args, **kwargs):
        composed_inputs.update(kwargs)
        return core.ReplayTrajectory(
            arm_poses=np.concatenate(
                (kwargs["live_arm_pose"][None], kwargs["episode_arm_poses"]),
                axis=0,
            ),
            hand_actions=np.concatenate(
                (kwargs["live_hand_qpos"][None], kwargs["episode_hand_actions"]),
                axis=0,
            ),
            times=np.arange(len(kwargs["episode_arm_times"]) + 1, dtype=float),
            transition_frame_count=2,
            transition_seconds=1.0,
        )

    monkeypatch.setattr(core, "_compose_replay_trajectory", fake_compose)
    monkeypatch.setattr(
        retrieval.apex_core,
        "_load_start_annotations",
        lambda _root: ([6, 7], {6: 0, 7: 0}),
    )
    monkeypatch.setattr(
        retrieval.apex_core,
        "_load_apex_annotations",
        lambda _root: ([6, 7], {6: 1, 7: 1}),
    )
    monkeypatch.setattr(
        retrieval.apex_core,
        "_load_release_annotations",
        lambda _root: ([6, 7], {6: 2, 7: 2}),
    )
    monkeypatch.setattr(
        retrieval.apex_core,
        "_map_video_frame_to_arm_index",
        lambda _episode, frame, **_kwargs: (1, 1.0) if frame < 2 else (2, 2.0),
    )
    monkeypatch.setattr(
        retrieval,
        "load_gotrack_world_poses",
        lambda _root, frames: (
            tmp_path / "world_pose_records.json",
            {frame: np.eye(4) for frame in frames},
        ),
    )
    args = Namespace(
        episode_root=tmp_path,
        candidate_episodes=(6, 7),
        retrieval_translation_scale_m=0.05,
        retrieval_rotation_scale_rad=0.5,
        replay_output_dir=tmp_path / "output",
        replay_preview=False,
        execute=False,
        dish_clearance_m=0.02,
        dish_transfer_linear_speed_mps=1.0,
        dish_transfer_max_distance_m=0.5,
        dish_transfer_min_seconds=0.1,
        dish_transfer_rate_hz=10.0,
    )

    plan_path = retrieval.replay_closest_episode(
        args,
        current_object_robot=current,
        current_c2r=np.eye(4),
        dish_point_robot=np.array([0.1, 0.2, 0.3]),
    )

    assert plan_path.is_file()
    assert transformed_inputs["arm_poses"] is selected_episode.arm_poses
    np.testing.assert_array_equal(
        composed_inputs["episode_arm_poses"], selected_episode.arm_poses[1:]
    )
    np.testing.assert_array_equal(
        composed_inputs["episode_arm_times"], selected_episode.arm_times[1:]
    )
    with np.load(plan_path) as plan:
        assert plan["selected_episode"].item() == "7"
        assert plan["selected_frame"].item() == 1
        assert plan["start_video_frame"].item() == 0
        assert plan["start_arm_frame"].item() == 1
        assert plan["apex_frame_in_replay"].item() == 0
        assert plan["release_video_frame"].item() == 2
        assert plan["release_arm_frame"].item() == 2
        np.testing.assert_allclose(
            plan["predicted_release_object_robot"][:3, 3],
            [0.1, 0.2, 0.32],
        )
        np.testing.assert_array_equal(plan["dish_point_robot"], [0.1, 0.2, 0.3])


def test_dish_transfer_corrects_the_remaining_replay_suffix():
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    poses[:, :3, 3] = [
        [0.0, 0.0, 0.5],
        [0.1, 0.0, 0.5],
        [0.2, 0.0, 0.5],
        [0.3, 0.0, 0.5],
        [0.4, 0.0, 0.5],
        [0.5, 0.1, 0.5],
        [0.6, 0.2, 0.4],
    ]
    hands = np.repeat(np.arange(7, dtype=float)[:, None], 16, axis=1)
    trajectory = core.ReplayTrajectory(
        arm_poses=poses,
        hand_actions=hands,
        times=np.arange(7, dtype=float) * 0.1,
        transition_frame_count=3,
        transition_seconds=0.2,
    )

    mapped_apex_object = np.eye(4)
    mapped_apex_object[:3, 3] = [0.4, 0.0, 0.5]
    mapped_release_object = np.eye(4)
    mapped_release_object[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0, 0, 1]]
    mapped_release_object[:3, 3] = [0.6, 0.2, 0.4]

    result = retrieval.interleave_dish_transfer(
        trajectory,
        episode_apex_frame=2,
        mapped_apex_object_robot=mapped_apex_object,
        mapped_release_object_robot=mapped_release_object,
        dish_point_robot=np.array([1.0, 2.0, 0.3]),
        clearance_m=0.02,
        linear_speed_mps=10.0,
        max_translation_m=10.0,
        min_seconds=1.0,
        rate_hz=2.0,
    )

    assert result.apex_plan_frame == 4
    assert result.transfer_frame_count == 2
    np.testing.assert_allclose(result.target_eef_robot[:3, 3], [0.8, 1.8, 0.42])
    np.testing.assert_allclose(
        result.release_object_target_robot[:3, 3], [1.0, 2.0, 0.32]
    )
    np.testing.assert_allclose(
        result.predicted_release_object_robot,
        result.release_object_target_robot,
    )
    np.testing.assert_allclose(result.suffix_correction[:3, :3], np.eye(3))
    assert result.release_position_error_m < 1e-12
    np.testing.assert_allclose(result.trajectory.arm_poses[6], result.target_eef_robot)
    np.testing.assert_array_equal(
        result.trajectory.hand_actions[5:7], np.repeat(hands[4][None], 2, axis=0)
    )
    corrected_first_suffix = result.trajectory.arm_poses[7]
    np.testing.assert_allclose(
        np.linalg.inv(result.target_eef_robot) @ corrected_first_suffix,
        np.linalg.inv(poses[4]) @ poses[5],
    )
    np.testing.assert_allclose(
        result.trajectory.arm_poses[7:],
        result.suffix_correction[None] @ poses[5:],
    )
    assert np.all(np.diff(result.trajectory.times) >= 0)


def test_default_candidate_set_is_exactly_the_requested_episodes():
    assert retrieval.DEFAULT_CANDIDATE_EPISODES == (
        6,
        7,
        9,
        10,
        11,
        13,
        15,
        18,
        19,
        21,
        22,
        23,
        24,
        25,
        28,
        30,
    )
