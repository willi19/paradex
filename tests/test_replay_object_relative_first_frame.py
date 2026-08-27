import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.retargetor.experiment import replay_object_relative as core
from paradex.retargetor.experiment import (
    replay_object_relative_first_frame as first_frame,
)


def test_parser_has_no_teleoperation_or_replay_key_options():
    parser = first_frame._parser()
    args = parser.parse_args(["--preview"])

    assert args.object == "banana"
    assert args.mesh_name == "banana"
    assert args.preview is True
    assert args.execute is False
    assert not hasattr(args, "replay_key")
    assert not hasattr(args, "allegro_command_rate_hz")


def test_first_frame_state_uses_read_only_live_receivers(monkeypatch):
    arm_pose = np.eye(4)
    arm_qpos = np.arange(6, dtype=float)
    hand_qpos = np.arange(16, dtype=float)
    calls = []

    monkeypatch.setattr(
        core,
        "_live_xarm_preview_state",
        lambda: calls.append("arm") or (arm_pose, arm_qpos),
    )
    monkeypatch.setattr(
        core,
        "_live_allegro_v5_preview_qpos",
        lambda: calls.append("hand") or hand_qpos,
    )

    actual = first_frame._read_first_frame_robot_state()

    np.testing.assert_array_equal(actual[0], arm_pose)
    np.testing.assert_array_equal(actual[1], arm_qpos)
    np.testing.assert_array_equal(actual[2], hand_qpos)
    assert calls == ["arm", "hand"]


def test_matching_uses_only_episode_frame_zero(tmp_path):
    current_object = np.eye(4)
    current_wrist = np.eye(4)
    current_wrist[0, 3] = 0.3
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0.6, 0.3, 0.1]
    episode = core.Episode(
        root=tmp_path / "7",
        arm_poses=poses,
        arm_times=np.arange(3, dtype=float),
        hand_commands=np.zeros((3, 16)),
        hand_times=np.arange(3, dtype=float),
        source_object_world=np.eye(4),
        source_c2r=np.eye(4),
    )

    matches = first_frame._rank_episode_frame_zero_matches(
        current_object,
        current_wrist,
        [episode],
        position_scale_m=0.05,
        rotation_scale_rad=0.5,
    )

    assert matches[0].frame_index == 0
    assert matches[0].wrist_object_distance_m == 0.6
    assert matches[0].distance_delta_m == 0.3


def test_candidate_loader_indexes_only_apex_target_episodes(tmp_path, monkeypatch):
    (tmp_path / "apex_annotations.json").write_text(
        json.dumps(
            {
                "target_episodes": [6, 9],
                "annotations": {"6": 30, "9": 40, "99": 50},
            }
        )
    )
    for episode_id in (6, 9, 99):
        (tmp_path / str(episode_id)).mkdir()

    loaded = []

    def fake_load(root):
        loaded.append(int(root.name))
        return core.Episode(
            root=root,
            arm_poses=np.eye(4)[None],
            arm_times=np.array([0.0]),
            hand_commands=np.zeros((1, 16)),
            hand_times=np.array([0.0]),
            source_object_world=np.eye(4),
            source_c2r=np.eye(4),
        )

    monkeypatch.setattr(first_frame, "_load_banana_episode", fake_load)

    episodes, annotations = first_frame._load_banana_candidates(tmp_path)

    assert loaded == [6, 9]
    assert [episode.root.name for episode in episodes] == ["6", "9"]
    assert annotations == {6: 30, 9: 40}


def test_start_position_annotations_use_the_same_camera_frame_schema(tmp_path):
    (tmp_path / "start_position.json").write_text(
        json.dumps(
            {
                "target_episodes": [6, 9],
                "annotations": {"6": 16, "9": 24},
            }
        )
    )

    episode_ids, annotations = first_frame._load_start_annotations(tmp_path)

    assert episode_ids == [6, 9]
    assert annotations == {6: 16, 9: 24}


def test_right_before_release_annotations_use_the_same_camera_frame_schema(tmp_path):
    (tmp_path / "right_before_release.json").write_text(
        json.dumps(
            {
                "target_episodes": [6, 9],
                "annotations": {"6": 35, "9": 43},
            }
        )
    )

    episode_ids, annotations = first_frame._load_release_annotations(tmp_path)

    assert episode_ids == [6, 9]
    assert annotations == {6: 35, 9: 43}


def test_apex_video_frame_maps_to_nearest_arm_timestamp(tmp_path):
    timestamp_dir = tmp_path / "8" / "raw" / "timestamps"
    timestamp_dir.mkdir(parents=True)
    np.save(timestamp_dir / "timestamp.npy", np.array([10.0, 10.1, 10.3]))
    np.save(timestamp_dir / "frame_id.npy", np.array([1, 2, 4]))
    episode = core.Episode(
        root=tmp_path / "8",
        arm_poses=np.repeat(np.eye(4)[None], 4, axis=0),
        arm_times=np.array([10.02, 10.08, 10.27, 10.40]),
        hand_commands=np.zeros((4, 16)),
        hand_times=np.arange(4, dtype=float),
        source_object_world=np.eye(4),
        source_c2r=np.eye(4),
    )

    arm_index, apex_time = first_frame._map_apex_video_frame_to_arm_index(episode, 3)

    assert arm_index == 2
    assert apex_time == 10.3


def test_pause_repeats_apex_pose_for_one_second_and_shifts_suffix():
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    poses[:, 0, 3] = np.arange(7)
    hands = np.repeat(np.arange(7, dtype=float)[:, None], 16, axis=1)
    trajectory = core.ReplayTrajectory(
        arm_poses=poses,
        hand_actions=hands,
        times=np.arange(7, dtype=float) * 0.1,
        transition_frame_count=3,
        transition_seconds=0.2,
    )

    paused, plan_index, hold_count = first_frame._insert_trajectory_pause(
        trajectory, episode_frame_index=2, pause_seconds=1.0
    )

    assert plan_index == 4
    assert hold_count == 10
    assert len(paused.times) == 17
    np.testing.assert_array_equal(
        paused.arm_poses[4:15], np.repeat(poses[4][None], 11, axis=0)
    )
    np.testing.assert_array_equal(
        paused.hand_actions[4:15], np.repeat(hands[4][None], 11, axis=0)
    )
    assert paused.times[4] == 0.4
    assert paused.times[14] == 1.4
    assert paused.times[15] == 1.5
    np.testing.assert_array_equal(paused.arm_poses[15:], poses[5:])
