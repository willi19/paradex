from pathlib import Path
from types import SimpleNamespace

import numpy as np

from paradex.retargetor.experiment import replay_object_relative as core
from paradex.retargetor.experiment import replay_object_relative_episode as explicit


def test_explicit_replay_parser_requires_episode_number():
    args = explicit._parser().parse_args(
        ["--object", "pink_clock", "--episode", "3", "--preview"]
    )

    assert args.object == "pink_clock"
    assert args.episode == 3
    assert args.preview is True


def test_selected_episode_loader_uses_only_the_requested_numeric_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "allegro_v5" / "pink_clock" / "3"
    root.mkdir(parents=True)
    episode = core.Episode(
        root=root,
        arm_poses=np.eye(4)[None],
        arm_times=np.array([0.0]),
        hand_commands=np.zeros((1, 16)),
        hand_times=np.array([0.0]),
        source_object_world=np.eye(4),
        source_c2r=np.eye(4),
    )
    loaded_roots = []

    def fake_load(_args, selected_root):
        loaded_roots.append(selected_root)
        return episode

    monkeypatch.setattr(core, "_load_episode_root", fake_load)
    args = SimpleNamespace(
        capture_root=Path(tmp_path),
        robot="allegro_v5",
        object="pink_clock",
        episode=3,
        source_object_pose=None,
    )

    actual = explicit._load_selected_episode(args)

    assert actual is episode
    assert loaded_roots == [root]


def test_preview_state_uses_temporary_read_only_receivers(monkeypatch):
    arm_pose = np.eye(4)
    arm_qpos = np.arange(6, dtype=float)
    hand_qpos = np.arange(16, dtype=float)
    calls = []

    monkeypatch.setattr(
        core,
        "_live_xarm_preview_state",
        lambda: calls.append("xarm read-only") or (arm_pose, arm_qpos),
    )
    monkeypatch.setattr(
        core,
        "_live_allegro_v5_preview_qpos",
        lambda: calls.append("allegro read-only") or hand_qpos,
    )

    actual = explicit._read_preview_robot_state()

    np.testing.assert_array_equal(actual[0], arm_pose)
    np.testing.assert_array_equal(actual[1], arm_qpos)
    np.testing.assert_array_equal(actual[2], hand_qpos)
    assert calls == ["xarm read-only", "allegro read-only"]
