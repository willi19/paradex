from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import trimesh

from paradex.retargetor.experiment import replay_object_relative as replay_core
from paradex.retargetor.experiment import grasp_indexing_visualization as grasp_vis
from paradex.retargetor.experiment.grasp_indexing_visualization import (
    PreloadedEpisodeHand,
    MiddlePedalPlayTrigger,
    ReplaySelection,
    _build_top_candidate_replay,
    _execute_replay_if_confirmed,
    _preview_top_candidate_replay,
    _place_preloaded_hands,
    _parser,
    closest_approach_frame,
    discover_episode_ids,
    hand_pose_from_eef,
    load_apex_arm_frames,
    load_or_create_grasp_frame_cache,
    rank_episode_grasp_frame_matches,
    resolve_candidate_episode_ids,
)
from paradex.retargetor.experiment.replay_pose_retrieval import (
    DEFAULT_CANDIDATE_EPISODES,
    DEFAULT_EPISODE_ROOT,
)


def _pose(x: float, yaw: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    cosine, sine = np.cos(yaw), np.sin(yaw)
    pose[:2, :2] = ((cosine, -sine), (sine, cosine))
    return pose


def _episode(episode_id: int, frame_zero_x: float) -> replay_core.Episode:
    return replay_core.Episode(
        root=Path(str(episode_id)),
        arm_poses=np.stack((_pose(frame_zero_x), _pose(frame_zero_x - 0.05))),
        arm_times=np.array((0.0, 1.0)),
        hand_commands=np.zeros((2, 16), dtype=np.float64),
        hand_times=np.array((0.0, 1.0)),
        source_object_world=np.eye(4, dtype=np.float64),
        source_c2r=np.eye(4, dtype=np.float64),
    )


def test_discovery_includes_every_numeric_episode_directory(tmp_path: Path) -> None:
    (tmp_path / "10").mkdir()
    (tmp_path / "2").mkdir()
    (tmp_path / "notes").mkdir()

    assert discover_episode_ids(tmp_path) == (2, 10)


def test_no_teleop_flag_is_available() -> None:
    assert _parser().parse_args(["--no-teleop"]).no_teleop is True


def test_replay_preview_is_enabled_by_default_and_can_be_disabled() -> None:
    assert _parser().parse_args([]).replay_preview is True
    assert _parser().parse_args(["--no-replay-preview"]).replay_preview is False


def test_auto_execute_flag_is_available() -> None:
    args = _parser().parse_args(["--execute", "--auto-execute"])
    assert args.execute is True
    assert args.auto_execute is True


def test_pedal_play_is_enabled_by_default_and_can_be_disabled() -> None:
    assert _parser().parse_args([]).pedal_play is True
    assert _parser().parse_args(["--no-pedal-play"]).pedal_play is False


def test_middle_pedal_triggers_once_per_press_edge() -> None:
    class FakePedalState:
        def __init__(self) -> None:
            self.state = 1
            self.closed = False

        def get_state(self) -> int:
            return self.state

        def close(self) -> None:
            self.closed = True

    state = FakePedalState()
    trigger = MiddlePedalPlayTrigger(state)
    assert trigger.poll_pressed() is False
    state.state = 0
    assert trigger.poll_pressed() is True
    assert trigger.poll_pressed() is False
    state.state = 1
    assert trigger.poll_pressed() is False
    state.state = 0
    assert trigger.poll_pressed() is True
    trigger.close()
    assert state.closed is True


def test_default_dataset_uses_capture_sil_candidate_set() -> None:
    assert (
        resolve_candidate_episode_ids(DEFAULT_EPISODE_ROOT, None)
        == DEFAULT_CANDIDATE_EPISODES
    )


def test_explicit_candidate_set_is_preserved() -> None:
    assert resolve_candidate_episode_ids(DEFAULT_EPISODE_ROOT, (7, 9)) == (7, 9)


def test_hand_pose_applies_fixed_palm_mount_to_eef() -> None:
    eef = _pose(0.4)
    palm_eef = np.eye(4)
    palm_eef[2, 3] = 0.146

    hand_pose = hand_pose_from_eef(eef, palm_eef)

    np.testing.assert_allclose(hand_pose[:3, 3], (0.4, 0.0, 0.146))


def test_preloaded_grasp_frame_hand_is_placed_in_current_object_frame() -> None:
    episode = _episode(7, 0.2)
    hand_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    preloaded = PreloadedEpisodeHand(
        episode=episode,
        frame_index=1,
        hand_qpos=np.ones(16),
        hand_mesh=hand_mesh,
    )

    candidates = _place_preloaded_hands(
        _pose(1.0),
        np.eye(4),
        [preloaded],
    )

    assert len(candidates) == 1
    assert candidates[0].hand_mesh is hand_mesh
    assert candidates[0].frame_index == 1
    np.testing.assert_allclose(candidates[0].eef_robot, _pose(1.15))


def test_closest_approach_frame_uses_minimum_wrist_object_distance() -> None:
    frame_index, distance = closest_approach_frame(_episode(7, 0.2))

    assert frame_index == 1
    np.testing.assert_allclose(distance, 0.15)


def test_grasp_frame_cache_is_written_then_reused(tmp_path: Path) -> None:
    cache_path = tmp_path / "grasp_index_frames.json"
    original = _episode(7, 0.2)

    assert load_or_create_grasp_frame_cache([original], cache_path) == {"7": 1}
    changed = _episode(7, 0.01)
    assert closest_approach_frame(changed)[0] == 0
    assert load_or_create_grasp_frame_cache([changed], cache_path) == {"7": 1}


def test_live_wrist_rotation_changes_top_candidate() -> None:
    first = _episode(1, 0.1)
    second = replace(
        _episode(2, 0.4),
        arm_poses=np.stack((_pose(0.4, 1.0), _pose(0.35, 1.0))),
    )

    near_first = rank_episode_grasp_frame_matches(
        np.eye(4),
        _pose(4.0, 0.1),
        [first, second],
    )
    near_second = rank_episode_grasp_frame_matches(
        np.eye(4),
        _pose(-4.0, 0.9),
        [first, second],
    )

    assert near_first[0].episode.root.name == "1"
    assert near_second[0].episode.root.name == "2"


def test_matching_uses_cached_grasp_frame_instead_of_frame_zero() -> None:
    first = replace(
        _episode(1, 0.1),
        arm_poses=np.stack((_pose(0.1, 0.0), _pose(0.05, 1.0))),
    )
    second = replace(
        _episode(2, 0.4),
        arm_poses=np.stack((_pose(0.4, 1.0), _pose(0.35, 0.0))),
    )

    ranked = rank_episode_grasp_frame_matches(
        np.eye(4),
        _pose(8.0, 0.0),
        [first, second],
        candidate_frame_indices={"1": 1, "2": 1},
    )

    assert ranked[0].episode.root.name == "2"
    assert ranked[0].frame_index == 1


def test_matching_rejects_missing_cached_grasp_frame() -> None:
    with pytest.raises(KeyError, match="episode 2"):
        rank_episode_grasp_frame_matches(
            np.eye(4),
            _pose(0.1),
            [_episode(1, 0.1), _episode(2, 0.2)],
            candidate_frame_indices={"1": 1},
        )


def test_hand_state_does_not_affect_rotation_only_score() -> None:
    first = _episode(1, 0.2)
    second = replace(
        _episode(2, 0.2),
        hand_commands=np.ones((2, 16), dtype=np.float64),
    )

    ranked = rank_episode_grasp_frame_matches(
        np.eye(4),
        _pose(0.2),
        [first, second],
        candidate_frame_indices={"1": 0, "2": 0},
    )

    assert ranked[0].episode.root.name == "1"
    assert ranked[0].score == ranked[1].score == 0.0


def test_matching_is_relative_to_current_object_pose() -> None:
    first = _episode(1, 0.1)
    second = replace(
        _episode(2, 0.4),
        arm_poses=np.stack((_pose(0.4, 1.0), _pose(0.35, 1.0))),
    )
    current_object = _pose(1.0, 0.5)

    ranked = rank_episode_grasp_frame_matches(
        current_object,
        _pose(1.39, 1.5),
        [first, second],
    )

    assert ranked[0].episode.root.name == "2"
    np.testing.assert_allclose(ranked[0].rotation_error_rad, 0.0, atol=1e-7)


def test_no_teleop_continuously_updates_read_only_robot_state(monkeypatch) -> None:
    class FakeExitEvent:
        def __init__(self) -> None:
            self.wait_count = 0

        def is_set(self) -> bool:
            return self.wait_count >= 2

        def wait(self, _: float) -> bool:
            self.wait_count += 1
            return self.is_set()

    class FakeMonitor:
        instance = None

        def __init__(self) -> None:
            self.closed = False
            self.read_count = 0
            FakeMonitor.instance = self

        def get_state(self):
            self.read_count += 1
            return _pose(float(self.read_count)), np.full(16, self.read_count)

        def close(self) -> None:
            self.closed = True

    class FakeVisualization:
        args = SimpleNamespace(visualization_rate_hz=20.0)

        def __init__(self) -> None:
            self.states = []
            self.replay_event = FakeExitEvent()

        def update_robot_state(self, arm_pose, hand_qpos) -> None:
            self.states.append((arm_pose, hand_qpos))

        def replay_selection(self):
            return None

    monkeypatch.setattr(grasp_vis, "Event", FakeExitEvent)
    monkeypatch.setattr(grasp_vis, "_poll_terminal_command", lambda: None)
    monkeypatch.setattr(grasp_vis, "ReadOnlyRobotStateMonitor", FakeMonitor)
    visualization = FakeVisualization()

    grasp_vis._run_visualization_only(visualization)

    assert len(visualization.states) == 2
    assert FakeMonitor.instance.read_count == 2
    assert FakeMonitor.instance.closed is True


def test_top_candidate_replay_starts_at_cached_frame_and_saves_plan(
    tmp_path: Path,
) -> None:
    episode = replace(
        _episode(7, 0.3),
        arm_poses=np.stack(
            (_pose(0.3), _pose(0.2), _pose(0.1), _pose(0.0))
        ),
        arm_times=np.array((0.0, 1.0, 2.0, 3.0)),
        hand_commands=np.stack(
            (
                np.zeros(16),
                np.ones(16),
                np.full(16, 2.0),
                np.full(16, 3.0),
            )
        ),
        hand_times=np.array((0.0, 1.0, 2.0, 3.0)),
    )
    args = SimpleNamespace(
        rate_scale=1.0,
        approach_linear_speed_mps=0.5,
        approach_angular_speed_rps=2.0,
        approach_min_seconds=0.1,
        approach_rate_hz=10.0,
    )
    selection = ReplaySelection(
        episode_id="7",
        frame_index=1,
        live_arm_pose=_pose(0.8),
        live_hand_qpos=np.zeros(16),
    )

    plan_path, trajectory = _build_top_candidate_replay(
        args,
        current_object_robot=_pose(1.0),
        current_c2r=np.eye(4),
        episodes=[episode],
        selection=selection,
        apex_frame_indices={"7": 2},
        apex_video_frames={"7": 42},
        output_dir=tmp_path,
    )

    saved = np.load(plan_path)
    assert saved["selected_episode"].item() == "7"
    assert saved["selected_frame"].item() == 1
    assert saved["apex_video_frame"].item() == 42
    assert saved["apex_arm_frame"].item() == 2
    assert len(saved["episode_arm_action"]) == 2
    np.testing.assert_allclose(saved["episode_arm_action"][0], _pose(1.2))
    np.testing.assert_allclose(saved["episode_hand_action"][0], np.ones(16))
    np.testing.assert_allclose(trajectory.arm_poses[0], selection.live_arm_pose)
    np.testing.assert_allclose(trajectory.arm_poses[-1], _pose(1.1))

    unsaved_path, immediate_trajectory = _build_top_candidate_replay(
        args,
        current_object_robot=_pose(1.0),
        current_c2r=np.eye(4),
        episodes=[episode],
        selection=selection,
        apex_frame_indices={"7": 2},
        apex_video_frames={"7": 42},
        output_dir=tmp_path / "immediate",
        persist_plan=False,
    )
    assert unsaved_path is None
    assert not (tmp_path / "immediate").exists()
    np.testing.assert_allclose(immediate_trajectory.arm_poses, trajectory.arm_poses)


def test_apex_annotations_are_mapped_to_arm_frames(monkeypatch, tmp_path: Path) -> None:
    episodes = [_episode(7, 0.2), _episode(9, 0.3)]
    monkeypatch.setattr(
        grasp_vis.apex_core,
        "_load_apex_annotations",
        lambda _: ([7, 9], {7: 70, 9: 90}),
    )
    monkeypatch.setattr(
        grasp_vis.apex_core,
        "_map_video_frame_to_arm_index",
        lambda episode, frame, label: (frame // 10, float(frame)),
    )

    arm_frames, video_frames = load_apex_arm_frames(tmp_path, episodes)

    assert arm_frames == {"7": 7, "9": 9}
    assert video_frames == {"7": 70, "9": 90}


def test_replay_rejects_apex_before_cached_frame(tmp_path: Path) -> None:
    episode = _episode(7, 0.3)
    selection = ReplaySelection(
        episode_id="7",
        frame_index=1,
        live_arm_pose=_pose(0.8),
        live_hand_qpos=np.zeros(16),
    )
    args = SimpleNamespace(
        rate_scale=1.0,
        approach_linear_speed_mps=0.5,
        approach_angular_speed_rps=2.0,
        approach_min_seconds=0.1,
        approach_rate_hz=10.0,
    )

    with pytest.raises(ValueError, match="apex arm frame 0"):
        _build_top_candidate_replay(
            args,
            current_object_robot=np.eye(4),
            current_c2r=np.eye(4),
            episodes=[episode],
            selection=selection,
            apex_frame_indices={"7": 0},
            apex_video_frames={"7": 10},
            output_dir=tmp_path,
        )


def test_execution_requires_exact_play_confirmation(monkeypatch) -> None:
    trajectory = replay_core.ReplayTrajectory(
        arm_poses=np.stack((_pose(0.0), _pose(0.1))),
        hand_actions=np.zeros((2, 16)),
        times=np.array((0.0, 1.0)),
        transition_frame_count=2,
        transition_seconds=1.0,
    )
    calls = []
    monkeypatch.setattr(replay_core, "_execute", lambda *args: calls.append(args))
    monkeypatch.setattr("builtins.input", lambda _: "play")

    assert _execute_replay_if_confirmed(
        SimpleNamespace(execute=True), trajectory
    ) is False
    assert calls == []

    monkeypatch.setattr("builtins.input", lambda _: "PLAY")
    assert _execute_replay_if_confirmed(
        SimpleNamespace(execute=True), trajectory
    ) is True
    assert len(calls) == 1

    monkeypatch.setattr(
        "builtins.input",
        lambda _: pytest.fail("auto execute must not request terminal input"),
    )
    assert _execute_replay_if_confirmed(
        SimpleNamespace(execute=True, auto_execute=True), trajectory
    ) is True
    assert len(calls) == 2


def test_auto_execute_without_execute_remains_plan_only(monkeypatch) -> None:
    trajectory = replay_core.ReplayTrajectory(
        arm_poses=np.stack((_pose(0.0), _pose(0.1))),
        hand_actions=np.zeros((2, 16)),
        times=np.array((0.0, 1.0)),
        transition_frame_count=2,
        transition_seconds=1.0,
    )
    monkeypatch.setattr(
        replay_core,
        "_execute",
        lambda *args: pytest.fail("--auto-execute alone must not command the robot"),
    )

    assert _execute_replay_if_confirmed(
        SimpleNamespace(execute=False, auto_execute=True), trajectory
    ) is False


def test_execution_reuses_supplied_teleoperation_controllers(monkeypatch) -> None:
    trajectory = replay_core.ReplayTrajectory(
        arm_poses=np.stack((_pose(0.0), _pose(0.1))),
        hand_actions=np.zeros((2, 16)),
        times=np.array((0.0, 1.0)),
        transition_frame_count=2,
        transition_seconds=1.0,
    )
    arm = object()
    hand = object()
    calls = []
    monkeypatch.setattr(
        replay_core,
        "_execute",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert _execute_replay_if_confirmed(
        SimpleNamespace(execute=True, auto_execute=True),
        trajectory,
        arm=arm,
        hand=hand,
    ) is True
    assert len(calls) == 1
    assert calls[0][1] == {"arm": arm, "hand": hand}


def test_pedal_execution_is_immediate_without_terminal_confirmation(
    monkeypatch,
) -> None:
    trajectory = replay_core.ReplayTrajectory(
        arm_poses=np.stack((_pose(0.0), _pose(0.1))),
        hand_actions=np.zeros((2, 16)),
        times=np.array((0.0, 1.0)),
        transition_frame_count=2,
        transition_seconds=1.0,
    )
    calls = []
    monkeypatch.setattr(
        "builtins.input",
        lambda _: pytest.fail("pedal fast path must not request PLAY input"),
    )
    monkeypatch.setattr(replay_core, "_execute", lambda *args: calls.append(args))

    assert _execute_replay_if_confirmed(
        SimpleNamespace(execute=True), trajectory, immediate=True
    ) is True
    assert len(calls) == 1


def test_pedal_replay_cycle_skips_preview_and_plan_persistence(
    monkeypatch, tmp_path: Path
) -> None:
    trajectory = replay_core.ReplayTrajectory(
        arm_poses=np.stack((_pose(0.0), _pose(0.1))),
        hand_actions=np.zeros((2, 16)),
        times=np.array((0.0, 1.0)),
        transition_frame_count=2,
        transition_seconds=1.0,
    )
    selection = ReplaySelection(
        episode_id="3",
        frame_index=8,
        live_arm_pose=np.eye(4),
        live_hand_qpos=np.zeros(16),
        trigger="pedal",
    )
    build_calls = []
    execute_calls = []
    monkeypatch.setattr(
        grasp_vis,
        "_build_top_candidate_replay",
        lambda *args, **kwargs: (
            build_calls.append(kwargs) or (None, trajectory)
        ),
    )
    monkeypatch.setattr(
        grasp_vis,
        "_preview_top_candidate_replay",
        lambda *args, **kwargs: pytest.fail("pedal fast path must skip preview"),
    )
    monkeypatch.setattr(
        grasp_vis,
        "_execute_replay_if_confirmed",
        lambda *args, **kwargs: execute_calls.append(kwargs) or True,
    )
    session = SimpleNamespace(arm=object(), hand=object())

    assert grasp_vis._run_selected_replay_cycle(
        SimpleNamespace(execute=True, replay_preview=True),
        current_object_robot=np.eye(4),
        current_c2r=np.eye(4),
        episodes=[],
        candidate_frame_indices={},
        apex_frame_indices={},
        apex_video_frames={},
        selection=selection,
        output_dir=tmp_path,
        pedal_trigger=None,
        session=session,
    ) is True
    assert build_calls[0]["persist_plan"] is False
    assert execute_calls[0]["immediate"] is True
    assert execute_calls[0]["arm"] is session.arm
    assert execute_calls[0]["hand"] is session.hand


def test_teleoperation_retains_initialized_controllers_for_replay(
    monkeypatch,
) -> None:
    selection = ReplaySelection(
        episode_id="4",
        frame_index=12,
        live_arm_pose=np.eye(4),
        live_hand_qpos=np.zeros(16),
    )

    class FakeArm:
        def __init__(self) -> None:
            self.stop_motion_count = 0

        def stop_motion_commands(self) -> None:
            self.stop_motion_count += 1

    class FakeRetargetor:
        def __init__(self) -> None:
            self.stop_count = 0

        def stop(self) -> None:
            self.stop_count += 1

    class FakeSession:
        latest = None
        init_count = 0

        def __init__(self, **_kwargs) -> None:
            FakeSession.init_count += 1
            self.arm = FakeArm()
            self.hand = object()
            self.retargetor = FakeRetargetor()
            self.end_count = 0
            FakeSession.latest = self

        def teleop(self, **_kwargs) -> str:
            return "exit"

        def end(self) -> None:
            self.end_count += 1

    fake_capture = ModuleType("paradex.dataset_acqusition.capture")
    fake_capture.CaptureSession = FakeSession
    monkeypatch.setitem(
        sys.modules, "paradex.dataset_acqusition.capture", fake_capture
    )
    visualization = SimpleNamespace(replay_selection=lambda: selection)

    result, session = grasp_vis._run_teleoperation(
        SimpleNamespace(device="vive", allegro_command_rate_hz=30.0),
        visualization,
    )

    assert result is selection
    assert session is FakeSession.latest
    assert session.arm.stop_motion_count == 1
    assert session.retargetor.stop_count == 1
    assert session.end_count == 0

    second_result, second_session = grasp_vis._run_teleoperation(
        SimpleNamespace(device="vive", allegro_command_rate_hz=30.0),
        visualization,
        session=session,
    )
    assert second_result is selection
    assert second_session is session
    assert FakeSession.init_count == 1
    assert session.arm.stop_motion_count == 2
    assert session.retargetor.stop_count == 2
    assert session.end_count == 0


def test_middle_pedal_can_confirm_robot_execution(monkeypatch) -> None:
    trajectory = replay_core.ReplayTrajectory(
        arm_poses=np.stack((_pose(0.0), _pose(0.1))),
        hand_actions=np.zeros((2, 16)),
        times=np.array((0.0, 1.0)),
        transition_frame_count=2,
        transition_seconds=1.0,
    )

    class FakePedalTrigger:
        def __init__(self) -> None:
            self.poll_count = 0

        def poll_pressed(self) -> bool:
            self.poll_count += 1
            return self.poll_count == 2

    calls = []
    pedal = FakePedalTrigger()
    monkeypatch.setattr(grasp_vis, "_poll_terminal_command", lambda: None)
    monkeypatch.setattr(grasp_vis.time, "sleep", lambda _: None)
    monkeypatch.setattr(replay_core, "_execute", lambda *args: calls.append(args))

    assert _execute_replay_if_confirmed(
        SimpleNamespace(execute=True, replay_preview=True),
        trajectory,
        pedal,
    ) is True
    assert pedal.poll_count == 2
    assert len(calls) == 1


def test_top_candidate_preview_uses_locked_episode_and_cached_frame(
    monkeypatch, tmp_path: Path
) -> None:
    first = _episode(1, 0.2)
    second = replace(
        _episode(2, 0.2),
        arm_poses=np.stack((_pose(0.2, 1.0), _pose(0.15, 1.0))),
    )
    selection = ReplaySelection(
        episode_id="2",
        frame_index=1,
        live_arm_pose=_pose(0.4, 1.0),
        live_hand_qpos=np.ones(16),
    )
    trajectory = replay_core.ReplayTrajectory(
        arm_poses=np.stack((_pose(0.4, 1.0), _pose(0.15, 1.0))),
        hand_actions=np.ones((2, 16)),
        times=np.array((0.0, 1.0)),
        transition_frame_count=2,
        transition_seconds=1.0,
    )
    preview_calls = []
    monkeypatch.setattr(
        replay_core,
        "_live_xarm_preview_state",
        lambda: (_pose(9.0), np.arange(6, dtype=np.float64)),
    )
    monkeypatch.setattr(
        replay_core,
        "_preview_replay",
        lambda *args, **kwargs: preview_calls.append((args, kwargs)),
    )

    _preview_top_candidate_replay(
        SimpleNamespace(),
        current_object_robot=np.eye(4),
        episodes=[first, second],
        candidate_frame_indices={"1": 1, "2": 1},
        selection=selection,
        trajectory=trajectory,
        output_dir=tmp_path,
    )

    positional, keywords = preview_calls[0]
    assert positional[1].episode.root.name == "2"
    assert positional[1].frame_index == 1
    np.testing.assert_allclose(keywords["live_arm_pose"], selection.live_arm_pose)
    np.testing.assert_allclose(keywords["live_arm_qpos"], np.arange(6))
    np.testing.assert_allclose(keywords["live_hand_qpos"], selection.live_hand_qpos)


def test_candidate_highlight_recreates_mesh_with_selected_color() -> None:
    class FakeHandle:
        def __init__(self) -> None:
            self.removed = False

        def remove(self) -> None:
            self.removed = True

    class FakeScene:
        def __init__(self) -> None:
            self.calls = []

        def add_mesh_simple(self, name, **kwargs):
            handle = FakeHandle()
            self.calls.append((name, kwargs, handle))
            return handle

    visualization = grasp_vis.LiveGraspIndexingVisualization.__new__(
        grasp_vis.LiveGraspIndexingVisualization
    )
    initial_handle = FakeHandle()
    scene = FakeScene()
    mesh = trimesh.creation.box()
    visualization.viewer = SimpleNamespace(server=SimpleNamespace(scene=scene))
    visualization.candidate_mesh_handles = {"7": initial_handle}
    visualization.candidate_mesh_specs = {"7": ("/candidate/7", mesh)}
    visualization.candidate_mesh_revisions = {"7": 0}

    visualization._set_candidate_selected("7", True)
    assert initial_handle.removed is True
    assert scene.calls[-1][0] == "/candidate/7/mesh_1"
    assert scene.calls[-1][1]["color"] == visualization._rgb255(
        grasp_vis.SELECTED_COLOR
    )

    selected_handle = visualization.candidate_mesh_handles["7"]
    visualization._set_candidate_selected("7", False)
    assert selected_handle.removed is True
    assert scene.calls[-1][0] == "/candidate/7/mesh_2"
    assert scene.calls[-1][1]["color"] == visualization._rgb255(
        grasp_vis.CANDIDATE_COLOR
    )
