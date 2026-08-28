from argparse import Namespace
import importlib.util
from pathlib import Path
import signal
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "object6d" / "capture_sil.py"
SPEC = importlib.util.spec_from_file_location("capture_sil", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
capture_sil = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_sil)


def _args(**overrides):
    values = {
        "auto_execute": False,
        "execute": False,
        "replay": True,
        "retrieval_only": False,
        "naive_replay": False,
        "put_into_box_pink": False,
        "replay_preview": None,
        "mesh_name": "banana",
        "episode_root": Path("/dataset"),
        "candidate_episodes": (6, 7),
        "retrieval_translation_scale_m": 0.05,
        "retrieval_rotation_scale_rad": 0.25,
        "rate_scale": 1.0,
        "approach_linear_speed_mps": 0.13,
        "approach_angular_speed_rps": 0.8,
        "approach_rate_hz": 50.0,
        "dish_transfer_linear_speed_mps": 0.2,
        "dish_transfer_max_distance_m": 1.0,
        "dish_transfer_rate_hz": 50.0,
        "preview_max_frames": 150,
        "preview_ik_max_nfev": 50,
        "preview_position_scale": 0.05,
        "preview_rotation_scale": 0.5,
        "preview_object_max_faces": 3_000,
        "preview_robot_link_max_faces": 1_500,
        "approach_min_seconds": 1.0,
        "dish_transfer_min_seconds": 1.0,
        "dish_clearance_m": 0.02,
        "box_pink_clearance_m": 0.30,
        "hand_open_seconds": 1.0,
        "return_joint_speed_rps": 0.5,
        "return_min_seconds": 2.0,
        "return_rate_hz": 50.0,
        "eef_apex_annotations": None,
        "settle_seconds": 1.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_replay_preview_defaults_on_and_can_be_disabled():
    default_args = _args()
    capture_sil.validate_args(default_args)
    assert default_args.replay_preview is True

    disabled_args = _args(replay_preview=False)
    capture_sil.validate_args(disabled_args)
    assert disabled_args.replay_preview is False


def test_retrieval_defaults_weight_rotation_twice_as_much(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["capture_sil.py", "--name", "banana"])

    args = capture_sil.parse_args()

    assert args.retrieval_translation_scale_m == 0.05
    assert args.retrieval_rotation_scale_rad == 0.25


def test_replay_speed_scale_argument_controls_episode_rate(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_sil.py",
            "--name",
            "banana",
            "--replay",
            "--replay-speed-scale",
            "2.5",
        ],
    )

    args = capture_sil.parse_args()
    capture_sil.validate_args(args)

    assert args.rate_scale == pytest.approx(2.5)


def test_replay_round_skips_dish_and_uses_full_trajectory_path(tmp_path, monkeypatch):
    observed = {}

    def fake_capture(capture_root, _remote_path, _entry):
        capture_root.mkdir(parents=True)
        np.save(capture_root / "C2R.npy", np.eye(4))

    def fake_rpc(_address, request, _timeout):
        observed["request"] = request
        return {"object_6d": {"pose_world": np.eye(4).tolist()}}

    def fake_full_replay(_args, **kwargs):
        observed["replay"] = kwargs

    monkeypatch.setattr(capture_sil, "shared_dir", tmp_path)
    monkeypatch.setattr(capture_sil, "capture_once", fake_capture)
    monkeypatch.setattr(capture_sil, "send_rpc_once", fake_rpc)
    monkeypatch.setattr(
        capture_sil,
        "replay_closest_episode_naive",
        fake_full_replay,
    )
    args = _args(replay=True, naive_replay=False, mesh_name="banana")
    args.save_path = "captures"
    args.rcc_entry = "image_main.py"
    args.rpc_addr = "tcp://test"
    args.rpc_timeout_ms = 1000
    args.mesh_root_dir = tmp_path / "meshes"
    args.no_vis = True
    episodes = [object()]

    capture_sil.run_round(args, replay_episodes=episodes)

    assert observed["request"]["wooden_object_triangulation"] is False
    assert observed["replay"]["episodes"] is episodes
    assert "dish_point_robot" not in observed["replay"]


def test_name_is_optional_only_for_interactive_replay(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["capture_sil.py", "--naive-replay"])
    interactive_args = capture_sil.parse_args()
    capture_sil.validate_args(interactive_args)
    assert interactive_args.mesh_name is None

    monkeypatch.setattr(sys, "argv", ["capture_sil.py"])
    one_shot_args = capture_sil.parse_args()
    with pytest.raises(ValueError, match="--name is required"):
        capture_sil.validate_args(one_shot_args)


def test_auto_execute_implies_execute():
    args = _args(auto_execute=True)
    capture_sil.validate_args(args)
    assert args.execute is True


def test_replay_and_retrieval_only_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        capture_sil.validate_args(_args(retrieval_only=True))


def test_naive_replay_uses_generic_object_dataset(tmp_path):
    object_root = tmp_path / "allegro_v5" / "hammer"
    (object_root / "2").mkdir(parents=True)
    args = _args(
        replay=False,
        naive_replay=True,
        mesh_name="hammer",
        capture_root=tmp_path,
        robot="allegro_v5",
        episode_root=None,
        candidate_episodes=None,
    )

    capture_sil.validate_args(args)
    capture_sil.configure_episode_selection(args)

    assert args.replay_preview is True
    assert args.episode_root == object_root
    assert args.candidate_episodes == (2,)


def test_put_into_box_pink_uses_generic_dataset_and_preview(tmp_path):
    object_root = tmp_path / "allegro_v5" / "apple"
    (object_root / "3").mkdir(parents=True)
    args = _args(
        replay=False,
        put_into_box_pink=True,
        mesh_name="apple",
        capture_root=tmp_path,
        robot="allegro_v5",
        episode_root=None,
        candidate_episodes=None,
    )

    capture_sil.validate_args(args)
    capture_sil.configure_episode_selection(args)

    assert args.replay_preview is True
    assert args.episode_root == object_root
    assert args.candidate_episodes == (3,)


def test_generic_retrieval_discovers_object_episodes(tmp_path):
    object_root = tmp_path / "allegro_v5" / "apple"
    for episode_id in (3, 4, 7):
        (object_root / str(episode_id)).mkdir(parents=True)

    args = _args(
        replay=False,
        retrieval_only=True,
        mesh_name="apple",
        capture_root=tmp_path,
        robot="allegro_v5",
        episode_root=None,
        candidate_episodes=None,
    )
    capture_sil.configure_episode_selection(args)

    assert args.episode_root == object_root
    assert args.candidate_episodes == (3, 4, 7)


def test_generic_retrieval_prefers_successful_grasp_labels(tmp_path):
    object_root = tmp_path / "allegro_v5" / "apple"
    for episode_id, success in ((3, False), (4, True), (7, True)):
        episode_root = object_root / str(episode_id)
        episode_root.mkdir(parents=True)
        (episode_root / "grasp_result.json").write_text(
            '{"grasp_success": ' + str(success).lower() + "}", encoding="utf-8"
        )

    assert capture_sil._discover_candidate_episode_ids(object_root) == (4, 7)


def test_idle_loop_runs_one_round_per_c_and_quits_on_q(monkeypatch):
    commands = iter(("c", "apple", "c", "hammer", "q"))
    rounds = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(capture_sil, "load_candidate_episodes", lambda *_args: [])
    monkeypatch.setattr(
        capture_sil,
        "run_round",
        lambda args, **_kwargs: rounds.append(args),
    )

    args = _args(replay=False, naive_replay=True)
    capture_sil.run_replay_idle_loop(args)

    assert [round_args.mesh_name for round_args in rounds] == ["apple", "hammer"]


def test_idle_loop_switches_objects_and_reuses_episode_cache(tmp_path, monkeypatch):
    for object_name, episode_id in (("apple", 3), ("hammer", 2)):
        (tmp_path / "allegro_v5" / object_name / str(episode_id)).mkdir(
            parents=True
        )
    commands = iter(("c", "apple", "c", "hammer", "c", "apple", "q"))
    preload_roots = []
    rounds = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))

    def preload(round_args):
        preload_roots.append(round_args.episode_root)
        return [round_args.mesh_name]

    monkeypatch.setattr(capture_sil, "_preload_replay_episodes", preload)
    monkeypatch.setattr(
        capture_sil,
        "run_round",
        lambda round_args, **kwargs: rounds.append(
            (round_args.mesh_name, kwargs["replay_episodes"])
        ),
    )
    args = _args(
        replay=False,
        naive_replay=True,
        mesh_name=None,
        capture_root=tmp_path,
        robot="allegro_v5",
        episode_root=None,
        candidate_episodes=None,
    )

    capture_sil.run_replay_idle_loop(args)

    assert preload_roots == [
        tmp_path / "allegro_v5" / "apple",
        tmp_path / "allegro_v5" / "hammer",
    ]
    assert rounds == [
        ("apple", ["apple"]),
        ("hammer", ["hammer"]),
        ("apple", ["apple"]),
    ]


def test_idle_loop_recovers_from_a_failed_round(monkeypatch):
    commands = iter(("c", "banana", "c", "banana", "q"))
    attempts = []

    def run_round(args, **_kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            raise RuntimeError("capture failed")

    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(capture_sil, "load_candidate_episodes", lambda *_args: [])
    monkeypatch.setattr(capture_sil, "run_round", run_round)

    args = _args()
    capture_sil.run_replay_idle_loop(args)

    assert attempts == [args, args]


def test_execute_mode_reuses_controllers_until_idle_exit(monkeypatch):
    commands = iter(("c", "banana", "c", "banana", "q"))
    calls = []

    class Controller:
        def __init__(self, name):
            self.name = name
            self.end_count = 0

        def end(self):
            self.end_count += 1

    arm = Controller("arm")
    hand = Controller("hand")
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    episodes = [object(), object()]
    monkeypatch.setattr(
        capture_sil,
        "load_candidate_episodes",
        lambda *_args: episodes,
    )

    import paradex.io.robot_controller as controllers

    monkeypatch.setattr(controllers, "get_arm", lambda *_args, **_kwargs: arm)
    monkeypatch.setattr(controllers, "get_hand", lambda *_args, **_kwargs: hand)
    monkeypatch.setattr(
        capture_sil,
        "run_round",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )

    args = _args(execute=True)
    capture_sil.run_replay_idle_loop(args)

    for _round_args, kwargs in calls:
        assert kwargs["replay_arm"] is arm
        assert kwargs["replay_hand"] is hand
        assert kwargs["replay_episodes"] is episodes
    assert hand.end_count == 1
    assert arm.end_count == 1


def test_execute_mode_keeps_ctrl_c_as_keyboard_interrupt_for_preview(monkeypatch):
    commands = iter(("c", "banana", "q"))
    observed_handlers = []

    class Controller:
        def end(self):
            pass

    def ros_sigint_handler(_signum, _frame):
        raise AssertionError("simulated rclpy handler must not receive preview Ctrl+C")

    def get_arm(*_args, **_kwargs):
        signal.signal(signal.SIGINT, ros_sigint_handler)
        return Controller()

    original_handler = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(capture_sil, "load_candidate_episodes", lambda *_args: [])

    import paradex.io.robot_controller as controllers

    monkeypatch.setattr(controllers, "get_arm", get_arm)
    monkeypatch.setattr(controllers, "get_hand", lambda *_args, **_kwargs: Controller())
    monkeypatch.setattr(
        capture_sil,
        "run_round",
        lambda *_args, **_kwargs: observed_handlers.append(
            signal.getsignal(signal.SIGINT)
        ),
    )

    capture_sil.run_replay_idle_loop(_args(execute=True))

    assert observed_handlers == [signal.default_int_handler]
    assert signal.getsignal(signal.SIGINT) == original_handler
