from argparse import Namespace
import importlib.util
from pathlib import Path
import sys


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
        "replay_preview": None,
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
        "approach_min_seconds": 1.0,
        "dish_transfer_min_seconds": 1.0,
        "dish_clearance_m": 0.02,
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


def test_auto_execute_implies_execute():
    args = _args(auto_execute=True)
    capture_sil.validate_args(args)
    assert args.execute is True


def test_idle_loop_runs_one_round_per_c_and_quits_on_q(monkeypatch):
    commands = iter(("c", "c", "q"))
    rounds = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(capture_sil, "load_candidate_episodes", lambda *_args: [])
    monkeypatch.setattr(
        capture_sil,
        "run_round",
        lambda args, **_kwargs: rounds.append(args),
    )

    args = _args()
    capture_sil.run_replay_idle_loop(args)

    assert rounds == [args, args]


def test_idle_loop_recovers_from_a_failed_round(monkeypatch):
    commands = iter(("c", "c", "q"))
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
    commands = iter(("c", "c", "q"))
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
