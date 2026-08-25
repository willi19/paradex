"""Command line interface for the ACT xArm6 + Allegro runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from paradex.inference.act_xarm_allegro.core import CameraBinding, RunnerConfig, SafetyFilter
from paradex.inference.act_xarm_allegro.policy import load_safety_config
from paradex.inference.act_xarm_allegro.runner import HardwareRunner, run_contract, run_replay
from paradex.inference.act_xarm_allegro.telemetry import RunLogger


def _binding(value: str) -> CameraBinding:
    try:
        policy_key, physical = value.split("=", 1)
        serial, capture_pc = physical.split("@", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected POLICY_KEY=SERIAL@CAPTURE_PC") from exc
    return CameraBinding(policy_key, serial, capture_pc)


def _xyz(value: str) -> np.ndarray:
    try:
        parsed = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from exc
    if parsed.shape != (3,):
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety-gated ACT inference on xArm6 + Allegro v5")
    parser.add_argument("mode", choices=("contract", "replay", "shadow", "live"))
    parser.add_argument("--policy", default="hahahataeyun/hrdexdb-act-two-view-all-v1-act-100k")
    parser.add_argument("--policy-revision")
    parser.add_argument("--dataset", default="hahahataeyun/hrdexdb-act-two-view-all-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.01,
        help="Recency weighting for overlapping ACT predictions (0 = uniform).",
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-chunks-per-enable", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("~/shared_data/inference/act_xarm_allegro").expanduser())
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--camera", action="append", type=_binding, dest="cameras")
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--command-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--enable-live", action="store_true")
    parser.add_argument("--no-manage-capture-session", action="store_true")
    parser.add_argument("--workspace-lower", type=_xyz)
    parser.add_argument("--workspace-upper", type=_xyz)
    parser.add_argument("--max-linear-speed", type=float, default=0.70)
    parser.add_argument("--max-angular-speed", type=float, default=240.0)
    parser.add_argument("--max-hand-speed", type=float, default=4.0)
    parser.add_argument("--max-camera-age-ms", type=float, default=100.0)
    parser.add_argument("--max-state-age-ms", type=float, default=100.0)
    parser.add_argument("--max-consecutive-faults", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {}
    if args.cameras is not None:
        kwargs["camera_bindings"] = tuple(args.cameras)
    config = RunnerConfig(
        mode=args.mode,
        policy_path=args.policy,
        policy_revision=args.policy_revision,
        dataset_repo_id=args.dataset,
        device=args.device,
        control_hz=args.control_hz,
        action_steps=args.action_steps,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        duration_seconds=args.duration,
        max_chunks_per_enable=args.max_chunks_per_enable,
        output_dir=args.output_dir,
        state_endpoint=args.state_endpoint,
        command_endpoint=args.command_endpoint,
        enable_live=args.enable_live,
        **kwargs,
    )
    if args.mode == "contract":
        result = run_contract(config)
        print(json.dumps(asdict(result["contract"]) | {key: value for key, value in result.items() if key != "contract"}, default=str, indent=2))
        return
    with RunLogger(config.output_dir) as logger:
        logger.event("startup", config=config)
        if args.mode == "replay":
            if args.replay_dir is None:
                raise ValueError("replay mode requires --replay-dir")
            completed = run_replay(config, args.replay_dir, logger, max_chunks=args.max_chunks)
        else:
            safety_config = load_safety_config(
                config.dataset_repo_id,
                control_hz=config.control_hz,
                max_linear_speed_m_s=args.max_linear_speed,
                max_angular_speed_deg_s=args.max_angular_speed,
                max_hand_speed_rad_s=args.max_hand_speed,
                max_observation_age_ms=args.max_camera_age_ms,
                max_state_age_ms=args.max_state_age_ms,
                max_consecutive_faults=args.max_consecutive_faults,
                workspace_lower=args.workspace_lower,
                workspace_upper=args.workspace_upper,
            )
            logger.event("safety_config", config=safety_config)
            runner = HardwareRunner(
                config,
                SafetyFilter(safety_config),
                logger,
                max_chunks=args.max_chunks,
                manage_capture_session=not args.no_manage_capture_session,
            )
            completed = runner.run()
        logger.event("complete", chunks=completed)
        print(f"[act] completed {completed} chunks; telemetry: {logger.run_dir}")


if __name__ == "__main__":
    main()
