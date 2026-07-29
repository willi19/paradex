#!/usr/bin/env python3
"""Validate every grasp episode for the supported ECCV 2026 hand sources.

The source dataset is read-only.  A single JSON result is written atomically to
an explicitly selected path outside the mesh and capture roots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validate.grasp import (  # noqa: E402
    DatasetFormatError,
    ValidationThresholds,
    validate_dataset_episode,
)
from src.validate.grasp.episode_validation import (  # noqa: E402
    DEFAULT_CAPTURE_ROOT,
    DEFAULT_MESH_ROOT,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "grasp_validation_results.json"
SOURCE_TO_VALIDATOR_HAND = {
    "allegro_v5": "allegro",
    "human": "human",
    "inspire_dftp": "inspire",
    "inspire_f1": "inspire_f1",
}


@dataclass(frozen=True, order=True)
class BatchJob:
    source_hand: str
    object_name: str
    episode: int
    episode_root: Path

    @property
    def key(self) -> tuple[str, str, int]:
        return self.source_hand, self.object_name, self.episode

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_hand": self.source_hand,
            "validator_hand": SOURCE_TO_VALIDATOR_HAND[self.source_hand],
            "object": self.object_name,
            "episode": self.episode,
            "episode_root": str(self.episode_root),
        }


def discover_batch_jobs(
    capture_root: Path,
    source_hands: Iterable[str],
) -> list[BatchJob]:
    """Return every numeric episode directory under the selected sources."""

    jobs: list[BatchJob] = []
    for source_hand in source_hands:
        if source_hand not in SOURCE_TO_VALIDATOR_HAND:
            raise ValueError(f"unsupported source hand: {source_hand}")
        source_root = capture_root.expanduser().resolve() / source_hand
        if not source_root.is_dir():
            continue
        for object_root in sorted(source_root.iterdir(), key=lambda path: path.name):
            if not object_root.is_dir() or object_root.name.startswith("."):
                continue
            for episode_root in sorted(
                object_root.iterdir(), key=lambda path: path.name
            ):
                if not episode_root.is_dir() or not episode_root.name.isdigit():
                    continue
                jobs.append(
                    BatchJob(
                        source_hand=source_hand,
                        object_name=object_root.name,
                        episode=int(episode_root.name),
                        episode_root=episode_root.resolve(),
                    )
                )
    source_order = {
        source: index for index, source in enumerate(SOURCE_TO_VALIDATOR_HAND)
    }
    jobs.sort(
        key=lambda job: (
            source_order[job.source_hand],
            job.object_name,
            job.episode,
        )
    )
    return jobs


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_path(
    output: Path,
    *,
    mesh_root: Path,
    capture_root: Path,
) -> Path:
    """Reject any result path inside either read-only source-data tree."""

    resolved = output.expanduser().resolve()
    protected_roots = (
        mesh_root.expanduser().resolve(),
        capture_root.expanduser().resolve(),
    )
    for protected in protected_roots:
        if _path_is_within(resolved, protected):
            raise ValueError(
                f"output must not be inside source data root {protected}: {resolved}"
            )
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"output path is a directory: {resolved}")
    return resolved


def _atomic_write_json(
    output: Path,
    payload: dict[str, Any],
    *,
    pretty: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2 if pretty else None,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _run_job(
    job: BatchJob,
    thresholds_payload: dict[str, Any],
    mesh_root: str,
    capture_root: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = job.to_dict()
    try:
        report = validate_dataset_episode(
            job.object_name,
            job.episode,
            SOURCE_TO_VALIDATOR_HAND[job.source_hand],
            thresholds=ValidationThresholds(**thresholds_payload),
            mesh_root=Path(mesh_root),
            capture_root=Path(capture_root),
        )
    except (FileNotFoundError, DatasetFormatError, ValueError) as exc:
        return {
            **base,
            "status": "error",
            "valid": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    except Exception as exc:  # Keep one malformed episode from aborting the batch.
        return {
            **base,
            "status": "error",
            "valid": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        **base,
        "status": "valid" if report.valid else "invalid",
        "valid": report.valid,
        "elapsed_seconds": time.perf_counter() - started,
        "report": report.to_dict(),
    }


def _summary(
    results: list[dict[str, Any]],
    total_jobs: int,
    source_hands: Iterable[str],
) -> dict[str, Any]:
    def contact_loss_count(
        selected_results: Iterable[dict[str, Any]],
        kind: str,
    ) -> int:
        count = 0
        for result in selected_results:
            event = result.get("report", {}).get("contact_loss_event") or {}
            count += event.get("kind") == kind
        return count

    by_hand: dict[str, dict[str, int]] = {}
    for source in source_hands:
        source_results = [
            result for result in results if result["source_hand"] == source
        ]
        by_hand[source] = {
            "completed": len(source_results),
            "valid": sum(item["status"] == "valid" for item in source_results),
            "invalid": sum(item["status"] == "invalid" for item in source_results),
            "error": sum(item["status"] == "error" for item in source_results),
            "tracking_error": contact_loss_count(
                source_results, "tracking_error"
            ),
            "normal_post_loss_motion": contact_loss_count(
                source_results, "normal_motion"
            ),
            "insufficient_post_loss_observation": contact_loss_count(
                source_results, "insufficient_observation"
            ),
        }
    return {
        "total_discovered": total_jobs,
        "completed": len(results),
        "pending": total_jobs - len(results),
        "valid": sum(item["status"] == "valid" for item in results),
        "invalid": sum(item["status"] == "invalid" for item in results),
        "error": sum(item["status"] == "error" for item in results),
        "tracking_error": contact_loss_count(results, "tracking_error"),
        "normal_post_loss_motion": contact_loss_count(
            results, "normal_motion"
        ),
        "insufficient_post_loss_observation": contact_loss_count(
            results, "insufficient_observation"
        ),
        "by_hand": by_hand,
    }


def _build_payload(
    *,
    results: list[dict[str, Any]],
    total_jobs: int,
    source_hands: list[str],
    thresholds: ValidationThresholds,
    mesh_root: Path,
    capture_root: Path,
    started_at: str,
    complete: bool,
) -> dict[str, Any]:
    ordered = sorted(
        results,
        key=lambda item: (
            list(SOURCE_TO_VALIDATOR_HAND).index(item["source_hand"]),
            item["object"],
            item["episode"],
        ),
    )
    return {
        "schema_version": 6,
        "complete": complete,
        "started_at_utc": started_at,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_policy": "read-only",
        "assumptions": {
            "robot_self_collision": "absent_and_not_evaluated",
        },
        "excluded_validity_checks": [
            "OBJECT_SLIP_TRANSLATION",
            "OBJECT_SLIP_ROTATION",
        ],
        "mesh_root": str(mesh_root),
        "capture_root": str(capture_root),
        "source_hands": source_hands,
        "thresholds": asdict(thresholds),
        "summary": _summary(ordered, total_jobs, source_hands),
        "results": ordered,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all numeric object/episode directories for human, "
            "allegro_v5, inspire_dftp, and inspire_f1."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"single JSON result path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    parser.add_argument(
        "--hands",
        nargs="+",
        choices=tuple(SOURCE_TO_VALIDATOR_HAND),
        default=list(SOURCE_TO_VALIDATOR_HAND),
        help="source directories to scan; defaults to all four",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel worker processes; use 1 for the lowest memory usage",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="atomically refresh the same output file every N completed episodes",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="maximum uniformly sampled frames per episode",
    )
    parser.add_argument("--contact-distance", type=float, default=0.01)
    parser.add_argument("--min-contact-fingers", type=int, default=2)
    parser.add_argument("--min-grasp-frames", type=int, default=10)
    parser.add_argument("--min-object-motion", type=float, default=0.01)
    parser.add_argument("--min-contact-loss-samples", type=int, default=2)
    parser.add_argument(
        "--gravity-min-observation", type=float, default=0.1
    )
    parser.add_argument(
        "--gravity-min-displacement", type=float, default=0.005
    )
    parser.add_argument(
        "--gravity-min-velocity-change", type=float, default=0.05
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=DEFAULT_MESH_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=DEFAULT_CAPTURE_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.workers <= 0:
            raise ValueError("--workers must be positive")
        if args.checkpoint_every <= 0:
            raise ValueError("--checkpoint-every must be positive")
        output = validate_output_path(
            args.output,
            mesh_root=args.mesh_root,
            capture_root=args.capture_root,
        )
        if output.exists() and not args.overwrite:
            raise ValueError(
                f"output already exists; pass --overwrite to replace it: {output}"
            )
        thresholds = ValidationThresholds(
            max_frames=args.max_frames,
            contact_distance_m=args.contact_distance,
            min_contact_fingers=args.min_contact_fingers,
            min_grasp_frames=args.min_grasp_frames,
            min_contact_loss_samples=args.min_contact_loss_samples,
            min_object_motion_m=args.min_object_motion,
            min_gravity_observation_s=args.gravity_min_observation,
            min_gravity_displacement_m=args.gravity_min_displacement,
            min_gravity_velocity_change_m_s=(
                args.gravity_min_velocity_change
            ),
        )
        jobs = discover_batch_jobs(args.capture_root, args.hands)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[batch-grasp] configuration error: {exc}", file=sys.stderr)
        return 2

    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    print(
        f"[batch-grasp] discovered {len(jobs)} episodes; "
        f"hands={','.join(args.hands)} workers={args.workers}"
    )
    thresholds_payload = asdict(thresholds)
    try:
        if args.workers == 1:
            for index, job in enumerate(jobs, start=1):
                result = _run_job(
                    job,
                    thresholds_payload,
                    str(args.mesh_root),
                    str(args.capture_root),
                )
                results.append(result)
                print(
                    f"[{index}/{len(jobs)}] {job.source_hand}/"
                    f"{job.object_name}/{job.episode}: {result['status']}"
                )
                if index % args.checkpoint_every == 0:
                    _atomic_write_json(
                        output,
                        _build_payload(
                            results=results,
                            total_jobs=len(jobs),
                            source_hands=args.hands,
                            thresholds=thresholds,
                            mesh_root=args.mesh_root,
                            capture_root=args.capture_root,
                            started_at=started_at,
                            complete=False,
                        ),
                        pretty=args.pretty,
                    )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_job = {
                    executor.submit(
                        _run_job,
                        job,
                        thresholds_payload,
                        str(args.mesh_root),
                        str(args.capture_root),
                    ): job
                    for job in jobs
                }
                for index, future in enumerate(as_completed(future_to_job), start=1):
                    job = future_to_job[future]
                    result = future.result()
                    results.append(result)
                    print(
                        f"[{index}/{len(jobs)}] {job.source_hand}/"
                        f"{job.object_name}/{job.episode}: {result['status']}"
                    )
                    if index % args.checkpoint_every == 0:
                        _atomic_write_json(
                            output,
                            _build_payload(
                                results=results,
                                total_jobs=len(jobs),
                                source_hands=args.hands,
                                thresholds=thresholds,
                                mesh_root=args.mesh_root,
                                capture_root=args.capture_root,
                                started_at=started_at,
                                complete=False,
                            ),
                            pretty=args.pretty,
                        )
    except KeyboardInterrupt:
        _atomic_write_json(
            output,
            _build_payload(
                results=results,
                total_jobs=len(jobs),
                source_hands=args.hands,
                thresholds=thresholds,
                mesh_root=args.mesh_root,
                capture_root=args.capture_root,
                started_at=started_at,
                complete=False,
            ),
            pretty=args.pretty,
        )
        print(f"[batch-grasp] interrupted; partial result saved to {output}")
        return 130

    payload = _build_payload(
        results=results,
        total_jobs=len(jobs),
        source_hands=args.hands,
        thresholds=thresholds,
        mesh_root=args.mesh_root,
        capture_root=args.capture_root,
        started_at=started_at,
        complete=True,
    )
    _atomic_write_json(output, payload, pretty=args.pretty)
    print(f"[batch-grasp] complete: {payload['summary']}")
    print(f"[batch-grasp] result: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
