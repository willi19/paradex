#!/usr/bin/env python3
"""Validate one ECCV 2026 grasp episode and print a JSON report.

Examples:
    python src/validate/grasp/grasp_sequence.py apple 3 allegro
    python src/validate/grasp/grasp_sequence.py apple 0 human --pretty
    python src/validate/grasp/grasp_sequence.py apple 0 inspire --pretty

The command reads source data without modifying it and prints the JSON report
to stdout. Frame projection is currently excluded from validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validate.grasp import (
    DatasetFormatError,
    ValidationThresholds,
    validate_dataset_episode,
)
from src.validate.grasp.episode_validation import (
    DEFAULT_CAPTURE_ROOT,
    DEFAULT_MESH_ROOT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one grasp sequence selected by object, episode, and hand type."
        )
    )
    parser.add_argument("object", help="object directory/name, for example: apple")
    parser.add_argument("episode", type=int, help="non-negative episode number")
    parser.add_argument(
        "hand",
        choices=(
            "human",
            "allegro",
            "allegro_v5",
            "inspire",
            "inspire_dftp",
            "inspire_f1",
        ),
        help="hand/capture type (allegro_v5 and inspire_dftp are aliases)",
    )
    parser.add_argument(
        "--max-pregrasp-translation",
        type=float,
        default=0.05,
        metavar="METERS",
        help="maximum object translation before persistent grasp onset",
    )
    parser.add_argument(
        "--max-pregrasp-rotation",
        type=float,
        default=120.0,
        metavar="DEGREES",
        help="maximum object rotation before persistent grasp onset",
    )
    parser.add_argument(
        "--pregrasp-grace-frames",
        type=int,
        default=15,
        metavar="FRAMES",
        help="frames immediately before grasp onset excluded from this check",
    )
    parser.add_argument(
        "--max-object-position-jump",
        type=float,
        default=0.1,
        metavar="METERS",
        help="absolute frame-to-frame translation jump limit",
    )
    parser.add_argument(
        "--object-jump-local-factor",
        type=float,
        default=8.0,
        metavar="FACTOR",
        help="required jump ratio over neighboring frame displacement",
    )
    parser.add_argument(
        "--object-jump-window",
        type=int,
        default=5,
        metavar="FRAMES",
        help="neighboring displacement window on each side of a jump",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="indent the stdout JSON report"
    )
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
        thresholds = ValidationThresholds(
            max_pregrasp_translation_m=args.max_pregrasp_translation,
            max_pregrasp_rotation_deg=args.max_pregrasp_rotation,
            pregrasp_grace_frames=args.pregrasp_grace_frames,
            max_object_position_jump_m=args.max_object_position_jump,
            object_jump_local_factor=args.object_jump_local_factor,
            object_jump_window_frames=args.object_jump_window,
        )
        report = validate_dataset_episode(
            args.object,
            args.episode,
            args.hand,
            thresholds=thresholds,
            mesh_root=args.mesh_root,
            capture_root=args.capture_root,
        )
    except (FileNotFoundError, DatasetFormatError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "object": args.object,
                    "episode": args.episode,
                    "hand": args.hand,
                    "valid": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2 if args.pretty else None,
            )
        )
        return 2

    payload = {
        "object": args.object,
        "episode": args.episode,
        "hand": args.hand,
        "valid": report.valid,
        "report": report.to_dict(),
    }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=False,
        )
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
