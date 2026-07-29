#!/usr/bin/env python3
"""Validate one ECCV 2026 grasp episode and print a JSON report.

Examples:
    python src/validate/grasp/grasp_sequence.py apple 3 allegro
    python src/validate/grasp/grasp_sequence.py apple 0 human --pretty
    python src/validate/grasp/grasp_sequence.py apple 0 inspire --pretty

The command intentionally has no output-file option.  It reads the source
dataset and writes only to stdout.
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
        "--max-frames",
        type=int,
        default=120,
        help="maximum number of uniformly sampled frames per validation",
    )
    parser.add_argument(
        "--contact-distance",
        type=float,
        default=0.01,
        metavar="METERS",
        help="maximum fingertip-to-object distance counted as contact",
    )
    parser.add_argument(
        "--min-contact-fingers",
        type=int,
        default=2,
        help="minimum simultaneous fingertip contacts during grasp",
    )
    parser.add_argument(
        "--min-grasp-frames",
        type=int,
        default=10,
        help="minimum original-frame span of the inferred grasp phase",
    )
    parser.add_argument(
        "--min-object-motion",
        type=float,
        default=0.01,
        metavar="METERS",
        help="minimum object displacement during the grasp phase",
    )
    parser.add_argument(
        "--min-contact-loss-samples",
        type=int,
        default=2,
        help="consecutive sampled no-contact frames required to classify loss",
    )
    parser.add_argument(
        "--gravity-min-observation",
        type=float,
        default=0.1,
        metavar="SECONDS",
        help="minimum post-contact-loss duration needed for a gravity verdict",
    )
    parser.add_argument(
        "--gravity-min-displacement",
        type=float,
        default=0.005,
        metavar="METERS",
        help="minimum post-loss displacement along gravity accepted as falling",
    )
    parser.add_argument(
        "--gravity-min-velocity-change",
        type=float,
        default=0.05,
        metavar="M/S",
        help="minimum increase in gravity-axis velocity accepted as falling",
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
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
        report = validate_dataset_episode(
            args.object,
            args.episode,
            args.hand,
            thresholds=thresholds,
            mesh_root=args.mesh_root,
            capture_root=args.capture_root,
            robot_urdf=args.robot_urdf,
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
