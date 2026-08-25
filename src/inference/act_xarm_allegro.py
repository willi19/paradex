#!/usr/bin/env python
"""Run ACT inference using the package in this checkout."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paradex.inference.act_xarm_allegro.cli import main


if __name__ == "__main__":
    main()
