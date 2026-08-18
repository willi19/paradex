"""Shared serialization and validation helpers for hand-alignment captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


class HandStateContractError(ValueError):
    """The input cannot be interpreted using the selected hand-state contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_manus_frame(frame: Mapping[str, np.ndarray]) -> dict[str, list[list[float]]]:
    """Convert named 4x4 MANUS transforms into portable JSON values."""
    serialized = {}
    for name, transform in frame.items():
        array = np.asarray(transform, dtype=np.float64)
        if array.shape != (4, 4) or not np.all(np.isfinite(array)):
            raise HandStateContractError(f"MANUS transform for {name!r} is not finite 4x4")
        serialized[str(name)] = array.tolist()
    if "wrist" not in serialized:
        raise HandStateContractError("MANUS frame must contain a wrist transform")
    return serialized


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)
