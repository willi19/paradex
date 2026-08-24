"""Crash-tolerant JSONL and inference-boundary artifact logging."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Iterator

import cv2
import numpy as np

from paradex.inference.act_xarm_allegro.core import ObservationPacket


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class RunLogger:
    def __init__(self, root: Path, *, run_name: str | None = None) -> None:
        stamp = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_dir = Path(root).expanduser() / stamp
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._events = (self.run_dir / "telemetry.jsonl").open("a", buffering=1)
        self._lock = threading.Lock()
        self._chunk_index = 0

    def event(self, kind: str, **fields: Any) -> None:
        record = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **fields,
        }
        with self._lock:
            self._events.write(json.dumps(_json_value(record), separators=(",", ":")) + "\n")
            self._events.flush()

    def inference_boundary(self, packet: ObservationPacket, raw_actions: np.ndarray, inference_ms: float) -> Path:
        index = self._chunk_index
        self._chunk_index += 1
        boundary = self.run_dir / f"chunk_{index:06d}"
        boundary.mkdir()
        for key, image in packet.images.items():
            safe_key = key.replace(".", "_")
            jpeg = packet.jpeg_bytes.get(key)
            if jpeg is None:
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f"Failed to encode replay image {key}")
                jpeg = encoded.tobytes()
            (boundary / f"{safe_key}.jpg").write_bytes(jpeg)
        np.savez_compressed(
            boundary / "observation_action.npz",
            state=np.asarray(packet.state, dtype=np.float64),
            raw_actions=np.asarray(raw_actions, dtype=np.float64),
            captured_monotonic_ns=np.int64(packet.captured_monotonic_ns),
            state_monotonic_ns=np.int64(packet.state_monotonic_ns),
        )
        metadata = {
            "frame_ids": dict(packet.frame_ids),
            "image_keys": list(packet.images),
            "inference_ms": float(inference_ms),
        }
        (boundary / "metadata.json").write_text(json.dumps(metadata, indent=2))
        self.event("inference", chunk=index, artifact=str(boundary), **metadata)
        return boundary

    def close(self) -> None:
        self._events.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def iter_replay_packets(run_dir: Path) -> Iterator[ObservationPacket]:
    for boundary in sorted(Path(run_dir).glob("chunk_*")):
        arrays = np.load(boundary / "observation_action.npz")
        metadata = json.loads((boundary / "metadata.json").read_text())
        images: dict[str, np.ndarray] = {}
        jpeg_bytes: dict[str, bytes] = {}
        for key in metadata["image_keys"]:
            jpeg = (boundary / f"{key.replace('.', '_')}.jpg").read_bytes()
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Could not decode replay image {key}")
            if bgr.shape[:2] != (480, 640):
                bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
            images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            jpeg_bytes[key] = jpeg
        yield ObservationPacket(
            images=images,
            state=np.asarray(arrays["state"], dtype=np.float64),
            frame_ids={key: int(value) for key, value in metadata["frame_ids"].items()},
            captured_monotonic_ns=int(arrays["captured_monotonic_ns"]),
            state_monotonic_ns=int(arrays["state_monotonic_ns"]),
            jpeg_bytes=jpeg_bytes,
        )
