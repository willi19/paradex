"""The general message envelope shared by every capture_pc transport.

One wire format for *both* workloads:

  * **realtime execution** — camera frames, robot/teleop state (binary payloads,
    latency-sensitive), and
  * **data processing** — job status / progress (small structured metadata).

A message is a multipart frame::

    [ topic , header , buf0, buf1, ... ]

    topic   : bytes  — subscription key (SUB prefix-matches on it); e.g. b"cam".
    header  : msgpack of {seq, ts, src, meta, n} — see below.
    buf0..  : ``n`` raw binary payloads, sent/received zero-copy. May be empty
              (a pure-metadata message, e.g. a status update).

Header fields:
    seq  : monotonic per-(publisher) counter — lets a receiver detect drops.
    ts   : send time in nanoseconds (``time.time_ns()``) — lets a receiver
           measure end-to-end latency and order messages.
    src  : publisher name (which PC / which producer).
    meta : arbitrary msgpack-serializable payload (a dict, or a list of
           per-item dicts). This is where camera shapes / encodings / job status
           live. Buffers are referenced from meta by their index when needed.
    n    : number of binary buffers that follow the header.

msgpack is used for the header (compact + fast vs JSON); buffers stay raw so
there is never a copy or re-encode of image bytes.
"""

import time
from dataclasses import dataclass, field
from typing import Any, List

import msgpack


def _default(obj):
    """Fallback encoder so stray numpy scalars / arrays don't blow up a send.

    Keeps the transport robust for robot/teleop metadata (which often carries
    ``np.float64`` etc.) without forcing every caller to sanitise first.
    """
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:
        return obj.item()                 # 0-d numpy scalar -> python scalar
    if hasattr(obj, "tolist"):
        return obj.tolist()               # ndarray -> nested list
    raise TypeError(f"cannot msgpack {type(obj)!r} in message meta")


@dataclass
class Message:
    """A decoded envelope. ``bufs`` are raw ``bytes`` (e.g. JPEG / ``.npy``)."""
    topic: str
    seq: int
    ts_ns: int
    src: str
    meta: Any = None
    bufs: List[bytes] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        """End-to-end age of this message in milliseconds (recv - send)."""
        return (time.time_ns() - self.ts_ns) / 1e6


def encode(topic: str, meta: Any, bufs: List[bytes], seq: int, src: str) -> List[bytes]:
    """Build the multipart parts for one message. ``bufs`` may be empty."""
    bufs = bufs or []
    header = {
        "seq": int(seq),
        "ts": time.time_ns(),
        "src": src,
        "meta": meta,
        "n": len(bufs),
    }
    packed = msgpack.packb(header, use_bin_type=True, default=_default)
    parts = [topic.encode("utf-8") if isinstance(topic, str) else topic, packed]
    parts.extend(bufs)
    return parts


def decode(parts: List[bytes]) -> Message:
    """Parse multipart ``parts`` back into a :class:`Message`.

    Raises ``ValueError`` on a truncated / malformed frame so callers can count
    and skip it rather than crash the receive loop.
    """
    if len(parts) < 2:
        raise ValueError(f"short message: {len(parts)} frames")
    topic = parts[0].decode("utf-8", "replace")
    try:
        header = msgpack.unpackb(parts[1], raw=False)
    except Exception as e:
        raise ValueError(f"bad header: {e}")
    n = header.get("n", len(parts) - 2)
    bufs = parts[2:2 + n]
    return Message(
        topic=topic,
        seq=header.get("seq", -1),
        ts_ns=header.get("ts", 0),
        src=header.get("src", ""),
        meta=header.get("meta"),
        bufs=bufs,
    )
