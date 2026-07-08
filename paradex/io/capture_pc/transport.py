"""Stable, high-throughput transport for the capture_pc :mod:`envelope` format.

Two producer/consumer pairs, one wire format, selectable delivery semantics:

    Publisher  --(realtime)-->  Collector      # PUB / SUB   — lossy-latest
    Publisher  --(lossless)-->  Collector      # PUSH / PULL — backpressured

Topology matches the rest of paradex: **the producer binds a port on each capture
PC, the main-PC collector connects to all of them**. That holds for both modes
(PULL can connect, PUSH can bind), so switching modes never changes wiring.

Delivery modes
--------------
``realtime`` (default) — PUB/SUB with a tiny high-water-mark. Under load the
    publisher *drops* rather than blocks, and the collector keeps only the newest
    item per name. Lowest latency, bounded memory; the right choice for previews,
    teleop state, and live status — anything where "newest wins".

``lossless`` — PUSH/PULL with a real queue. Under load the publisher *blocks*
    (backpressure) instead of dropping. Use for pipelines that must not lose a
    frame (e.g. recording every frame to a downstream consumer).

Every message carries ``seq``/``ts`` (see :mod:`envelope`), so the collector
reports **drops** (seq gaps) and **latency** for free — the observability the old
PUB/SUB pair lacked.
"""

import threading
import time
from typing import Any, Dict, List, Optional

import zmq

from paradex.utils.system import get_pc_list, get_pc_ip
from paradex.io.capture_pc.envelope import encode, decode, Message

REALTIME = "realtime"
LOSSLESS = "lossless"

# Small send/recv queues in realtime mode → drop-oldest instead of building a
# backlog. Big enough to ride out a scheduling hiccup, small enough to stay live.
_REALTIME_HWM = 4
_LOSSLESS_HWM = 1000


class Publisher:
    """Producer side. Binds ``port`` and publishes :mod:`envelope` messages.

    Args:
        port:  TCP port to bind.
        name:  producer id, stamped into every header as ``src`` (default: host
               pc name if resolvable, else ``publisher_<port>``).
        mode:  ``"realtime"`` (PUB, lossy-latest) or ``"lossless"`` (PUSH,
               backpressured).
    """

    def __init__(self, port: int = 1234, name: Optional[str] = None,
                 mode: str = REALTIME):
        if mode not in (REALTIME, LOSSLESS):
            raise ValueError(f"mode must be {REALTIME!r} or {LOSSLESS!r}")
        self.port = port
        self.name = name or f"publisher_{port}"
        self.mode = mode
        self._seq = 0
        self._lock = threading.Lock()

        self.context = zmq.Context.instance()
        sock_type = zmq.PUB if mode == REALTIME else zmq.PUSH
        self.socket = self.context.socket(sock_type)
        hwm = _REALTIME_HWM if mode == REALTIME else _LOSSLESS_HWM
        self.socket.setsockopt(zmq.SNDHWM, hwm)
        self.socket.setsockopt(zmq.LINGER, 0)
        if mode == LOSSLESS:
            # Block (backpressure) rather than raise when the queue is full.
            self.socket.setsockopt(zmq.SNDTIMEO, -1)
        self.socket.bind(f"tcp://*:{self.port}")

        # PUB drops messages to subscribers that connected "late"; give SUBs a
        # beat to finish connecting. (Lossless PUSH queues, so no wait needed.)
        if mode == REALTIME:
            time.sleep(0.1)

    def send(self, meta: Any = None, bufs: Optional[List[bytes]] = None,
             topic: str = "data") -> int:
        """Publish one message. Returns its ``seq``.

        Args:
            meta:  msgpack-serializable metadata (a dict, or a list of per-item
                   dicts). For multi-buffer messages, reference buffers by index.
            bufs:  list of raw binary payloads (JPEG bytes, ``.npy`` bytes, ...).
            topic: subscription key; collectors can filter on it.
        """
        with self._lock:
            seq = self._seq
            self._seq += 1
        parts = encode(topic, meta, bufs or [], seq, self.name)
        # copy=False keeps image buffers zero-copy; track=False since we don't
        # wait on completion. In realtime mode a full queue drops silently (PUB);
        # in lossless mode SNDTIMEO=-1 blocks until space frees.
        self.socket.send_multipart(parts, copy=False)
        return seq

    # Backward-compatible alias used by the DataPublisher shim.
    def send_data(self, metadata: Any, data: Optional[List[bytes]] = None,
                  topic: str = "data") -> int:
        return self.send(meta=metadata, bufs=data, topic=topic)

    def close(self) -> None:
        time.sleep(0.05)  # let the last frame flush
        self.socket.close()


class Collector:
    """Consumer side. Connects to every producer PC and keeps their latest items.

    A background thread drains all sockets; :meth:`get` returns the newest item
    per name (application-level conflation), so a caller always reads fresh state
    without managing queues. Sequence gaps are counted per source as ``drops``.

    Args:
        pc_list:  producer PCs to connect to (default: ``get_pc_list()``).
        port:     producer port (must match the :class:`Publisher`).
        mode:     ``"realtime"`` (SUB) or ``"lossless"`` (PULL). Match the producer.
        topics:   SUB topic prefixes to subscribe (realtime only; default: all).
    """

    def __init__(self, pc_list: Optional[List[str]] = None, port: int = 1234,
                 mode: str = REALTIME, topics: Optional[List[str]] = None):
        if mode not in (REALTIME, LOSSLESS):
            raise ValueError(f"mode must be {REALTIME!r} or {LOSSLESS!r}")
        self.pc_list = list(pc_list or get_pc_list())
        self.port = port
        self.mode = mode

        self.context = zmq.Context.instance()
        self.sockets: Dict[str, "zmq.Socket"] = {}
        self.poller = zmq.Poller()

        # name -> latest item dict (conflated). Also per-source liveness/drop stats.
        self.latest: Dict[str, dict] = {}
        self._last_seq: Dict[str, int] = {}
        self.stats: Dict[str, dict] = {}
        self._store_lock = threading.Lock()

        self.collecting = False
        self.thread: Optional[threading.Thread] = None

        sock_type = zmq.SUB if mode == REALTIME else zmq.PULL
        hwm = _REALTIME_HWM if mode == REALTIME else _LOSSLESS_HWM
        for pc_name in self.pc_list:
            ip = get_pc_ip(pc_name)
            sock = self.context.socket(sock_type)
            sock.setsockopt(zmq.RCVHWM, hwm)
            sock.setsockopt(zmq.LINGER, 0)
            if mode == REALTIME:
                for t in (topics or [""]):
                    sock.setsockopt_string(zmq.SUBSCRIBE, t)
            sock.connect(f"tcp://{ip}:{self.port}")
            self.sockets[pc_name] = sock
            self.poller.register(sock, zmq.POLLIN)
            self.stats[pc_name] = {"recv": 0, "drops": 0, "last_ts_ns": 0,
                                   "latency_ms": 0.0}

    # -- receive loop ------------------------------------------------------- #
    def _loop(self) -> None:
        while self.collecting:
            for sock, _ in self.poller.poll(timeout=100):
                pc_name = next(k for k, s in self.sockets.items() if s is sock)
                try:
                    while True:  # drain everything queued on this socket
                        parts = sock.recv_multipart(flags=zmq.NOBLOCK, copy=True)
                        self._ingest(pc_name, parts)
                except zmq.Again:
                    pass
                except Exception as e:
                    print(f"[Collector] {pc_name}: {e}")

    def _ingest(self, pc_name: str, parts: List[bytes]) -> None:
        try:
            msg = decode(parts)
        except ValueError as e:
            print(f"[Collector] {pc_name}: dropping malformed frame ({e})")
            return

        st = self.stats[pc_name]
        st["recv"] += 1
        st["last_ts_ns"] = msg.ts_ns
        st["latency_ms"] = msg.latency_ms
        # drop detection via monotonic seq per source
        prev = self._last_seq.get(msg.src)
        if prev is not None and msg.seq > prev + 1:
            st["drops"] += msg.seq - prev - 1
        self._last_seq[msg.src] = msg.seq

        # meta is either a list of per-item dicts (each with a 'name') or a
        # single dict. Store each item under its name, newest-wins.
        items = msg.meta if isinstance(msg.meta, list) else [msg.meta]
        with self._store_lock:
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                item["src"] = msg.src
                item["pc"] = pc_name
                item["seq"] = msg.seq
                item["ts_ns"] = msg.ts_ns
                # attach a buffer if this item indexes one
                di = item.pop("data_index", None)
                if di is None and len(items) == len(msg.bufs):
                    di = i
                if di is not None and 0 <= di < len(msg.bufs):
                    item["data"] = msg.bufs[di]
                name = item.get("name")
                if name is not None:
                    self.latest[name] = item

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        if self.collecting:
            return
        self.collecting = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.collecting = False
        if self.thread:
            self.thread.join(timeout=2)

    def end(self) -> None:
        self.stop()
        for sock in self.sockets.values():
            sock.close()

    # -- queries ------------------------------------------------------------ #
    def get(self, name: Optional[str] = None) -> Any:
        """Latest item by name, or a shallow copy of the whole latest map."""
        with self._store_lock:
            if name is not None:
                return self.latest.get(name)
            return dict(self.latest)

    def get_stats(self) -> Dict[str, dict]:
        """Per-source ``{recv, drops, latency_ms, last_ts_ns}`` — liveness/health."""
        return {k: dict(v) for k, v in self.stats.items()}
