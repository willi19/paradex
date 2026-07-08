"""Backward-compatible data-transport shims.

The real implementation now lives in :mod:`paradex.io.capture_pc.transport`
(:class:`Publisher` / :class:`Collector`) on the shared :mod:`paradex.io.capture_pc.envelope`
wire format — one stable, high-throughput format for both realtime frames and job
status, with sequence/latency/drop tracking built in.

``DataPublisher`` / ``DataCollector`` are kept as thin adapters so existing callers
(calibration clients, validate, object6d, camera tuning, paradex.process) keep
working unchanged. **New code should use** :class:`Publisher` / :class:`Collector`
directly — they expose delivery modes (realtime vs lossless), topics, and
``get_stats()``.

Wire/behaviour notes for the adapters (realtime PUB/SUB, newest-wins):
- ``send_data(metadata, data)`` publishes a list of per-item dicts plus raw
  buffers; an item is linked to a buffer by ``data_index`` (or positionally when
  item and buffer counts match).
- ``get_data(name)`` returns the newest item seen for that name (or the whole map).
"""

from typing import Any, Dict, List, Optional

from paradex.io.capture_pc.transport import Publisher, Collector, REALTIME, LOSSLESS

__all__ = ["DataPublisher", "DataCollector", "Publisher", "Collector",
           "REALTIME", "LOSSLESS"]


class DataPublisher(Publisher):
    """Deprecated alias for :class:`Publisher` (realtime mode).

    Kept for the existing ``dp.send_data(metadata, data)`` call sites.
    """

    def __init__(self, port: int = 1234, name: Optional[str] = None):
        super().__init__(port=port, name=name, mode=REALTIME)
        print(f"[{self.name}] Publisher started on port {self.port}")

    def send_data(self, metadata: List[Dict[str, Any]],
                  data: Optional[List[bytes]] = None) -> int:
        return self.send(meta=metadata, bufs=data, topic="data")

    def close(self) -> None:
        super().close()
        print(f"[{self.name}] Closed")


class DataCollector(Collector):
    """Deprecated alias for :class:`Collector` (realtime mode).

    Kept for the existing ``dc.get_data(...)`` call sites.
    """

    def __init__(self, pc_list: Optional[List[str]] = None, port: int = 1234):
        super().__init__(pc_list=pc_list, port=port, mode=REALTIME)
        for pc_name in self.pc_list:
            print(f"[Collector] Subscribed to {pc_name} at "
                  f"{self.port} ({self.stats[pc_name]['recv']} recv)")

    def get_data(self, name: Optional[str] = None) -> Any:
        """Latest item by name, or the whole latest map. See :meth:`Collector.get`."""
        return self.get(name)
