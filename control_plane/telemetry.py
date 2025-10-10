"""Telemetry and metrics aggregation helpers.

Implements an in-memory metrics collector that mimics OpenTelemetry concepts so
we can easily swap to real exporters later. Provides counters, gauges, and event
streams and supports observers (sinks) to feed dashboards.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class MetricEvent:
    name: str
    value: float
    timestamp: datetime
    attributes: Dict[str, Any]


Observer = Callable[[MetricEvent], None]


class TelemetryCollector:
    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._observers: List[Observer] = []
        self._lock = threading.RLock()

    def register_observer(self, observer: Observer) -> None:
        with self._lock:
            self._observers.append(observer)

    def record_counter(self, name: str, value: float = 1.0, **attributes: Any) -> None:
        with self._lock:
            self._counters[name] += value
            event = MetricEvent(name, self._counters[name], datetime.utcnow(), attributes)
            for observer in self._observers:
                observer(event)

    def record_gauge(self, name: str, value: float, **attributes: Any) -> None:
        with self._lock:
            self._gauges[name] = value
            event = MetricEvent(name, value, datetime.utcnow(), attributes)
            for observer in self._observers:
                observer(event)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }


TELEMETRY = TelemetryCollector()
"""Global telemetry collector instance."""


def log_observer(event: MetricEvent) -> None:
    print(f"[Telemetry] {event.timestamp.isoformat()} | {event.name}={event.value} | {event.attributes}")


def register_default_observers(observers: Optional[Iterable[Observer]] = None) -> None:
    observers = observers or [log_observer]
    for observer in observers:
        TELEMETRY.register_observer(observer)
