"""Observability helpers bridging telemetry and dashboards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from control_plane.telemetry import TELEMETRY, MetricEvent, register_default_observers

_DASHBOARD_PATH = Path(__file__).parent / "dashboards" / "default.json"


class ObservabilityManager:
    def __init__(self) -> None:
        self._latest_metrics: Dict[str, float] = {}
        register_default_observers([self._update])

    def _update(self, event: MetricEvent) -> None:
        self._latest_metrics[event.name] = event.value

    def export_snapshot(self) -> Dict[str, float]:
        return dict(self._latest_metrics)

    def load_dashboard_config(self) -> Dict:
        if _DASHBOARD_PATH.exists():
            return json.loads(_DASHBOARD_PATH.read_text())
        return {}

    def emit_counter(self, name: str, value: float = 1.0, **attrs) -> None:
        TELEMETRY.record_counter(name, value, **attrs)

    def emit_gauge(self, name: str, value: float, **attrs) -> None:
        TELEMETRY.record_gauge(name, value, **attrs)


OBSERVABILITY = ObservabilityManager()
"""Singleton observability manager to be imported across modules."""
