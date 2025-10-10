"""Adapter registry for managing pluggable capability adapters.

Supports registration, versioning, capability tagging, lineage metadata, and
soft-deletion. Designed to integrate with the tenants and model orchestrator.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set


@dataclass
class AdapterMetadata:
    adapter_id: str
    name: str
    version: str
    capabilities: Set[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    lineage: Optional[str] = None
    owner: Optional[str] = None
    config: Dict[str, str] = field(default_factory=dict)


class AdapterRegistry:
    def __init__(self):
        self._adapters: Dict[str, AdapterMetadata] = {}
        self._lock = threading.RLock()

    def register_adapter(self, metadata: AdapterMetadata) -> None:
        with self._lock:
            metadata.last_updated = datetime.utcnow()
            self._adapters[metadata.adapter_id] = metadata

    def deactivate_adapter(self, adapter_id: str) -> None:
        with self._lock:
            if adapter_id in self._adapters:
                self._adapters[adapter_id].is_active = False
                self._adapters[adapter_id].last_updated = datetime.utcnow()

    def list_adapters(self, *, include_inactive: bool = False) -> List[AdapterMetadata]:
        with self._lock:
            if include_inactive:
                return list(self._adapters.values())
            return [a for a in self._adapters.values() if a.is_active]

    def get_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        with self._lock:
            return self._adapters.get(adapter_id)

    def find_by_capability(self, capability: str) -> List[AdapterMetadata]:
        with self._lock:
            return [a for a in self._adapters.values() if capability in a.capabilities and a.is_active]


ADAPTER_REGISTRY = AdapterRegistry()
