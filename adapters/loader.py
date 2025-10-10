"""Adapter loader that composes runtime adapters for Symbio AI."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from registry.adapter_registry import ADAPTER_REGISTRY

logger = logging.getLogger(__name__)


class AdapterNotFoundError(Exception):
    pass


class AdapterLoader:
    def resolve(self, adapter_id: Optional[str]) -> Dict[str, Any]:
        if not adapter_id:
            raise AdapterNotFoundError("No adapter id provided")
        adapter = ADAPTER_REGISTRY.get_adapter(adapter_id)
        if not adapter:
            raise AdapterNotFoundError(f"Adapter {adapter_id} missing")
        if not adapter.is_active:
            raise AdapterNotFoundError(f"Adapter {adapter_id} inactive")
        logger.debug("Resolved adapter %s -> %s", adapter_id, adapter)
        return {
            "config": adapter.config,
            "name": adapter.name,
            "version": adapter.version,
            "capabilities": adapter.capabilities,
        }


ADAPTER_LOADER = AdapterLoader()
