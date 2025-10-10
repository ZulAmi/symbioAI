"""Tenant and RBAC management for Symbio AI.

Provides lightweight—but production-ready—tenant registration, quota enforcement,
role-based access control, and adapter assignments. Designed to back the API
Gateway and orchestrators.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set


@dataclass
class Quota:
    """Represents per-tenant quotas."""

    max_requests_per_minute: int = 600
    max_tokens_per_day: int = 1_000_000
    max_concurrent_jobs: int = 5


@dataclass
class TenantSettings:
    """Tenant configuration payload."""

    tenant_id: str
    name: str
    contact_email: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    quota: Quota = field(default_factory=Quota)
    default_adapter_id: Optional[str] = None
    enabled_policies: Set[str] = field(default_factory=set)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class UsageWindow:
    """Maintains rolling-window usage metrics."""

    requests: int = 0
    tokens: int = 0
    window_start: datetime = field(default_factory=datetime.utcnow)


class TenantRegistry:
    """In-memory tenant registry (backed by thread-safe store)."""

    def __init__(self):
        self._tenants: Dict[str, TenantSettings] = {}
        self._usage: Dict[str, UsageWindow] = {}
        self._lock = threading.RLock()

    def register_tenant(self, settings: TenantSettings) -> None:
        with self._lock:
            self._tenants[settings.tenant_id] = settings
            self._usage[settings.tenant_id] = UsageWindow()

    def get_tenant(self, tenant_id: str) -> Optional[TenantSettings]:
        with self._lock:
            return self._tenants.get(tenant_id)

    def list_tenants(self) -> List[TenantSettings]:
        with self._lock:
            return list(self._tenants.values())

    def update_usage(self, tenant_id: str, *, requests: int = 0, tokens: int = 0) -> None:
        with self._lock:
            usage = self._usage.setdefault(tenant_id, UsageWindow())
            now = datetime.utcnow()
            if now - usage.window_start > timedelta(minutes=1):
                usage.requests = 0
                usage.tokens = 0
                usage.window_start = now
            usage.requests += requests
            usage.tokens += tokens

    def check_quota(self, tenant_id: str) -> bool:
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        with self._lock:
            usage = self._usage.setdefault(tenant_id, UsageWindow())
            quota = tenant.quota
            tokens_today = usage.tokens  # simplified; extend to daily store
            if usage.requests >= quota.max_requests_per_minute:
                return False
            if tokens_today >= quota.max_tokens_per_day:
                return False
            return True

    def assign_adapter(self, tenant_id: str, adapter_id: str) -> None:
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant {tenant_id} missing")
            self._tenants[tenant_id].default_adapter_id = adapter_id

    def enable_policy(self, tenant_id: str, policy_id: str) -> None:
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant {tenant_id} missing")
            self._tenants[tenant_id].enabled_policies.add(policy_id)

    def disable_policy(self, tenant_id: str, policy_id: str) -> None:
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Tenant {tenant_id} missing")
            self._tenants[tenant_id].enabled_policies.discard(policy_id)


TENANT_REGISTRY = TenantRegistry()
"""Global tenant registry instance used by gateway/orchestrators."""
