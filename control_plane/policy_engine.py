"""Policy engine for request-time enforcement.

Policies run before orchestration kicks in and can be chained. Policies can
inspect request, tenant metadata, adapters, and runtime context. Returns allow/
denied decisions with optional remediation metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from control_plane.tenancy import TenantSettings

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    remediation: Optional[Dict[str, Any]] = None

    @classmethod
    def allow(cls, reason: str = "") -> "PolicyDecision":
        return cls(allowed=True, reason=reason)

    @classmethod
    def deny(cls, reason: str, remediation: Optional[Dict[str, Any]] = None) -> "PolicyDecision":
        return cls(allowed=False, reason=reason, remediation=remediation)


@dataclass
class PolicyContext:
    tenant: TenantSettings
    request_payload: Dict[str, Any]
    adapter_id: Optional[str]
    route: str
    metadata: Dict[str, Any] = field(default_factory=dict)


PolicyCallable = Callable[[PolicyContext], PolicyDecision]


class PolicyEngine:
    """Chains policy callables and short-circuits on denial."""

    def __init__(self, policies: Optional[Iterable[PolicyCallable]] = None):
        self._policies: List[PolicyCallable] = list(policies or [])

    def register(self, policy: PolicyCallable) -> None:
        logger.debug("Registering policy %s", policy)
        self._policies.append(policy)

    def evaluate(self, context: PolicyContext) -> PolicyDecision:
        logger.debug("Evaluating %d policies for tenant %s", len(self._policies), context.tenant.tenant_id)
        for policy in self._policies:
            decision = policy(context)
            if not decision.allowed:
                logger.info("Policy %s denied request: %s", policy.__name__, decision.reason)
                return decision
        return PolicyDecision.allow("All policies passed")


def quota_guard_policy(context: PolicyContext) -> PolicyDecision:
    """Example policy that validates quota hints embedded in metadata."""

    quota_hint = context.metadata.get("quota_exceeded")
    if quota_hint:
        return PolicyDecision.deny("Quota exceeded", {"retry_after": quota_hint})
    return PolicyDecision.allow()


DEFAULT_POLICY_ENGINE = PolicyEngine(policies=[quota_guard_policy])
"""Singleton policy engine instance used for quick integration."""
