"""
Failure Monitoring and Self-Healing Data Builder

Production-grade monitoring of inference outcomes to detect failing cases,
compute uncertainty metrics (confidence, entropy), and prepare targeted
datasets for automatic model surgery (fine-tuning/LoRA adapters).

This module is framework-agnostic and integrates with Symbio AI components:
- models.inference_engine.InferenceRequest/InferenceResponse
- models.adaptive_fusion.FusionContext (optional)
- training.auto_surgery.AutoModelSurgery (downstream)
"""

from __future__ import annotations

import time
import math
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Deque
from collections import defaultdict, deque


def _shannon_entropy(probabilities: List[float]) -> float:
    """Compute Shannon entropy in nats for a probability distribution.
    Falls back gracefully if inputs are invalid.
    """
    if not probabilities:
        return 0.0
    eps = 1e-12
    total = sum(max(p, 0.0) for p in probabilities)
    if total <= eps:
        return 0.0
    ent = 0.0
    for p in probabilities:
        q = max(p, 0.0) / total
        if q > eps:
            ent -= q * math.log(q + eps)
    return float(ent)


@dataclass
class FailureEvent:
    """Represents a single failure or low-confidence inference outcome."""
    timestamp: float
    model_id: str
    request_id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    context: Dict[str, Any]
    reason: str
    metrics: Dict[str, float]
    ground_truth: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)


class FailureMonitor:
    """
    Monitors inference responses and identifies candidates for self-healing.

    Capabilities:
    - Track rolling windows of outcomes per model/domain/task.
    - Detect low-confidence, high-entropy, or evaluator-marked failures.
    - Cluster failing inputs by simple keys (domain, task_type, tag).
    - Produce curated fine-tuning datasets for Automatic Model Surgery.
    """

    def __init__(
        self,
        window_size: int = 5000,
        confidence_threshold: float = 0.6,
        entropy_threshold: float = 2.0,
        failure_ratio_trigger: float = 0.15,
        min_failures_trigger: int = 25,
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.failure_ratio_trigger = failure_ratio_trigger
        self.min_failures_trigger = min_failures_trigger
        self.logger = logging.getLogger(__name__)

        # Rolling windows
        self._events_by_model: Dict[str, Deque[FailureEvent]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self._counts_by_bucket: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record(
        self,
        *,
        model_id: str,
        request_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        is_failure: Optional[bool] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[FailureEvent]:
        """
        Record an inference outcome and determine if it constitutes a failure.

        Heuristics used when is_failure is None:
        - If outputs/metadata provide 'confidence', compare with threshold.
        - If metadata contains a probability distribution (e.g., 'probs', 'logits'),
          compute entropy and compare with threshold.
        - If ground_truth is provided, mark failure when mismatch is detected
          using a lightweight comparator.
        """
        confidence = self._extract_confidence(outputs, metadata)
        entropy = self._extract_entropy(outputs, metadata)

        gt_mismatch = False
        if ground_truth is not None:
            gt_mismatch = not self._matches_ground_truth(outputs, ground_truth)

        if is_failure is None:
            failure_heur = (
                (confidence is not None and confidence < self.confidence_threshold)
                or (entropy is not None and entropy > self.entropy_threshold)
                or gt_mismatch
            )
        else:
            failure_heur = is_failure

        # Build standardized context bucket
        ctx = context or {}
        task_type = str(ctx.get("task_type", "unknown"))
        domain = str(ctx.get("domain", "general"))
        bucket_key = f"{model_id}|{task_type}|{domain}"

        if failure_heur:
            event = FailureEvent(
                timestamp=time.time(),
                model_id=model_id,
                request_id=request_id,
                inputs=inputs,
                outputs=outputs,
                context=ctx,
                reason=self._failure_reason(confidence, entropy, gt_mismatch),
                metrics={
                    "confidence": confidence if confidence is not None else float("nan"),
                    "entropy": entropy if entropy is not None else float("nan"),
                },
                ground_truth=ground_truth,
                tags=tags or [],
            )

            # Record event
            self._events_by_model[model_id].append(event)
            self._counts_by_bucket[bucket_key]["failures"] += 1
            self._counts_by_bucket[bucket_key]["total"] += 1
            return event

        # Non-failure still contributes to denominator for ratio
        self._counts_by_bucket[bucket_key]["total"] += 1
        return None

    def get_recent_failures(self, model_id: str, limit: int = 100) -> List[FailureEvent]:
        """Return recent failure events for a model."""
        events = list(self._events_by_model.get(model_id, []))
        return events[-limit:]

    def should_trigger_finetune(self, model_id: str, task_type: str, domain: str) -> bool:
        """
        Decide if Automatic Model Surgery should be triggered based on
        recent failure ratio and absolute failure counts in a bucket.
        """
        bucket_key = f"{model_id}|{task_type}|{domain}"
        counts = self._counts_by_bucket.get(bucket_key, {})
        failures = counts.get("failures", 0)
        total = counts.get("total", 0)
        ratio = (failures / total) if total > 0 else 0.0
        return failures >= self.min_failures_trigger and ratio >= self.failure_ratio_trigger

    def build_training_dataset(
        self,
        model_id: str,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        max_items: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Create a curated training set from failures for targeted fine-tuning.
        Returns a list of dicts with fields: input, output_target, context, tags.
        """
        failures = self.get_recent_failures(model_id, limit=self.window_size)
        samples: List[Dict[str, Any]] = []
        for ev in reversed(failures):  # newest first
            if task_type and str(ev.context.get("task_type")) != task_type:
                continue
            if domain and str(ev.context.get("domain")) != domain:
                continue
            sample = {
                "input": ev.inputs,
                # Prefer ground truth when available; else heuristic target
                "output_target": ev.ground_truth if ev.ground_truth is not None else self._derive_target(ev),
                "context": ev.context,
                "tags": ev.tags,
                "failure_reason": ev.reason,
                "metrics": ev.metrics,
                "request_id": ev.request_id,
            }
            samples.append(sample)
            if len(samples) >= max_items:
                break
        return samples

    # ----------------------------
    # Internal helper methods
    # ----------------------------
    def _extract_confidence(self, outputs: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[float]:
        # Common patterns across engines
        if isinstance(outputs.get("confidence"), (int, float)):
            return float(outputs["confidence"])
        if isinstance(metadata.get("confidence"), (int, float)):
            return float(metadata["confidence"])
        # Some engines place confidence in nested metadata
        for k in ("meta", "metrics", "logits_info"):
            v = metadata.get(k)
            if isinstance(v, dict) and isinstance(v.get("confidence"), (int, float)):
                return float(v["confidence"])
        return None

    def _extract_entropy(self, outputs: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[float]:
        # Try direct probabilities
        probs = outputs.get("probs") or metadata.get("probs")
        if isinstance(probs, list) and probs and isinstance(probs[0], (int, float)):
            return _shannon_entropy([float(p) for p in probs])

        # If logits provided, softmax then entropy
        logits = outputs.get("logits") or metadata.get("logits")
        if isinstance(logits, list) and logits and isinstance(logits[0], (int, float)):
            # Numerically stable softmax
            m = max(float(x) for x in logits)
            exps = [math.exp(float(x) - m) for x in logits]
            s = sum(exps)
            if s <= 0:
                return None
            probs_from_logits = [e / s for e in exps]
            return _shannon_entropy(probs_from_logits)
        return None

    def _matches_ground_truth(self, outputs: Dict[str, Any], gt: Dict[str, Any]) -> bool:
        # Generic lightweight comparator: exact match for key overlaps
        # and relaxed comparison for text outputs.
        common_keys = set(outputs.keys()) & set(gt.keys())
        for k in common_keys:
            ov, gv = outputs[k], gt[k]
            if isinstance(ov, str) and isinstance(gv, str):
                # simple normalization
                if ov.strip().lower() == gv.strip().lower():
                    return True
            elif ov == gv:
                return True
        # If no overlap or mismatch, treat as failure
        return False

    def _failure_reason(self, confidence: Optional[float], entropy: Optional[float], gt_mismatch: bool) -> str:
        reasons = []
        if confidence is not None and confidence < self.confidence_threshold:
            reasons.append("low_confidence")
        if entropy is not None and entropy > self.entropy_threshold:
            reasons.append("high_entropy")
        if gt_mismatch:
            reasons.append("ground_truth_mismatch")
        return ",".join(reasons) if reasons else "heuristic_failure"
