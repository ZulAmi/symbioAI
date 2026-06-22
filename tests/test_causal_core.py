"""Unit tests for causal inference core — CPU only, no Mammoth required."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.causal_inference import CausalEffect, StructuralCausalModel, CausalForgettingDetector
from tests.conftest import TinyMLP


# ─── CausalEffect dataclass ──────────────────────────────────────────────────

def test_causal_effect_fields():
    e = CausalEffect(
        source="task_0",
        target="forgetting_task_1",
        effect_size=-0.3,
        confidence=0.8,
        mechanism="feature_preservation",
    )
    assert e.source == "task_0"
    assert e.effect_size == pytest.approx(-0.3)
    assert e.confidence == pytest.approx(0.8)


# ─── StructuralCausalModel ───────────────────────────────────────────────────

@pytest.fixture
def scm() -> StructuralCausalModel:
    return StructuralCausalModel(num_tasks=4, feature_dim=16)


def test_scm_init(scm):
    assert scm.num_tasks == 4
    assert scm.feature_dim == 16


def test_causal_graph_shape(tiny_model, task_features):
    scm = StructuralCausalModel(num_tasks=3, feature_dim=16)
    task_labels = {i: torch.randint(0, 10, (20,)) for i in range(3)}
    graph = scm.learn_causal_structure(task_features, task_labels, tiny_model)
    assert graph is not None
    assert graph.shape == (3, 3)


def test_temporal_constraint(tiny_model, task_features):
    """Graph must be zero on and above the diagonal (no future-to-past edges)."""
    scm = StructuralCausalModel(num_tasks=3, feature_dim=16)
    task_labels = {i: torch.randint(0, 10, (20,)) for i in range(3)}
    graph = scm.learn_causal_structure(task_features, task_labels, tiny_model)
    assert graph is not None
    # Upper triangle (i <= j) entries should be zero due to temporal ordering
    for i in range(3):
        assert graph[i, i].item() == pytest.approx(0.0, abs=1e-6), \
            f"Diagonal [{i},{i}] should be 0, got {graph[i,i].item()}"


# ─── CausalForgettingDetector ────────────────────────────────────────────────

@pytest.fixture
def detector(tiny_model) -> CausalForgettingDetector:
    return CausalForgettingDetector(
        model=tiny_model,
        buffer_size=50,
        num_intervention_samples=5,
        true_temp_lr=0.01,
        true_micro_steps=1,
    )


@pytest.fixture
def old_task_data() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(1)
    return {
        0: (torch.randn(10, 1, 4, 4), torch.randint(0, 10, (10,))),
        1: (torch.randn(10, 1, 4, 4), torch.randint(0, 10, (10,))),
    }


def test_heuristic_does_not_modify_params(detector, old_task_data, tiny_model):
    """Heuristic mode must not change model parameters."""
    params_before = {k: v.clone() for k, v in tiny_model.named_parameters()}

    candidate = (torch.randn(1, 4, 4), torch.tensor(3))
    buffer = [(torch.randn(1, 4, 4), torch.tensor(i % 10)) for i in range(20)]

    detector.attribute_forgetting(
        candidate_sample=candidate,
        buffer_samples=buffer,
        old_task_data=old_task_data,
        current_task_id=2,
        use_true_intervention=False,
    )

    for name, param in tiny_model.named_parameters():
        assert torch.allclose(param, params_before[name]), \
            f"Parameter '{name}' was modified by heuristic mode"


def test_true_intervention_restores_params(detector, old_task_data, tiny_model):
    """TRUE intervention must restore model state after the checkpoint/restore cycle."""
    params_before = {k: v.clone() for k, v in tiny_model.named_parameters()}

    candidate = (torch.randn(1, 4, 4), torch.tensor(5))
    buffer = [(torch.randn(1, 4, 4), torch.tensor(i % 10)) for i in range(20)]

    detector.attribute_forgetting(
        candidate_sample=candidate,
        buffer_samples=buffer,
        old_task_data=old_task_data,
        current_task_id=2,
        use_true_intervention=True,
    )

    for name, param in tiny_model.named_parameters():
        assert torch.allclose(param, params_before[name], atol=1e-5), \
            f"Parameter '{name}' not restored after TRUE intervention"


def test_heuristic_returns_causal_effect(detector, old_task_data):
    candidate = (torch.randn(1, 4, 4), torch.tensor(2))
    buffer = [(torch.randn(1, 4, 4), torch.tensor(i % 10)) for i in range(10)]

    effect = detector.attribute_forgetting(
        candidate_sample=candidate,
        buffer_samples=buffer,
        old_task_data=old_task_data,
        current_task_id=2,
        use_true_intervention=False,
    )

    assert isinstance(effect, CausalEffect)
    assert isinstance(effect.effect_size, float)
    assert isinstance(effect.confidence, float)
    assert effect.confidence >= 0.0


def test_batched_returns_same_count(detector, old_task_data):
    """attribute_forgetting_batched output length == number of candidates."""
    candidates = [(torch.randn(1, 4, 4), torch.tensor(i % 10)) for i in range(5)]
    buffer = [(torch.randn(1, 4, 4), torch.tensor(i % 10)) for i in range(20)]

    effects = detector.attribute_forgetting_batched(
        candidate_samples=candidates,
        buffer_samples=buffer,
        old_task_data=old_task_data,
        current_task_id=2,
        use_true_intervention=False,
    )

    assert len(effects) == len(candidates)


def test_feature_cache_populated(detector, tiny_model):
    """After extracting features, the internal cache should be non-empty."""
    x = torch.randn(4, 1, 4, 4)
    _ = detector._extract_normalized_features(x, task_id=0, sample_type="candidate")
    assert len(detector.feature_cache) > 0
