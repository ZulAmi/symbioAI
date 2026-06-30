"""Shared fixtures for all tests. No Mammoth or CUDA required."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """Minimal CPU backbone — stands in for ResNet-18 during testing."""

    def __init__(self, in_dim: int = 16, out_dim: int = 10, feat_dim: int = 16) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.net = self  # some callers use model.net(x)
        self._features = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, feat_dim), nn.ReLU())
        self.classifier = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor, returnt: str | None = None) -> torch.Tensor:
        feats = self._features(x)
        if returnt == "features":
            return feats
        return self.classifier(feats)


@pytest.fixture
def tiny_model() -> TinyMLP:
    return TinyMLP()


@pytest.fixture
def task_features() -> dict[int, torch.Tensor]:
    """Three tasks, 20 samples each, 16-dim features."""
    torch.manual_seed(0)
    return {i: torch.randn(20, 16) for i in range(3)}


@pytest.fixture
def task_inputs_labels() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Three tasks of (4, 1, 4, 4) images with 10 classes."""
    torch.manual_seed(0)
    return {
        i: (torch.randn(20, 1, 4, 4), torch.randint(0, 10, (20,)))
        for i in range(3)
    }
