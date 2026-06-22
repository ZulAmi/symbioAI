"""Unit tests for ContinualLearningMetrics — no GPU or Mammoth required."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure the training package is importable without Mammoth by adding repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.metrics_tracker import ContinualLearningMetrics, estimate_ate_for_sample


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def metrics_2task() -> ContinualLearningMetrics:
    """Tracker seeded with a simple 2-task scenario."""
    m = ContinualLearningMetrics(num_tasks=2)
    # After task 0: acc on task 0 = 80%
    m.record_accuracy(task_id=0, trained_up_to=0, accuracy=80.0)
    # After task 1: acc on task 0 drops to 60%, task 1 = 75%
    m.record_accuracy(task_id=0, trained_up_to=1, accuracy=60.0)
    m.record_accuracy(task_id=1, trained_up_to=1, accuracy=75.0)
    m.update_current_task(1)
    return m


@pytest.fixture
def metrics_3task() -> ContinualLearningMetrics:
    """Tracker for 3 tasks with diagonal entries (right after training each task)."""
    m = ContinualLearningMetrics(num_tasks=3)
    # Diagonal: accuracy right after training each task
    m.record_accuracy(task_id=0, trained_up_to=0, accuracy=90.0)
    m.record_accuracy(task_id=1, trained_up_to=1, accuracy=85.0)
    m.record_accuracy(task_id=2, trained_up_to=2, accuracy=80.0)
    # After task 2: old tasks have dropped
    m.record_accuracy(task_id=0, trained_up_to=2, accuracy=70.0)
    m.record_accuracy(task_id=1, trained_up_to=2, accuracy=75.0)
    m.update_current_task(2)
    return m


# ─── Initialisation ──────────────────────────────────────────────────────────

def test_init_shapes():
    m = ContinualLearningMetrics(num_tasks=5)
    assert m.acc_matrix.shape == (5, 5)
    assert np.all(np.isnan(m.acc_matrix))


def test_init_empty_lists():
    m = ContinualLearningMetrics(num_tasks=3)
    assert m.task_accuracies == []
    assert m.initial_accuracies == []


# ─── record_accuracy ─────────────────────────────────────────────────────────

def test_record_accuracy_stored(metrics_2task):
    assert metrics_2task.acc_matrix[0, 0] == 80.0
    assert metrics_2task.acc_matrix[1, 0] == 60.0
    assert metrics_2task.acc_matrix[1, 1] == 75.0


# ─── compute_forgetting ──────────────────────────────────────────────────────

def test_forgetting_zero_for_single_task():
    m = ContinualLearningMetrics(num_tasks=3)
    m.record_accuracy(task_id=0, trained_up_to=0, accuracy=80.0)
    m.update_current_task(0)
    assert m.compute_forgetting(0) == 0.0


def test_forgetting_correct_value(metrics_2task):
    # Task 0: max=80, current=60 → forgetting=20
    f = metrics_2task.compute_forgetting(1)
    assert f == pytest.approx(20.0)


def test_forgetting_no_negative(metrics_2task):
    assert metrics_2task.compute_forgetting(1) >= 0.0


# ─── compute_backward_transfer ───────────────────────────────────────────────

def test_bwt_negative_when_acc_drops(metrics_2task):
    # acc[1,0] - acc[0,0] = 60 - 80 = -20
    bwt = metrics_2task.compute_backward_transfer(1)
    assert bwt == pytest.approx(-20.0)


def test_bwt_zero_for_single_task():
    m = ContinualLearningMetrics(num_tasks=2)
    m.record_accuracy(task_id=0, trained_up_to=0, accuracy=80.0)
    m.update_current_task(0)
    assert m.compute_backward_transfer(0) == 0.0


def test_bwt_3task(metrics_3task):
    # Tasks 0 and 1 dropped: (70-90) + (75-85) = -20 + -10 = -15, mean = -15
    bwt = metrics_3task.compute_backward_transfer(2)
    assert bwt == pytest.approx(-15.0)


# ─── compute_forward_transfer ────────────────────────────────────────────────

def test_fwt_zero_for_one_task():
    m = ContinualLearningMetrics(num_tasks=2)
    m.update_current_task(0)
    assert m.compute_forward_transfer(0) == 0.0


def test_fwt_uses_random_baseline():
    m = ContinualLearningMetrics(num_tasks=10)
    # Before training task 1, model scores 15% (better than 10% random)
    m.record_accuracy(task_id=1, trained_up_to=0, accuracy=15.0)
    m.update_current_task(1)
    # FWT = 15 - (100/10) = 5
    fwt = m.compute_forward_transfer(1)
    assert fwt == pytest.approx(5.0)


# ─── causal metrics ──────────────────────────────────────────────────────────

def test_causal_summary_empty():
    m = ContinualLearningMetrics(num_tasks=3)
    summary = m.get_causal_metrics_summary()
    assert summary["ate_mean"] == 0.0
    assert summary["num_ate_samples"] == 0
    assert summary["harmful_samples"] == 0
    assert summary["beneficial_samples"] == 0


def test_record_ate_accumulates():
    m = ContinualLearningMetrics(num_tasks=3)
    m.record_ate_score(0, 0.1)
    m.record_ate_score(0, 0.3)
    m.record_ate_score(1, -0.2)
    summary = m.get_causal_metrics_summary()
    assert summary["num_ate_samples"] == 3
    assert summary["ate_mean"] == pytest.approx((0.1 + 0.3 - 0.2) / 3)


def test_sample_attribution_classification():
    m = ContinualLearningMetrics(num_tasks=2)
    m.record_sample_attribution(0, ate=-0.1, threshold=0.05)  # harmful (ate < -threshold)
    m.record_sample_attribution(0, ate=0.1, threshold=0.05)   # beneficial (ate > threshold)
    m.record_sample_attribution(0, ate=0.01, threshold=0.05)  # neutral
    assert m.harmful_samples.get(0, 0) == 1
    assert m.beneficial_samples.get(0, 0) == 1


# ─── save_to_file / load round-trip ─────────────────────────────────────────

def test_save_load_roundtrip(metrics_2task):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    metrics_2task.save_to_file(path)

    with open(path) as f:
        data = json.load(f)

    assert data["num_tasks"] == 2
    # NaN serialises to null in JSON — check round-trip preserves structure
    matrix = data["acc_matrix"]
    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    # Known entries
    assert matrix[0][0] == pytest.approx(80.0)
    assert matrix[1][0] == pytest.approx(60.0)


# ─── get_summary ─────────────────────────────────────────────────────────────

def test_get_summary_keys(metrics_2task):
    s = metrics_2task.get_summary(1)
    for key in ("average_accuracy", "forgetting", "backward_transfer", "forward_transfer"):
        assert key in s


def test_average_accuracy(metrics_2task):
    # After task 1: row 1 = [60, 75] → mean = 67.5
    avg = metrics_2task.compute_average_accuracy(1)
    assert avg == pytest.approx(67.5)
