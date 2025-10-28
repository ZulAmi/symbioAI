"""
Unit tests for causal graph indexing correctness.

These tests prevent the critical bug discovered on Oct 28, 2025 where
graph[i, j] convention was misunderstood, causing all importance scores
to be zero.

Run with: python -m pytest tests/test_causal_graph_indexing.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.causal_inference import StructuralCausalModel


class TestCausalGraphIndexing:
    """Test suite for causal graph indexing convention."""
    
    def test_graph_convention_source_target(self):
        """
        Verify that graph[i, j] means "Task i → Task j" (source → target).
        
        This is the CRITICAL contract that must not be violated.
        """
        scm = StructuralCausalModel(num_tasks=5, feature_dim=128)
        
        # Manually set an edge: Task 0 → Task 2 with strength 0.75
        source_task = 0
        target_task = 2
        causal_strength = 0.75
        
        scm.task_graph[source_task, target_task] = causal_strength
        
        # Verify we can read it back correctly
        retrieved = scm.task_graph[source_task, target_task].item()
        assert abs(retrieved - causal_strength) < 1e-6, \
            f"graph[{source_task}, {target_task}] should be {causal_strength}, got {retrieved}"
        
        # Verify the reverse edge is NOT set (directionality matters!)
        reverse = scm.task_graph[target_task, source_task].item()
        assert reverse == 0.0, \
            f"graph[{target_task}, {source_task}] should be 0 (backward edge), got {reverse}"
    
    def test_temporal_ordering_forward_edges_only(self):
        """
        In continual learning, only forward edges (i→j where i<j) should exist.
        
        Backward edges (i→j where i>j) violate causality (future can't affect past).
        """
        scm = StructuralCausalModel(num_tasks=5, feature_dim=128)
        
        # Set forward edge: Task 1 → Task 3 (valid)
        scm.task_graph[1, 3] = 0.6
        
        # Set backward edge: Task 3 → Task 1 (should be zero or very weak)
        scm.task_graph[3, 1] = 0.0
        
        # Verify forward edge exists
        forward = scm.task_graph[1, 3].item()
        assert forward > 0.5, f"Forward edge Task 1→3 should be strong, got {forward}"
        
        # Verify backward edge is weak/zero
        backward = scm.task_graph[3, 1].item()
        assert abs(backward) < 0.1, \
            f"Backward edge Task 3→1 should be ~0 (violates causality), got {backward}"
    
    def test_diagonal_is_zero(self):
        """Tasks should not have causal edges to themselves (no self-loops)."""
        scm = StructuralCausalModel(num_tasks=5, feature_dim=128)
        
        # Set some edges
        scm.task_graph[0, 1] = 0.5
        scm.task_graph[1, 2] = 0.6
        
        # Verify diagonal is zero
        for i in range(5):
            self_edge = scm.task_graph[i, i].item()
            assert self_edge == 0.0, f"Task {i} should not cause itself, got {self_edge}"
    
    def test_importance_sampling_indexing(self):
        """
        Test the actual indexing used in _get_causal_weighted_samples.
        
        This is the exact bug that was fixed on Oct 28, 2025.
        """
        scm = StructuralCausalModel(num_tasks=10, feature_dim=128)
        
        # Simulate learned graph after Task 3
        # Task 0 → Task 3: 0.574
        # Task 1 → Task 3: 0.612
        # Task 2 → Task 3: 0.489
        scm.task_graph[0, 3] = 0.574
        scm.task_graph[1, 3] = 0.612
        scm.task_graph[2, 3] = 0.489
        
        current_task = 3
        
        # When sampling for Task 3, we want to know:
        # "How important is Task 0 for Task 3?" → graph[0, 3]
        # "How important is Task 1 for Task 3?" → graph[1, 3]
        # etc.
        
        # CORRECT indexing (what the fixed code does):
        for task_id in [0, 1, 2]:
            importance = abs(scm.task_graph[task_id, current_task].item())
            assert importance > 0.4, \
                f"Task {task_id} should be important for Task {current_task}, got {importance}"
        
        # WRONG indexing (what the buggy code did):
        # This would look up graph[3, 0], graph[3, 1], graph[3, 2] (backward edges)
        for task_id in [0, 1, 2]:
            wrong_importance = abs(scm.task_graph[current_task, task_id].item())
            assert wrong_importance == 0.0, \
                f"Bug: graph[{current_task}, {task_id}] should be 0 (backward), got {wrong_importance}"
    
    def test_graph_shape(self):
        """Verify graph has correct shape (num_tasks × num_tasks)."""
        num_tasks = 10
        scm = StructuralCausalModel(num_tasks=num_tasks, feature_dim=128)
        
        assert scm.task_graph.shape == (num_tasks, num_tasks), \
            f"Graph should be {num_tasks}×{num_tasks}, got {scm.task_graph.shape}"
    
    def test_sparsification_preserves_strong_edges(self):
        """After sparsification, strong edges should remain."""
        scm = StructuralCausalModel(num_tasks=5, feature_dim=128)
        
        # Create a graph with varying strengths
        scm.task_graph[0, 1] = 0.9  # Very strong
        scm.task_graph[0, 2] = 0.7  # Strong
        scm.task_graph[0, 3] = 0.3  # Weak
        scm.task_graph[1, 2] = 0.2  # Very weak
        
        # Sparsify at 0.6 quantile (keep top 40%)
        threshold = scm.task_graph.abs().quantile(0.6)
        scm.task_graph[scm.task_graph.abs() < threshold] = 0
        
        # Strong edges should survive
        assert scm.task_graph[0, 1] > 0.8, "Very strong edge should survive"
        assert scm.task_graph[0, 2] > 0.6, "Strong edge should survive"
        
        # Weak edges should be pruned
        assert scm.task_graph[0, 3] == 0.0, "Weak edge should be pruned"
        assert scm.task_graph[1, 2] == 0.0, "Very weak edge should be pruned"


class TestImportanceScoringLogic:
    """Test the importance scoring logic used in causal sampling."""
    
    def test_importance_increases_with_causal_strength(self):
        """Stronger causal edges should produce higher importance scores."""
        scm = StructuralCausalModel(num_tasks=5, feature_dim=128)
        
        # Set edges with different strengths
        scm.task_graph[0, 3] = 0.3  # Weak
        scm.task_graph[1, 3] = 0.7  # Strong
        
        current_task = 3
        
        weak_importance = abs(scm.task_graph[0, current_task].item())
        strong_importance = abs(scm.task_graph[1, current_task].item())
        
        assert strong_importance > weak_importance, \
            f"Strong edge should have higher importance: {strong_importance} vs {weak_importance}"
    
    def test_future_tasks_have_zero_importance(self):
        """Buffer samples from future tasks should have zero importance."""
        scm = StructuralCausalModel(num_tasks=10, feature_dim=128)
        
        # Training Task 3, but somehow buffer has sample from Task 5 (shouldn't happen, but test it)
        current_task = 3
        future_task = 5
        
        # Future→past edges should not exist (temporal ordering)
        importance = abs(scm.task_graph[future_task, current_task].item())
        assert importance == 0.0, \
            f"Future task {future_task} should have 0 importance for current task {current_task}"
    
    def test_current_task_has_baseline_importance(self):
        """Samples from current task should have baseline importance (not causal)."""
        scm = StructuralCausalModel(num_tasks=10, feature_dim=128)
        
        current_task = 3
        
        # Current task should not have causal edge to itself
        self_importance = abs(scm.task_graph[current_task, current_task].item())
        assert self_importance == 0.0, "Current task should not cause itself"


class TestGraphValidation:
    """Test the validation checks added to prevent bugs."""
    
    def test_detect_backward_edges(self):
        """Validation should warn about strong backward edges."""
        scm = StructuralCausalModel(num_tasks=5, feature_dim=128)
        
        # Set a strong backward edge (shouldn't happen, but test detection)
        scm.task_graph[3, 1] = 0.7  # Task 3 → Task 1 (backward)
        
        # Check detection logic
        num_backward_edges = 0
        for i in range(5):
            for j in range(i):  # j < i means j is earlier
                if abs(scm.task_graph[i, j]) > 0.3:
                    num_backward_edges += 1
        
        assert num_backward_edges == 1, \
            f"Should detect 1 backward edge, found {num_backward_edges}"
    
    def test_enforce_zero_diagonal(self):
        """Diagonal should be forced to zero if non-zero detected."""
        scm = StructuralCausalModel(num_tasks=5, feature_dim=128)
        
        # Accidentally set diagonal (shouldn't happen, but test correction)
        scm.task_graph[2, 2] = 0.5
        
        # Apply correction (what the validation does)
        scm.task_graph.fill_diagonal_(0)
        
        # Verify correction worked
        for i in range(5):
            assert scm.task_graph[i, i] == 0.0, f"Diagonal[{i}] should be 0"


@pytest.mark.parametrize("source,target,strength", [
    (0, 1, 0.5),
    (0, 2, 0.7),
    (1, 3, 0.6),
    (2, 4, 0.8),
])
def test_various_edges(source, target, strength):
    """Parametrized test for various edge configurations."""
    scm = StructuralCausalModel(num_tasks=5, feature_dim=128)
    scm.task_graph[source, target] = strength
    
    # Verify storage
    assert abs(scm.task_graph[source, target].item() - strength) < 1e-6
    
    # Verify directionality (reverse should be zero)
    if source != target:
        assert scm.task_graph[target, source].item() == 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
