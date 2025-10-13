#!/usr/bin/env python3
"""
Phase 1 Critical Test: Active Learning & Curiosity (Test 15/15)

Tests active learning and curiosity-driven exploration:
- Uncertainty-based sample selection
- Curiosity-driven exploration
- Information gain maximization
- Efficient data acquisition

Competitive Advantage:
Autonomous learning with minimal supervision through intelligent
exploration and selective data collection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from training.active_learning_curiosity import (
    ActiveLearningCuriosity,
    create_active_learning_system
)


class TestActiveLearningCuriosity:
    """Tests for active learning and curiosity."""
    
    def test_active_learning_system_creation(self):
        """Test 15.1: Create active learning system."""
        system = create_active_learning_system(
            model_dim=128,
            num_classes=10,
            pool_size=1000
        )
        
        assert system is not None, "Active learning system creation failed"
        assert isinstance(system, ActiveLearningCuriosity), "Wrong type"
    
    def test_uncertainty_based_selection(self):
        """Test 15.2: Select most uncertain samples for labeling."""
        system = create_active_learning_system()
        
        # Unlabeled pool
        unlabeled_pool = torch.randn(100, 128)
        
        # Select most uncertain samples
        selected_indices = system.select_uncertain_samples(
            unlabeled_pool,
            num_samples=10
        )
        
        assert selected_indices is not None, "Uncertainty selection failed"
        assert len(selected_indices) == 10, "Wrong number of samples selected"
        assert all(0 <= idx < 100 for idx in selected_indices), "Invalid indices"
    
    def test_curiosity_driven_exploration(self):
        """Test 15.3: Explore based on curiosity/novelty."""
        system = create_active_learning_system()
        
        # Current experience
        seen_samples = torch.randn(50, 128)
        
        # New candidates
        candidates = torch.randn(20, 128)
        
        # Select novel/curious samples
        curious_indices = system.select_curious_samples(
            candidates,
            seen_samples,
            num_samples=5
        )
        
        assert curious_indices is not None, "Curiosity selection failed"
        assert len(curious_indices) == 5, "Wrong number selected"
    
    def test_information_gain_maximization(self):
        """Test 15.4: Maximize information gain from selected samples."""
        system = create_active_learning_system()
        
        # Unlabeled candidates
        candidates = torch.randn(50, 128)
        
        # Current model state
        current_performance = 0.7
        
        # Select samples maximizing information gain
        selected = system.maximize_information_gain(
            candidates,
            current_performance=current_performance,
            num_samples=8
        )
        
        assert selected is not None, "Information gain selection failed"
        assert len(selected) == 8, "Wrong selection count"
    
    def test_efficient_data_acquisition(self):
        """Test 15.5: Efficient learning with minimal labels."""
        system = create_active_learning_system()
        
        # Large unlabeled pool
        unlabeled = torch.randn(500, 128)
        
        # Budget: only 20 labels
        budget = 20
        
        # Iteratively select and "label"
        selected_count = 0
        for iteration in range(4):
            # Select 5 samples per iteration
            indices = system.select_uncertain_samples(unlabeled, num_samples=5)
            selected_count += len(indices)
            
            # Simulate labeling and training
            selected_data = unlabeled[indices]
            labels = torch.randint(0, 10, (len(indices),))
            
            # Update model
            system.update_with_labeled_data(selected_data, labels)
        
        assert selected_count == budget, "Budget not met"
        print(f"   Efficiently used {selected_count} labels from pool of {len(unlabeled)}")


def run_all_tests():
    """Run all active learning and curiosity tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 15: Active Learning & Curiosity")
    print("=" * 80)
    
    test_suite = TestActiveLearningCuriosity()
    results = {'total': 5, 'passed': 0, 'failed': 0, 'errors': []}
    
    tests = [
        ('Active Learning System', test_suite.test_active_learning_system_creation),
        ('Uncertainty Selection', test_suite.test_uncertainty_based_selection),
        ('Curiosity Exploration', test_suite.test_curiosity_driven_exploration),
        ('Information Gain', test_suite.test_information_gain_maximization),
        ('Efficient Data Acquisition', test_suite.test_efficient_data_acquisition),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}...")
            test_func()
            print(f"✅ PASSED: {test_name}")
            results['passed'] += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            results['failed'] += 1
            results['errors'].append({'test': test_name, 'error': str(e)})
    
    print(f"\n{'='*80}")
    print(f"Total: {results['total']}, Passed: {results['passed']}, Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed']/results['total']*100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
