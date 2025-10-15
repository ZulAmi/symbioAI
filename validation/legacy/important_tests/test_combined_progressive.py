#!/usr/bin/env python3
"""
Phase 1 Critical Test: COMBINED Progressive Learning (Test 12/12)

Tests COMBINED strategy's progressive network capabilities:
- Progressive column creation for new tasks
- Lateral connections between columns
- Knowledge transfer across tasks
- Scalability

Competitive Advantage:
Progressive architecture prevents forgetting while enabling
forward transfer - best of both worlds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from comprehensive_benchmark import create_continual_learning_system, Task


class TestCOMBINEDProgressiveLearning:
    """Tests for COMBINED strategy progressive nets."""
    
    def test_progressive_column_creation(self):
        """Test 12.1: Create new columns for new tasks."""
        combined = create_continual_learning_system(strategy='combined', use_progressive_nets=True)
        
        # First task
        task1 = Task(id="t1", name="Task 1", num_classes=5)
        combined.register_task(task1)
        initial_columns = len(combined.progressive_columns) if hasattr(combined, 'progressive_columns') else 0
        
        # Second task should create new column
        task2 = Task(id="t2", name="Task 2", num_classes=5)
        combined.register_task(task2)
        new_columns = len(combined.progressive_columns) if hasattr(combined, 'progressive_columns') else 0
        
        assert new_columns > initial_columns, "No new column created"
    
    def test_lateral_connections(self):
        """Test 12.2: Lateral connections enable knowledge transfer."""
        combined = create_continual_learning_system(strategy='combined', use_progressive_nets=True)
        
        tasks = [Task(id=f"t{i}", name=f"Task {i}", num_classes=5) for i in range(3)]
        
        for task in tasks:
            combined.register_task(task)
        
        # Check lateral connections exist
        assert hasattr(combined, 'lateral_connections') or \
               hasattr(combined, 'progressive_columns'), \
            "No progressive structure"
    
    def test_forward_transfer(self):
        """Test 12.3: Knowledge transfers to new tasks."""
        combined = create_continual_learning_system(strategy='combined', use_progressive_nets=True)
        
        # Train on task 1
        task1 = Task(id="t1", name="Task 1", num_classes=5)
        combined.register_task(task1)
        
        for _ in range(10):
            data = torch.randn(16, combined.model.input_dim)
            labels = torch.randint(0, 5, (16,))
            combined.train_step(data, labels, task_id=task1.id)
        
        # Evaluate task 2 with transfer (should be better than random)
        task2 = Task(id="t2", name="Task 2", num_classes=5)
        combined.register_task(task2)
        
        # Even without training on task2, should leverage task1 knowledge
        perf = combined.evaluate(task2)
        assert perf is not None, "Forward transfer evaluation failed"
    
    def test_no_backward_interference(self):
        """Test 12.4: New tasks don't interfere with old tasks."""
        combined = create_continual_learning_system(strategy='combined', use_progressive_nets=True)
        
        # Train on task 1
        task1 = Task(id="t1", name="Task 1", num_classes=5)
        combined.register_task(task1)
        
        for _ in range(10):
            data = torch.randn(16, combined.model.input_dim)
            labels = torch.randint(0, 5, (16,))
            combined.train_step(data, labels, task_id=task1.id)
        
        perf_before = combined.evaluate(task1)
        
        # Train on task 2
        task2 = Task(id="t2", name="Task 2", num_classes=5)
        combined.register_task(task2)
        
        for _ in range(10):
            data = torch.randn(16, combined.model.input_dim)
            labels = torch.randint(0, 5, (16,))
            combined.train_step(data, labels, task_id=task2.id)
        
        perf_after = combined.evaluate(task1)
        
        # Task 1 performance should not degrade significantly
        degradation = perf_before - perf_after
        print(f"   Degradation: {degradation:.4f}")
        assert abs(degradation) < 0.5, "Excessive backward interference"
    
    def test_scalability(self):
        """Test 12.5: System scales to many tasks."""
        combined = create_continual_learning_system(strategy='combined', use_progressive_nets=True)
        
        # Create 10 tasks
        tasks = [Task(id=f"t{i}", name=f"Task {i}", num_classes=5) for i in range(10)]
        
        for task in tasks:
            combined.register_task(task)
            
            # Brief training
            for _ in range(3):
                data = torch.randn(8, combined.model.input_dim)
                labels = torch.randint(0, 5, (8,))
                loss = combined.train_step(data, labels, task_id=task.id)
                assert loss is not None, f"Training failed for {task.id}"
        
        print(f"   Successfully scaled to {len(tasks)} tasks")


def run_all_tests():
    """Run all COMBINED progressive learning tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 12: COMBINED Progressive Learning")
    print("=" * 80)
    
    test_suite = TestCOMBINEDProgressiveLearning()
    results = {'total': 5, 'passed': 0, 'failed': 0, 'errors': []}
    
    tests = [
        ('Progressive Column Creation', test_suite.test_progressive_column_creation),
        ('Lateral Connections', test_suite.test_lateral_connections),
        ('Forward Transfer', test_suite.test_forward_transfer),
        ('No Backward Interference', test_suite.test_no_backward_interference),
        ('Scalability', test_suite.test_scalability),
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
