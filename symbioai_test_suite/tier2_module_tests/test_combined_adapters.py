#!/usr/bin/env python3
"""
Phase 1 Critical Test: COMBINED Task Adapters (Test 11/12)

Tests COMBINED strategy's task-specific adaptation:
- LoRA-style adapter creation and management
- Parameter-efficient fine-tuning
- Adapter composition and reuse
- Memory efficiency

Competitive Advantage:
Efficient task-specific adaptation without full model retraining.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from comprehensive_benchmark import (
    create_continual_learning_system,
    Task
)


class TestCOMBINEDTaskAdapters:
    """Tests for COMBINED strategy adapter system."""
    
    def test_adapter_creation(self):
        """Test 11.1: Create task-specific adapters."""
        combined = create_continual_learning_system(strategy='combined', use_adapters=True)
        
        task = Task(id="t1", name="Task 1", num_classes=5)
        combined.register_task(task)
        
        # Create adapter
        adapter = combined.create_adapter(task.id)
        
        assert adapter is not None, "Adapter creation failed"
        assert task.id in combined.adapters, "Adapter not registered"
    
    def test_parameter_efficiency(self):
        """Test 11.2: Adapters are parameter-efficient."""
        combined = create_continual_learning_system(strategy='combined', use_adapters=True)
        
        # Get base model parameters
        base_params = sum(p.numel() for p in combined.model.parameters())
        
        # Create adapter
        task = Task(id="t1", name="Task 1", num_classes=5)
        adapter = combined.create_adapter(task.id)
        
        # Adapter should be much smaller than base model
        adapter_params = sum(p.numel() for p in adapter.parameters())
        
        assert adapter_params < base_params * 0.1, \
            f"Adapter too large: {adapter_params} vs base {base_params}"
        
        print(f"   Base params: {base_params:,}")
        print(f"   Adapter params: {adapter_params:,}")
        print(f"   Efficiency: {adapter_params/base_params*100:.2f}%")
    
    def test_adapter_composition(self):
        """Test 11.3: Compose adapters for multi-task scenarios."""
        combined = create_continual_learning_system(strategy='combined', use_adapters=True)
        
        # Create multiple task adapters
        tasks = [Task(id=f"t{i}", name=f"Task {i}", num_classes=5) for i in range(3)]
        adapters = []
        
        for task in tasks:
            adapter = combined.create_adapter(task.id)
            adapters.append(adapter)
        
        # Test adapter composition
        composed = combined.compose_adapters([t.id for t in tasks])
        
        assert composed is not None, "Adapter composition failed"
    
    def test_adapter_reuse(self):
        """Test 11.4: Reuse adapters across similar tasks."""
        combined = create_continual_learning_system(strategy='combined', use_adapters=True)
        
        # Similar tasks
        task1 = Task(id="vision_1", name="Vision A", num_classes=10)
        task2 = Task(id="vision_2", name="Vision B", num_classes=10)
        
        adapter1 = combined.create_adapter(task1.id)
        
        # Reuse adapter for similar task
        reused = combined.reuse_adapter(source_task=task1.id, target_task=task2.id)
        
        assert reused is not None, "Adapter reuse failed"
    
    def test_adapter_switching(self):
        """Test 11.5: Switch between adapters at inference."""
        combined = create_continual_learning_system(strategy='combined', use_adapters=True)
        
        # Create adapters for different tasks
        task1 = Task(id="t1", name="Task 1", num_classes=5)
        task2 = Task(id="t2", name="Task 2", num_classes=5)
        
        combined.create_adapter(task1.id)
        combined.create_adapter(task2.id)
        
        # Switch to task1 adapter
        combined.activate_adapter(task1.id)
        output1 = combined.model(torch.randn(1, combined.model.input_dim))
        
        # Switch to task2 adapter
        combined.activate_adapter(task2.id)
        output2 = combined.model(torch.randn(1, combined.model.input_dim))
        
        assert output1 is not None and output2 is not None, "Adapter switching failed"


def run_all_tests():
    """Run all COMBINED task adapter tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 11: COMBINED Task Adapters")
    print("=" * 80)
    
    test_suite = TestCOMBINEDTaskAdapters()
    results = {'total': 5, 'passed': 0, 'failed': 0, 'errors': []}
    
    tests = [
        ('Adapter Creation', test_suite.test_adapter_creation),
        ('Parameter Efficiency', test_suite.test_parameter_efficiency),
        ('Adapter Composition', test_suite.test_adapter_composition),
        ('Adapter Reuse', test_suite.test_adapter_reuse),
        ('Adapter Switching', test_suite.test_adapter_switching),
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
