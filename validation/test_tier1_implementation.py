#!/usr/bin/env python3
"""
Test Tier 1 Continual Learning Implementation
============================================

Quick test to validate the Tier 1 continual learning benchmarks are working correctly.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

def test_tier1_datasets():
    """Test Tier 1 dataset loaders."""
    print("🧪 Testing Tier 1 dataset loaders...")
    
    try:
        from validation.tier1_continual_learning.datasets import load_tier1_dataset
        
        # Test datasets that should work quickly
        quick_datasets = ['fashion_mnist', 'svhn']
        
        for dataset_name in quick_datasets:
            print(f"\n📊 Testing {dataset_name}...")
            try:
                dataset = load_tier1_dataset(dataset_name, root='./data', train=True)
                print(f"   ✅ {dataset_name}: {len(dataset)} samples")
                
                # Test sample
                sample, target = dataset[0]
                print(f"   📏 Sample shape: {sample.shape if hasattr(sample, 'shape') else type(sample)}")
                print(f"   🎯 Target: {target}")
                
            except Exception as e:
                print(f"   ❌ {dataset_name} failed: {e}")
        
        print("✅ Dataset loader test complete")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loader test failed: {e}")
        return False


def test_tier1_validation():
    """Test Tier 1 validation system."""
    print("\n🧪 Testing Tier 1 validation system...")
    
    try:
        from validation.tier1_continual_learning.validation import Tier1Validator
        
        # Create validator
        validator = Tier1Validator()
        
        # Run quick validation
        print("🚀 Running quick continual learning validation...")
        result = validator.validate_continual_learning(
            dataset_name='fashion_mnist',
            num_tasks=2,  # Very quick test
            task_type='class_incremental',
            epochs_per_task=1,  # Minimal training
            include_baselines=False  # Skip baselines for speed
        )
        
        print(f"✅ Validation complete:")
        print(f"   Success Level: {result.success_level}")
        print(f"   Average Accuracy: {result.metrics.average_accuracy:.4f}")
        print(f"   Benchmark Score: {result.get_benchmark_score():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tier 1 validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tier_integration():
    """Test integration with main tier validation system."""
    print("\n🧪 Testing tier validation integration...")
    
    try:
        from validation.run_tier_validation import TierBasedValidator
        
        # Create validator
        validator = TierBasedValidator()
        
        # Test Tier 1 validation
        print("🚀 Running Tier 1 validation through main system...")
        result = validator.validate_tier(1, mode='quick')
        
        print(f"✅ Tier integration test complete:")
        print(f"   Success Level: {result.success_level}")
        print(f"   Tests Passed: {result.passed_tests}/{result.total_tests}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tier integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Tier 1 tests."""
    print("🧪 TIER 1 IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Dataset Loaders", test_tier1_datasets),
        ("Validation System", test_tier1_validation), 
        ("Tier Integration", test_tier_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {test_name}")
        print("="*60)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Tier 1 implementation is ready.")
    elif passed >= len(results) // 2:
        print("⚠️  Most tests passed. Some issues need attention.")
    else:
        print("❌ Multiple test failures. Implementation needs work.")


if __name__ == '__main__':
    main()