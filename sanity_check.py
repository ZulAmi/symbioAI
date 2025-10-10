#!/usr/bin/env python3
"""
Sanity Check - Verify Symbio AI core systems are functional
Quick validation without running full demos (30 seconds)
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def check_imports():
    """Check that all core modules can be imported."""
    print("🔍 Checking core module imports...")
    
    modules_to_check = [
        ("training.automated_theorem_proving", "Formal Verification"),
        ("training.recursive_self_improvement", "Meta-Evolution"),
        ("training.cross_task_transfer", "Transfer Learning"),
        ("training.quantization_aware_evolution", "Model Compression"),
        ("training.compositional_concept_learning", "Concept Learning"),
        ("deployment.observability", "Observability System"),
        ("monitoring.production", "Production Monitoring"),
    ]
    
    results = []
    for module_name, display_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}: OK")
            results.append(True)
        except Exception as e:
            print(f"  ❌ {display_name}: FAILED - {e}")
            results.append(False)
    
    return all(results)


def check_dependencies():
    """Check critical dependencies."""
    print("\n🔍 Checking critical dependencies...")
    
    deps_to_check = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("z3", "Z3 Solver"),
        ("numpy", "NumPy"),
        ("jwt", "PyJWT"),
    ]
    
    results = []
    for module_name, display_name in deps_to_check:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}: Installed")
            results.append(True)
        except ImportError:
            print(f"  ❌ {display_name}: MISSING")
            results.append(False)
    
    return all(results)


def check_basic_functionality():
    """Test basic functionality of core systems."""
    print("\n🔍 Testing basic functionality...")
    
    # Test 1: Observability system
    try:
        from deployment.observability import OBSERVABILITY
        OBSERVABILITY.record_metric("test_metric", 42.0)
        OBSERVABILITY.increment_counter("test_counter")
        health = OBSERVABILITY.get_system_health()
        assert health['status'] in ('healthy', 'degraded')
        print("  ✅ Observability System: Working")
        test1 = True
    except Exception as e:
        print(f"  ❌ Observability System: FAILED - {e}")
        test1 = False
    
    # Test 2: Z3 Solver
    try:
        from z3 import Int, Solver
        x = Int('x')
        s = Solver()
        s.add(x > 0)
        s.add(x < 10)
        assert str(s.check()) == 'sat'
        print("  ✅ Z3 Solver: Working")
        test2 = True
    except Exception as e:
        print(f"  ❌ Z3 Solver: FAILED - {e}")
        test2 = False
    
    # Test 3: PyTorch
    try:
        import torch
        tensor = torch.randn(3, 3)
        assert tensor.shape == (3, 3)
        print("  ✅ PyTorch: Working")
        test3 = True
    except Exception as e:
        print(f"  ❌ PyTorch: FAILED - {e}")
        test3 = False
    
    # Test 4: Production Logger
    try:
        from monitoring.production import ProductionLogger
        logger = ProductionLogger("test_logger")
        assert logger.config is not None
        print("  ✅ Production Logger: Working")
        test4 = True
    except Exception as e:
        print(f"  ❌ Production Logger: FAILED - {e}")
        test4 = False
    
    return test1 and test2 and test3 and test4


def main():
    """Run sanity checks."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              SYMBIO AI - SANITY CHECK (30 SECONDS)                ║
║          Quick validation of core systems and dependencies        ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Working directory: {project_root}\n")
    
    # Run checks
    imports_ok = check_imports()
    deps_ok = check_dependencies()
    func_ok = check_basic_functionality()
    
    # Summary
    print("\n" + "=" * 70)
    print("SANITY CHECK SUMMARY")
    print("=" * 70)
    
    checks = [
        ("Core Module Imports", imports_ok),
        ("Critical Dependencies", deps_ok),
        ("Basic Functionality", func_ok),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        all_passed = all_passed and passed
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL SANITY CHECKS PASSED!")
        print("\n✅ System is healthy and ready for IRL testing")
        print("\n📚 Next steps:")
        print("  • Run quick test: python quick_test_irl.py")
        print("  • Run full suite: python test_irl.py")
        print("  • Read guide: open TEST_IRL_COMPLETE_GUIDE.md")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\n🔧 Troubleshooting:")
        print("  • Install dependencies: pip install -r requirements-core.txt")
        print("  • Install PyJWT: pip install PyJWT")
        print("  • Check Python version: python --version (need 3.11+)")
        print("  • See detailed guide: open TEST_IRL_COMPLETE_GUIDE.md")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSanity check interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
