#!/usr/bin/env python3
"""
Quick IRL Test - Test Symbio AI in 5 minutes
Runs the most essential demos to verify system works in real-world scenarios.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_quick_test(script: str, name: str):
    """Run a quick test and show results."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"Script: {script}")
    print('=' * 70)
    
    start = time.time()
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.absolute()
        
        # Set PYTHONPATH to include project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        result = subprocess.run(
            [sys.executable, script],
            timeout=120,
            capture_output=False,  # Show output directly
            env=env,  # Pass environment with PYTHONPATH
            cwd=str(project_root)  # Run from project root
        )
        duration = time.time() - start
        
        if result.returncode == 0:
            print(f"\n✅ {name} - SUCCESS ({duration:.1f}s)")
            return True
        else:
            print(f"\n❌ {name} - FAILED (exit code {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ {name} - TIMEOUT")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {name} - INTERRUPTED")
        raise
    except Exception as e:
        print(f"\n❌ {name} - ERROR: {e}")
        return False


def main():
    """Run quick IRL tests."""
    # Change to script directory (project root)
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              SYMBIO AI - QUICK IRL TEST (5 MINUTES)               ║
║         Verify core capabilities work in real scenarios           ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("examples/theorem_proving_demo.py", "Formal Verification (Safety Properties)"),
        ("examples/recursive_self_improvement_demo.py", "Meta-Evolution (Self-Improvement)"),
        ("examples/cross_task_transfer_demo.py", "Transfer Learning (Multi-Task)"),
    ]
    
    results = []
    start_time = time.time()
    
    print("\n🚀 Starting quick IRL tests...")
    print("⏱️  Expected time: ~5 minutes\n")
    
    for script, name in tests:
        try:
            success = run_quick_test(script, name)
            results.append((name, success))
        except KeyboardInterrupt:
            print("\n\n⚠️  Testing interrupted by user")
            break
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n{'=' * 70}")
    print("QUICK TEST SUMMARY")
    print('=' * 70)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nResults: {passed}/{total} passed ({total_time:.1f}s total)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System is working IRL!")
        print("\n📚 Next steps:")
        print("  • Run full test suite: python test_irl.py")
        print("  • Read testing guide: open HOW_TO_TEST_IRL.md")
        print("  • Try custom scenarios (see guide)")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\n🔧 Troubleshooting:")
        print("  • Check error messages above")
        print("  • Verify dependencies: pip list")
        print("  • See docs: open HOW_TO_TEST_IRL.md")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTesting cancelled")
        sys.exit(130)
