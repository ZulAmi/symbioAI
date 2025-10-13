#!/usr/bin/env python3
"""
🌟 PHASE 1: CRITICAL TESTS - MASTER RUNNER 🌟

Runs all 15 critical tests for SymbioAI's unique differentiators vs SakanaAI.

Test Categories:
1. Neural-Symbolic Integration (Tests 1-3) - 15 tests total
2. Causal Discovery & Reasoning (Tests 4-6) - 15 tests total
3. Multi-Agent Coordination (Tests 7-9) - 15 tests total
4. COMBINED Strategy - FLAGSHIP (Tests 10-12) - 15 tests total
5. Demonstration & Embodied Learning (Tests 13-15) - 15 tests total

TOTAL: 75 comprehensive tests across 5 critical capability areas

Purpose: Validate SymbioAI's competitive advantages for Fukuoka university funding proposals
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Import all test modules
sys.path.insert(0, str(Path(__file__).parent.parent))
test_dir = Path(__file__).parent / "tier2_module_tests"
sys.path.insert(0, str(test_dir))

# Import test runners
from test_neural_symbolic_integration import run_all_tests as run_ns_integration
from test_neural_symbolic_reasoning import run_all_tests as run_ns_reasoning
from test_neural_symbolic_agents import run_all_tests as run_ns_agents
from test_causal_discovery import run_all_tests as run_causal_discovery
from test_counterfactual_reasoning import run_all_tests as run_counterfactual
from test_causal_self_diagnosis import run_all_tests as run_self_diagnosis
from test_multi_agent_coordination import run_all_tests as run_ma_coordination
from test_emergent_communication import run_all_tests as run_emergent_comm
from test_adversarial_multi_agent import run_all_tests as run_adversarial_ma
from test_combined_strategy_core import run_all_tests as run_combined_core
from test_combined_adapters import run_all_tests as run_combined_adapters
from test_combined_progressive import run_all_tests as run_combined_progressive
from test_demonstration_learning import run_all_tests as run_demo_learning
from test_embodied_learning import run_all_tests as run_embodied
from test_active_learning_curiosity import run_all_tests as run_active_learning


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def run_phase1_critical_tests():
    """Run all Phase 1 critical tests."""
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print_header("🌟 SYMBIOAI PHASE 1: CRITICAL TESTS 🌟", "=")
    print("Validating competitive advantages vs SakanaAI")
    print("For: Fukuoka University Funding Proposals")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define all test suites
    test_suites = [
        # Category 1: Neural-Symbolic Integration
        ("1. Neural-Symbolic Integration", run_ns_integration),
        ("2. Neural-Symbolic Reasoning", run_ns_reasoning),
        ("3. Neural-Symbolic Agents", run_ns_agents),
        
        # Category 2: Causal Discovery
        ("4. Causal Discovery", run_causal_discovery),
        ("5. Counterfactual Reasoning", run_counterfactual),
        ("6. Causal Self-Diagnosis", run_self_diagnosis),
        
        # Category 3: Multi-Agent Coordination
        ("7. Multi-Agent Coordination", run_ma_coordination),
        ("8. Emergent Communication", run_emergent_comm),
        ("9. Adversarial Multi-Agent", run_adversarial_ma),
        
        # Category 4: COMBINED Strategy (FLAGSHIP)
        ("10. COMBINED Strategy Core ⭐", run_combined_core),
        ("11. COMBINED Task Adapters ⭐", run_combined_adapters),
        ("12. COMBINED Progressive ⭐", run_combined_progressive),
        
        # Category 5: Demonstration & Embodied Learning
        ("13. Demonstration Learning", run_demo_learning),
        ("14. Embodied Learning", run_embodied),
        ("15. Active Learning & Curiosity", run_active_learning),
    ]
    
    # Aggregate results
    all_results = {
        'phase': 'Phase 1 - Critical Tests',
        'timestamp': timestamp,
        'test_suites': {},
        'summary': {
            'total_suites': len(test_suites),
            'total_tests': 0,
            'total_passed': 0,
            'total_failed': 0,
            'suites_passed': 0,
            'suites_failed': 0,
        }
    }
    
    # Run each test suite
    for suite_name, test_runner in test_suites:
        print_header(f"Running: {suite_name}", "-")
        
        try:
            results = test_runner()
            
            all_results['test_suites'][suite_name] = results
            all_results['summary']['total_tests'] += results['total']
            all_results['summary']['total_passed'] += results['passed']
            all_results['summary']['total_failed'] += results['failed']
            
            if results['failed'] == 0:
                all_results['summary']['suites_passed'] += 1
            else:
                all_results['summary']['suites_failed'] += 1
                
        except Exception as e:
            print(f"❌ TEST SUITE CRASHED: {suite_name}")
            print(f"   Error: {str(e)}")
            all_results['test_suites'][suite_name] = {
                'total': 5,
                'passed': 0,
                'failed': 5,
                'errors': [{'test': 'suite_crash', 'error': str(e)}]
            }
            all_results['summary']['total_tests'] += 5
            all_results['summary']['total_failed'] += 5
            all_results['summary']['suites_failed'] += 1
    
    # Calculate final statistics
    execution_time = time.time() - start_time
    all_results['execution_time'] = execution_time
    
    success_rate = (all_results['summary']['total_passed'] / 
                   all_results['summary']['total_tests'] * 100)
    
    # Print final summary
    print_header("🎯 PHASE 1 CRITICAL TESTS - FINAL SUMMARY 🎯", "=")
    
    print("📊 Overall Statistics:")
    print(f"   Total Test Suites: {all_results['summary']['total_suites']}")
    print(f"   Total Tests: {all_results['summary']['total_tests']}")
    print(f"   Tests Passed: {all_results['summary']['total_passed']}")
    print(f"   Tests Failed: {all_results['summary']['total_failed']}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Execution Time: {execution_time:.2f}s")
    
    print("\n📈 Category Breakdown:")
    categories = [
        ("Neural-Symbolic Integration", ["1.", "2.", "3."]),
        ("Causal Discovery & Reasoning", ["4.", "5.", "6."]),
        ("Multi-Agent Coordination", ["7.", "8.", "9."]),
        ("COMBINED Strategy (FLAGSHIP)", ["10.", "11.", "12."]),
        ("Demonstration & Learning", ["13.", "14.", "15."]),
    ]
    
    for category_name, suite_prefixes in categories:
        category_passed = sum(
            all_results['test_suites'][suite]['passed']
            for suite in all_results['test_suites']
            if any(suite.startswith(prefix) for prefix in suite_prefixes)
        )
        category_total = sum(
            all_results['test_suites'][suite]['total']
            for suite in all_results['test_suites']
            if any(suite.startswith(prefix) for prefix in suite_prefixes)
        )
        
        status = "✅" if category_passed == category_total else "⚠️"
        print(f"   {status} {category_name}: {category_passed}/{category_total}")
    
    # Save detailed results
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    
    json_path = report_dir / f"phase1_critical_tests_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed report saved: {json_path}")
    
    # Final verdict
    print_header("🏆 COMPETITIVE ADVANTAGE VALIDATION 🏆", "=")
    
    if success_rate >= 80:
        print("✅ PHASE 1 CRITICAL TESTS: PASSED")
        print("   SymbioAI's competitive advantages are VALIDATED!")
        print("   Ready for Fukuoka University funding proposals.")
        print("\n   Unique Differentiators vs SakanaAI:")
        print("   ✓ Neural-Symbolic Reasoning with explainable AI")
        print("   ✓ Causal Discovery with counterfactual reasoning")
        print("   ✓ True Multi-Agent Collaboration")
        print("   ✓ COMBINED Strategy - Adaptive Continual Learning")
        print("   ✓ Demonstration & Embodied Learning")
    elif success_rate >= 60:
        print("⚠️  PHASE 1 CRITICAL TESTS: PARTIAL SUCCESS")
        print(f"   Success Rate: {success_rate:.1f}% (>60% but <80%)")
        print("   Some features validated, but improvements needed.")
    else:
        print("❌ PHASE 1 CRITICAL TESTS: NEEDS WORK")
        print(f"   Success Rate: {success_rate:.1f}% (<60%)")
        print("   Critical features require debugging.")
    
    print(f"\n{'=' * 80}\n")
    
    return all_results


if __name__ == "__main__":
    try:
        results = run_phase1_critical_tests()
        exit_code = 0 if results['summary']['total_failed'] == 0 else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR IN TEST RUNNER:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
