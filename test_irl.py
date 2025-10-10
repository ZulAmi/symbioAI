#!/usr/bin/env python3
"""
Real-World Testing Script - Run All IRL Tests
Tests Symbio AI in real-world scenarios without shell issues.
"""

import time
import subprocess
import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE} {text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✅ {text}{RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}❌ {text}{RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠️  {text}{RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"{BLUE}ℹ️  {text}{RESET}")


def run_demo(script_path: str, demo_name: str, timeout: int = 120) -> bool:
    """Run a demo script and return success status."""
    print_info(f"Running: {demo_name}")
    print_info(f"Script: {script_path}")
    print_info(f"Timeout: {timeout}s")
    
    try:
        # Get project root and set PYTHONPATH
        project_root = Path(__file__).parent.absolute()
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,  # Pass environment with PYTHONPATH
            cwd=str(project_root)  # Run from project root
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"{demo_name} completed in {duration:.2f}s")
            return True
        else:
            print_error(f"{demo_name} failed with exit code {result.returncode}")
            if result.stderr:
                print_error(f"Error output:\n{result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"{demo_name} timed out after {timeout}s")
        return False
    except Exception as e:
        print_error(f"{demo_name} encountered error: {e}")
        return False


def test_level_1_basic_demos():
    """Level 1: Run basic real-world demos."""
    print_header("LEVEL 1: Basic Real-World Demos (5-30 minutes)")
    
    demos = [
        ("examples/theorem_proving_demo.py", "Formal Verification & Safety Properties", 180),
        ("examples/recursive_self_improvement_demo.py", "Meta-Evolution & Self-Improvement", 300),
        ("examples/cross_task_transfer_demo.py", "Transfer Learning & Task Synthesis", 240),
        ("examples/quantization_evolution_demo.py", "Model Compression & Optimization", 200),
        ("examples/compositional_concept_demo.py", "Concept Learning & Reasoning", 180),
    ]
    
    results = []
    for script, name, timeout in demos:
        success = run_demo(script, name, timeout)
        results.append((name, success))
        print()  # Blank line
    
    return results


def test_level_2_advanced_demos():
    """Level 2: Advanced demos that may need dependencies."""
    print_header("LEVEL 2: Advanced Demos (may have dependencies)")
    
    demos = [
        ("examples/continual_learning_demo.py", "Continual Learning & Adaptation", 240),
        ("examples/unified_multimodal_demo.py", "Multi-Modal Processing", 200),
        ("examples/embodied_ai_demo.py", "Embodied AI & Robotics", 180),
    ]
    
    results = []
    for script, name, timeout in demos:
        # Check if file exists
        if not os.path.exists(script):
            print_warning(f"Skipping {name} - script not found")
            results.append((name, None))
            continue
            
        success = run_demo(script, name, timeout)
        results.append((name, success))
        print()
    
    return results


def test_individual_systems():
    """Test individual systems through quickstart."""
    print_header("Testing Individual Systems via Quickstart")
    
    systems = [
        "recursive_self_improvement",
        "transfer_learning",
        "theorem_proving",
        "quantization_evolution",
    ]
    
    results = []
    for system in systems:
        print_info(f"Testing system: {system}")
        try:
            result = subprocess.run(
                [sys.executable, "quickstart.py", "system", system],
                capture_output=True,
                text=True,
                timeout=120
            )
            success = result.returncode == 0
            if success:
                print_success(f"{system} works!")
            else:
                print_error(f"{system} failed")
            results.append((system, success))
        except Exception as e:
            print_error(f"{system} error: {e}")
            results.append((system, False))
    
    return results


def print_summary(level1_results, level2_results, system_results):
    """Print comprehensive summary."""
    print_header("TEST SUMMARY - Real-World Performance")
    
    # Level 1 Summary
    print(f"\n{BOLD}Level 1: Basic Demos{RESET}")
    level1_passed = sum(1 for _, s in level1_results if s)
    level1_total = len(level1_results)
    for name, success in level1_results:
        status = f"{GREEN}✅ PASS{RESET}" if success else f"{RED}❌ FAIL{RESET}"
        print(f"  {status} - {name}")
    print(f"  {BOLD}Result: {level1_passed}/{level1_total} passed{RESET}")
    
    # Level 2 Summary
    print(f"\n{BOLD}Level 2: Advanced Demos{RESET}")
    level2_passed = sum(1 for _, s in level2_results if s)
    level2_tested = sum(1 for _, s in level2_results if s is not None)
    for name, success in level2_results:
        if success is None:
            status = f"{YELLOW}⊘ SKIP{RESET}"
        elif success:
            status = f"{GREEN}✅ PASS{RESET}"
        else:
            status = f"{RED}❌ FAIL{RESET}"
        print(f"  {status} - {name}")
    print(f"  {BOLD}Result: {level2_passed}/{level2_tested} passed (of {len(level2_results)} total){RESET}")
    
    # System Tests Summary
    print(f"\n{BOLD}Individual Systems{RESET}")
    system_passed = sum(1 for _, s in system_results if s)
    system_total = len(system_results)
    for name, success in system_results:
        status = f"{GREEN}✅ PASS{RESET}" if success else f"{RED}❌ FAIL{RESET}"
        print(f"  {status} - {name}")
    print(f"  {BOLD}Result: {system_passed}/{system_total} passed{RESET}")
    
    # Overall Summary
    total_passed = level1_passed + level2_passed + system_passed
    total_tests = level1_total + level2_tested + system_total
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}OVERALL RESULTS:{RESET}")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_tests - total_passed}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print(f"\n{GREEN}{BOLD}🎉 EXCELLENT - System is production-ready!{RESET}")
    elif success_rate >= 60:
        print(f"\n{YELLOW}{BOLD}⚠️  GOOD - Some issues to address{RESET}")
    else:
        print(f"\n{RED}{BOLD}❌ NEEDS WORK - Multiple failures detected{RESET}")
    
    print(f"{BOLD}{'=' * 80}{RESET}\n")


def main():
    """Main testing orchestrator."""
    print(f"{BOLD}{BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           SYMBIO AI - REAL-WORLD TESTING SUITE                   ║")
    print("║              Comprehensive IRL Validation                         ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")
    
    print_info("This script tests Symbio AI with real-world scenarios")
    print_info("Tests are divided into levels based on complexity")
    print_info("Estimated time: 15-30 minutes for full suite\n")
    
    input("Press ENTER to start testing...")
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run test levels
    start_time = time.time()
    
    level1_results = test_level_1_basic_demos()
    level2_results = test_level_2_advanced_demos()
    system_results = test_individual_systems()
    
    total_time = time.time() - start_time
    
    # Print summary
    print_summary(level1_results, level2_results, system_results)
    
    print_info(f"Total testing time: {total_time / 60:.1f} minutes")
    print_info("Testing complete!")
    
    # Return exit code based on results
    total_passed = sum(1 for _, s in level1_results if s)
    total_passed += sum(1 for _, s in level2_results if s)
    total_passed += sum(1 for _, s in system_results if s)
    
    total_tests = len(level1_results)
    total_tests += sum(1 for _, s in level2_results if s is not None)
    total_tests += len(system_results)
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    if success_rate >= 80:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Testing interrupted by user{RESET}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
