#!/usr/bin/env python3
"""
LOCAL PRE-FLIGHT TEST: Verify Graph and TRUE causality modes work correctly.

This runs a MINIMAL test (3 tasks, 1 epoch) to verify:
1. Graph heuristic mode (use_causal_sampling=1, enable_causal_graph_learning=1)
2. TRUE interventional mode (use_causal_sampling=3)

Run this BEFORE spending money on RunPod to catch any obvious bugs!
"""

import sys
import subprocess
import os

def run_test(name, args):
    """Run a test and capture output."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Command: python3 mammoth/utils/main.py {' '.join(args)}")
    print()
    
    cmd = ['python3', 'mammoth/utils/main.py'] + args
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Check for specific success indicators
    output = result.stdout + result.stderr
    
    success = True
    issues = []
    
    # Check for completion
    if "Accuracy for" not in output:
        success = False
        issues.append("❌ No final accuracy reported")
    
    # Check for crashes
    if result.returncode != 0:
        success = False
        issues.append(f"❌ Process exited with code {result.returncode}")
    
    # Check for TRUE causality specific issues
    if 'use_causal_sampling=3' in ' '.join(args):
        if "CAUSALITY] Sampling" not in output:
            success = False
            issues.append("❌ TRUE causality not triggered")
        if "all causal effects are 0.0000" in output.lower():
            issues.append("⚠️  WARNING: All causal effects are zero (might be bug)")
    
    # Check for Graph heuristic specific issues  
    if 'use_causal_sampling=1' in ' '.join(args) and 'enable_causal_graph_learning=1' in ' '.join(args):
        if "PC algorithm" not in output and "causal graph" not in output.lower():
            issues.append("⚠️  WARNING: Graph learning might not be active")
    
    print()
    if success and not issues:
        print(f"✅ {name} PASSED")
    elif not success:
        print(f"❌ {name} FAILED")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"⚠️  {name} COMPLETED WITH WARNINGS")
        for issue in issues:
            print(f"  {issue}")
    
    return success, issues, output

def main():
    print("="*60)
    print("LOCAL PRE-FLIGHT CAUSALITY TEST")
    print("="*60)
    print("Testing 3 tasks, 1 epoch each (FAST, ~3-5 minutes total)")
    print()
    
    # Common args for all tests (minimal config)
    base_args = [
        '--dataset', 'seq-cifar100',
        '--buffer_size', '100',  # Small buffer for speed
        '--n_epochs', '1',        # Only 1 epoch
        '--lr', '0.03',
        '--alpha', '0.1',
        '--beta', '0.5',
        '--seed', '42',
        '--batch_size', '32'
    ]
    
    results = {}
    
    # TEST 1: Graph Heuristic (Mode 1)
    print("\n" + "="*60)
    print("TEST 1: GRAPH HEURISTIC MODE")
    print("="*60)
    graph_args = base_args + [
        '--model', 'derpp-causal',
        '--enable_causal_graph_learning', '1',
        '--use_causal_sampling', '1',
    ]
    success, issues, output = run_test("Graph Heuristic", graph_args)
    results['graph'] = (success, issues, output)
    
    # TEST 2: TRUE Interventional (Mode 3)
    print("\n" + "="*60)
    print("TEST 2: TRUE INTERVENTIONAL MODE")
    print("="*60)
    true_args = base_args + [
        '--model', 'derpp-causal',
        '--use_causal_sampling', '3',
        '--debug', '1',
        '--temperature', '2.0',
        '--causal_eval_interval', '1',  # Measure every step for testing
        '--causal_num_interventions', '10',  # Small for speed
    ]
    success, issues, output = run_test("TRUE Interventional", true_args)
    results['true'] = (success, issues, output)
    
    # SUMMARY
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, (success, issues, _) in results.items():
        status = "✅ PASS" if success and not issues else ("⚠️  WARN" if success else "❌ FAIL")
        print(f"{status} - {name.upper()}")
        if issues:
            for issue in issues:
                print(f"     {issue}")
        all_passed = all_passed and success
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED - SAFE TO RUN ON RUNPOD")
        print()
        print("Recommended RunPod command (5 epochs, correct config):")
        print()
        print("# Graph Heuristic:")
        print("python utils/main.py --model derpp-causal --dataset seq-cifar100 \\")
        print("  --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 \\")
        print("  --lr_milestones 3 4 --enable_causal_graph_learning 1 \\")
        print("  --use_causal_sampling 1 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/graph_5ep.log")
        print()
        print("# TRUE Interventional:")
        print("python utils/main.py --model derpp-causal --dataset seq-cifar100 \\")
        print("  --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 \\")
        print("  --lr_milestones 3 4 --use_causal_sampling 3 --debug 1 \\")
        print("  --temperature 2.0 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/true_5ep.log")
        print()
        print("# Vanilla DER++:")
        print("python utils/main.py --model derpp --dataset seq-cifar100 \\")
        print("  --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 \\")
        print("  --lr_milestones 3 4 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/vanilla_5ep.log")
        return 0
    else:
        print("❌ TESTS FAILED - DO NOT RUN ON RUNPOD YET")
        print("Fix the issues above before spending money!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
