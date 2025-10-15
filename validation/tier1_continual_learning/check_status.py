#!/usr/bin/env python3
"""
Quick Benchmark Status Checker
Shows current progress and estimated time remaining
"""

import json
import os
from pathlib import Path
from datetime import datetime
import glob

def check_benchmark_status():
    """Check status of running benchmarks."""
    
    results_dir = Path("../../validation/results")
    
    print("🔍 Benchmark Status Check")
    print("=" * 60)
    print()
    
    # Check if any results exist
    if not results_dir.exists():
        print("❌ No results directory found yet")
        print("   Benchmarks may still be starting up...")
        return
    
    # Find all result files
    result_files = list(results_dir.glob("*.json"))
    
    if not result_files:
        print("⏳ No completed benchmarks yet")
        print("   Check back in a few minutes...")
        return
    
    print(f"✅ Found {len(result_files)} completed benchmark(s)")
    print()
    
    # Parse and display results
    for result_file in sorted(result_files, key=os.path.getmtime):
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        dataset = result.get('dataset_name', 'unknown')
        method = result.get('method', 'unknown')
        success = result.get('success', False)
        
        print(f"📊 {dataset} - {method}")
        
        if success:
            metrics = result.get('metrics', {})
            print(f"   ✅ Success: {result.get('success_level', 'N/A')}")
            print(f"   📈 Avg Accuracy: {metrics.get('average_accuracy', 0)*100:.2f}%")
            print(f"   📉 Forgetting: {metrics.get('forgetting_measure', 0)*100:.2f}%")
            print(f"   ⏱️  Time: {result.get('total_time', 0):.1f}s")
        else:
            print(f"   ❌ Failed: {result.get('error_message', 'Unknown error')}")
        
        print()
    
    print("=" * 60)
    print()
    print("💡 To view detailed logs:")
    print("   - TensorBoard: tensorboard --logdir=./runs")
    print("   - Results JSON: ls -lh ../../validation/results/")
    print()

if __name__ == "__main__":
    check_benchmark_status()
