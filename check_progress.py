#!/usr/bin/env python3
"""
Benchmark Progress Monitor
Run this to check the status of your benchmarks.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta

def check_progress():
    """Check benchmark progress."""
    
    results_dir = Path('experiments/results')
    
    if not results_dir.exists():
        print("❌ No results directory found")
        print("   Benchmarks haven't started yet")
        return
    
    # Find latest results
    result_files = list(results_dir.glob('benchmark_results_*.json'))
    
    if not result_files:
        print("⏳ Benchmarks running...")
        print("   No results saved yet")
        print("   Check back in 1-2 hours")
        return
    
    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest) as f:
        data = json.load(f)
    
    print("="*70)
    print("📊 BENCHMARK PROGRESS")
    print("="*70)
    
    mode = data.get('mode', 'unknown')
    timestamp = data.get('timestamp', 'unknown')
    
    print(f"Mode: {mode}")
    print(f"Started: {timestamp}")
    
    results = data.get('results', {})
    
    for benchmark, bench_data in results.items():
        print(f"\n🔬 {benchmark.upper()}")
        print(f"   Tasks: {bench_data.get('num_tasks', '?')}")
        print(f"   Epochs/task: {bench_data.get('epochs_per_task', '?')}")
        
        strategies = bench_data.get('strategies', {})
        for strategy, strat_data in strategies.items():
            status = strat_data.get('status', 'pending')
            acc = strat_data.get('accuracy', 0)
            
            if status == 'complete':
                print(f"   ✅ {strategy}: {acc:.1%} accuracy")
            else:
                print(f"   ⏳ {strategy}: {status}")
    
    total_time = data.get('total_time_seconds', 0)
    if total_time > 0:
        print(f"\nTotal time: {total_time/3600:.2f} hours")
    
    print("="*70)

def estimate_completion():
    """Estimate when benchmarks will complete."""
    print("\n⏰ TIME ESTIMATES:")
    print("─"*70)
    
    estimates = {
        'test': ('30 minutes', 0.5),
        'fast': ('4-6 hours', 5),
        'full': ('12-16 hours', 14),
        'publication': ('24-48 hours', 36)
    }
    
    for mode, (duration, hours) in estimates.items():
        cost = hours * 0.50  # RTX 4090 price
        print(f"{mode:12s}: {duration:15s} (~${cost:.2f})")
    
    print("─"*70)

if __name__ == '__main__':
    check_progress()
    estimate_completion()
    
    print("\n💡 TIP: Run this script periodically to check progress")
    print("   Or check the tmux session: tmux attach -t benchmarks")
