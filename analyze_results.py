#!/usr/bin/env python3
"""
Analyze and summarize all experimental results for the research paper.
"""

import re
import numpy as np
from pathlib import Path
from scipy import stats

def extract_accuracy(log_file, metric="Task-IL"):
    """Extract final accuracy from log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Find last occurrence of accuracy
        pattern = rf"Accuracy for \d+ task.*?\[{metric}\]:\s*([\d.]+)\s*%"
        matches = re.findall(pattern, content)
        
        if matches:
            return float(matches[-1])
        return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None

def analyze_multiseed(results_dir, dataset_name):
    """Analyze multi-seed results for a dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} Multi-Seed Analysis")
    print(f"{'='*70}")
    
    task_il_scores = []
    class_il_scores = []
    
    log_files = sorted(Path(results_dir).glob("seed_*.log"))
    
    for log_file in log_files:
        seed_num = re.search(r'seed_(\d+)', str(log_file)).group(1)
        
        task_il = extract_accuracy(log_file, "Task-IL")
        class_il = extract_accuracy(log_file, "Class-IL")
        
        if task_il:
            task_il_scores.append(task_il)
            print(f"  Seed {seed_num}: Task-IL = {task_il:.2f}%, Class-IL = {class_il:.2f}%")
            class_il_scores.append(class_il)
    
    if task_il_scores:
        task_mean = np.mean(task_il_scores)
        task_std = np.std(task_il_scores, ddof=1)
        class_mean = np.mean(class_il_scores)
        class_std = np.std(class_il_scores, ddof=1)
        
        print(f"\n  Summary Statistics:")
        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  Task-IL:  {task_mean:.2f} ± {task_std:.2f}% (n={len(task_il_scores)})")
        print(f"  Class-IL: {class_mean:.2f} ± {class_std:.2f}% (n={len(class_il_scores)})")
        print(f"  Min/Max Task-IL: {min(task_il_scores):.2f}% / {max(task_il_scores):.2f}%")
        
        return {
            'task_il_mean': task_mean,
            'task_il_std': task_std,
            'task_il_scores': task_il_scores,
            'class_il_mean': class_mean,
            'class_il_std': class_std,
            'class_il_scores': class_il_scores,
        }
    return None

def main():
    base_dir = Path(__file__).parent / "validation" / "results"
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║           CAUSALDER EXPERIMENTAL RESULTS SUMMARY                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # CIFAR-100
    cifar100_results = analyze_multiseed(
        base_dir / "multiseed/quickwins_20251024_101754",
        "CIFAR-100"
    )
    
    # MNIST
    mnist_results = analyze_multiseed(
        base_dir / "quick_validation_3datasets/mnist",
        "MNIST"
    )
    
    # CIFAR-10 (single seed - need to check both baseline and causal)
    print(f"\n{'='*70}")
    print(f"  CIFAR-10 Single-Seed Analysis")
    print(f"{'='*70}")
    
    cifar10_baseline = extract_accuracy(
        base_dir / "quick_validation_3datasets/cifar10/baseline_seed1.log",
        "Task-IL"
    )
    cifar10_causal = extract_accuracy(
        base_dir / "quick_validation_3datasets/cifar10/causal_seed1.log",
        "Task-IL"
    )
    
    if cifar10_baseline and cifar10_causal:
        print(f"  Baseline DER++: {cifar10_baseline:.2f}%")
        print(f"  CausalDER:      {cifar10_causal:.2f}%")
        print(f"  Gap:            {cifar10_causal - cifar10_baseline:.2f}%")
    
    # Summary Table
    print(f"\n{'='*70}")
    print(f"  PUBLICATION-READY SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n  Dataset       | Task-IL Acc (%)  | Class-IL Acc (%) | # Seeds")
    print(f"  ──────────────┼──────────────────┼──────────────────┼─────────")
    
    if cifar100_results:
        print(f"  CIFAR-100     | {cifar100_results['task_il_mean']:.2f} ± {cifar100_results['task_il_std']:.2f}      | {cifar100_results['class_il_mean']:.2f} ± {cifar100_results['class_il_std']:.2f}      | 5")
    
    if cifar10_causal:
        print(f"  CIFAR-10      | {cifar10_causal:.2f}             | (see log)        | 1")
    
    if mnist_results:
        print(f"  MNIST         | {mnist_results['task_il_mean']:.2f} ± {mnist_results['task_il_std']:.2f}       | {mnist_results['class_il_mean']:.2f} ± {mnist_results['class_il_std']:.2f}      | 5")
    
    print(f"\n{'='*70}")
    print(f"  NEXT STEPS CHECKLIST")
    print(f"{'='*70}")
    print(f"\n  ✅ COMPLETED:")
    print(f"     • CIFAR-100 multi-seed validation (5 seeds)")
    print(f"     • MNIST multi-seed validation (5 seeds)")
    print(f"     • CIFAR-10 preliminary test (1 seed)")
    print(f"\n  🔴 CRITICAL - DO NEXT:")
    print(f"     • Run DER++ baseline on CIFAR-100 (5 seeds) for statistical comparison")
    print(f"     • Component ablation study (graph only, sampling only, etc.)")
    print(f"     • CIFAR-10 multi-seed (3-5 seeds)")
    print(f"\n  🟡 HIGH PRIORITY:")
    print(f"     • Compare against ER-ACE, GDumb (SOTA methods)")
    print(f"     • TinyImageNet scalability test")
    print(f"\n  🟢 MEDIUM PRIORITY:")
    print(f"     • Statistical significance tests (t-test, Wilcoxon)")
    print(f"     • Computational overhead analysis")
    print(f"     • Sensitivity analysis (hyperparameters)")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
