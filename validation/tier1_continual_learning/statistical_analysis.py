#!/usr/bin/env python3
"""
Statistical Analysis and Visualization

Compares DER++ vs Causal-DER with statistical tests and plots.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


def load_results(results_dir):
    """Load all result files."""
    results_dir = Path(results_dir)
    
    # Load DER++ baseline
    with open(results_dir / 'der_plus_plus_baseline.json', 'r') as f:
        der_baseline = json.load(f)
    
    # Load Causal-DER results
    causal_results = []
    for seed_file in results_dir.glob('causal_der_seed*.json'):
        with open(seed_file, 'r') as f:
            causal_results.append(json.load(f))
    
    # Load ablation
    try:
        with open(results_dir / 'ablation_study.json', 'r') as f:
            ablation = json.load(f)
    except:
        ablation = None
    
    return der_baseline, causal_results, ablation


def statistical_comparison(der_baseline, causal_results):
    """Perform statistical tests."""
    
    # Extract accuracies
    der_accs = [r['average_accuracy'] for r in der_baseline['individual_results']]
    causal_accs = [r['average_accuracy'] for r in causal_results]
    
    der_mean = np.mean(der_accs)
    der_std = np.std(der_accs)
    causal_mean = np.mean(causal_accs)
    causal_std = np.std(causal_accs)
    
    # T-test
    t_stat, p_value = stats.ttest_ind(causal_accs, der_accs)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((der_std**2 + causal_std**2) / 2)
    cohens_d = (causal_mean - der_mean) / pooled_std
    
    print("="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    print(f"\nDER++ (n={len(der_accs)}):")
    print(f"  Mean: {der_mean:.4f} ({der_mean*100:.2f}%)")
    print(f"  Std:  {der_std:.4f} ({der_std*100:.2f}%)")
    
    print(f"\nCausal-DER (n={len(causal_accs)}):")
    print(f"  Mean: {causal_mean:.4f} ({causal_mean*100:.2f}%)")
    print(f"  Std:  {causal_std:.4f} ({causal_std*100:.2f}%)")
    
    print(f"\nDifference: {(causal_mean - der_mean)*100:+.2f}%")
    
    print(f"\nStatistical Tests:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"  ✅ Statistically significant (p < 0.05)")
    else:
        print(f"  ❌ Not significant (p >= 0.05)")
    
    print(f"\n  Cohen's d:   {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"  Effect size: {effect}")
    
    return {
        'der_mean': der_mean,
        'der_std': der_std,
        'causal_mean': causal_mean,
        'causal_std': causal_std,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }


def plot_results(der_baseline, causal_results, stats_results, save_dir):
    """Create comparison plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data
    der_accs = [r['average_accuracy'] * 100 for r in der_baseline['individual_results']]
    causal_accs = [r['average_accuracy'] * 100 for r in causal_results]
    
    # Plot 1: Bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['DER++', 'Causal-DER']
    means = [stats_results['der_mean']*100, stats_results['causal_mean']*100]
    stds = [stats_results['der_std']*100, stats_results['causal_std']*100]
    
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(methods, means, yerr=stds, capsize=10, color=colors, alpha=0.7)
    
    ax.set_ylabel('Average Accuracy (%)', fontsize=14)
    ax.set_title('CIFAR-100 Continual Learning Benchmark', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add significance marker
    if stats_results['significant']:
        y_max = max(means) + max(stds) + 2
        ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        ax.text(0.5, y_max + 1, f"p = {stats_results['p_value']:.4f}", 
               ha='center', fontsize=12)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 1,
               f'{mean:.2f}%\n±{std:.2f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_der = np.zeros(len(der_accs)) + np.random.normal(0, 0.02, len(der_accs))
    x_causal = np.ones(len(causal_accs)) + np.random.normal(0, 0.02, len(causal_accs))
    
    ax.scatter(x_der, der_accs, color='#3498db', s=100, alpha=0.6, label='DER++')
    ax.scatter(x_causal, causal_accs, color='#e74c3c', s=100, alpha=0.6, label='Causal-DER')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(methods)
    ax.set_ylabel('Average Accuracy (%)', fontsize=14)
    ax.set_title('Individual Run Results', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Learning curves (if available)
    if causal_results and 'task_accuracies' in causal_results[0]:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Average across seeds
        der_curves = np.array([r['task_accuracies'] for r in der_baseline['individual_results']])
        causal_curves = np.array([r['task_accuracies'] for r in causal_results])
        
        # Plot average accuracy after each task
        tasks = np.arange(1, 6)
        
        der_avg = []
        causal_avg = []
        
        for t in range(5):
            # Average accuracy on all tasks seen so far
            der_task_avg = np.mean([np.mean(der_curves[seed][t]) for seed in range(len(der_curves))])
            causal_task_avg = np.mean([np.mean(causal_curves[seed][t]) for seed in range(len(causal_curves))])
            der_avg.append(der_task_avg * 100)
            causal_avg.append(causal_task_avg * 100)
        
        ax.plot(tasks, der_avg, 'o-', color='#3498db', linewidth=2, 
               markersize=8, label='DER++')
        ax.plot(tasks, causal_avg, 's-', color='#e74c3c', linewidth=2,
               markersize=8, label='Causal-DER')
        
        ax.set_xlabel('Task', fontsize=14)
        ax.set_ylabel('Average Accuracy (%)', fontsize=14)
        ax.set_title('Learning Curve: Average Accuracy After Each Task', 
                    fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xticks(tasks)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Plots saved to: {save_dir}")


def main():
    results_dir = Path(__file__).parent / 'validation' / 'results'
    
    # Load results
    print("Loading results...")
    der_baseline, causal_results, ablation = load_results(results_dir)
    
    # Statistical comparison
    stats_results = statistical_comparison(der_baseline, causal_results)
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(der_baseline, causal_results, stats_results, results_dir / 'plots')
    
    # Save statistical results
    with open(results_dir / 'statistical_analysis.json', 'w') as f:
        json.dump({
            'comparison': stats_results,
            'ablation': ablation
        }, f, indent=2)
    
    print(f"\n✅ Statistical analysis saved to: {results_dir / 'statistical_analysis.json'}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if stats_results['significant'] and stats_results['causal_mean'] > stats_results['der_mean']:
        improvement = (stats_results['causal_mean'] - stats_results['der_mean']) * 100
        print(f"✅ Causal-DER significantly outperforms DER++!")
        print(f"✅ Improvement: +{improvement:.2f}%")
        print(f"✅ p-value: {stats_results['p_value']:.6f}")
        print(f"✅ Effect size: Cohen's d = {stats_results['cohens_d']:.4f}")
        print("\n🎉 READY FOR PUBLICATION!")
    else:
        print(f"❌ No significant improvement found.")
        print(f"   Difference: {(stats_results['causal_mean'] - stats_results['der_mean'])*100:+.2f}%")
        print(f"   p-value: {stats_results['p_value']:.6f}")
        print("\n⚠️  Need to tune hyperparameters or refine algorithm.")


if __name__ == "__main__":
    main()
