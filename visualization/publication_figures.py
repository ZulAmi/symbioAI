"""
Publication-Quality Visualization Framework
==========================================

Generates publication-ready figures for continual learning experiments.

Figures included:
1. Accuracy vs Task progression
2. Task Causal Graph (network visualization)
3. ATE histogram (distribution of causal effects)
4. Forgetting curves
5. Ablation study comparisons

Author: Symbio AI
Date: October 22, 2025
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
sns.set_palette("husl")


def plot_accuracy_vs_task(
    results_dict: Dict[str, List[float]],
    output_path: str = 'figures/accuracy_vs_task.pdf',
    title: str = 'Accuracy Progression: Causal-DER on CIFAR-100'
):
    """
    Plot Task-IL and Class-IL accuracy progression across tasks.
    
    Args:
        results_dict: Dictionary with 'task_il' and 'class_il' keys
        output_path: Where to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    tasks = list(range(len(results_dict['task_il'])))
    
    # Plot Task-IL and Class-IL
    ax.plot(tasks, results_dict['task_il'], 'o-', label='Task-IL', 
            linewidth=2, markersize=8, alpha=0.8)
    ax.plot(tasks, results_dict['class_il'], 's-', label='Class-IL', 
            linewidth=2, markersize=8, alpha=0.8)
    
    # Add baseline if provided
    if 'baseline' in results_dict:
        ax.axhline(results_dict['baseline'], color='red', linestyle='--', 
                   linewidth=2, label=f"DER++ Baseline ({results_dict['baseline']:.1f}%)")
    
    ax.set_xlabel('Task', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved accuracy plot to {output_path}")
    plt.close()


def plot_forgetting_curve(
    forgetting_per_task: List[float],
    output_path: str = 'figures/forgetting_curve.pdf',
    title: str = 'Forgetting Over Tasks'
):
    """
    Plot forgetting metric across tasks.
    
    Args:
        forgetting_per_task: List of forgetting values per task
        output_path: Where to save
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    tasks = list(range(len(forgetting_per_task)))
    
    ax.plot(tasks, forgetting_per_task, 'o-', linewidth=2, markersize=8, 
            color='orangered', label='Forgetting')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Task', fontsize=14)
    ax.set_ylabel('Forgetting (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved forgetting curve to {output_path}")
    plt.close()


def plot_causal_graph(
    adjacency_matrix: np.ndarray,
    task_names: Optional[List[str]] = None,
    output_path: str = 'figures/causal_graph.pdf',
    title: str = 'Learned Task Causal Graph',
    threshold: float = 0.1
):
    """
    Visualize learned task causal graph as a directed network.
    
    Args:
        adjacency_matrix: N×N numpy array of edge weights
        task_names: Optional task names (default: T0, T1, ...)
        output_path: Where to save
        title: Plot title
        threshold: Minimum edge weight to display
    """
    try:
        import networkx as nx
    except ImportError:
        print("⚠️  networkx not installed. Run: pip install networkx")
        return
    
    num_tasks = adjacency_matrix.shape[0]
    if task_names is None:
        task_names = [f"T{i}" for i in range(num_tasks)]
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with weights above threshold
    edge_count = 0
    for i in range(num_tasks):
        for j in range(num_tasks):
            weight = adjacency_matrix[i, j]
            if i != j and abs(weight) > threshold:
                G.add_edge(task_names[i], task_names[j], weight=weight)
                edge_count += 1
    
    if edge_count == 0:
        print(f"⚠️  No edges above threshold {threshold}. Skipping graph plot.")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue', 
                           edgecolors='black', linewidths=2, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax)
    
    # Draw edges with varying width based on weight
    edges = G.edges()
    weights = np.array([G[u][v]['weight'] for u, v in edges])
    
    # Normalize weights for visualization
    max_weight = np.abs(weights).max()
    widths = (np.abs(weights) / max_weight) * 5 + 0.5
    
    # Color by sign
    edge_colors = ['green' if w > 0 else 'red' for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7, arrows=True, 
                           arrowsize=20, arrowstyle='->', edge_color=edge_colors, ax=ax)
    
    # Add legend
    green_line = plt.Line2D([0], [0], color='green', linewidth=3, label='Positive (helps)')
    red_line = plt.Line2D([0], [0], color='red', linewidth=3, label='Negative (interferes)')
    ax.legend(handles=[green_line, red_line], loc='upper right', fontsize=12)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved causal graph to {output_path}")
    plt.close()


def plot_ate_histogram(
    ate_scores: List[float],
    output_path: str = 'figures/ate_histogram.pdf',
    title: str = 'Distribution of Sample-Level ATE Scores',
    bins: int = 50
):
    """
    Plot histogram of ATE scores for buffer samples.
    
    Args:
        ate_scores: List of ATE values
        output_path: Where to save
        title: Plot title
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot histogram
    ax.hist(ate_scores, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add vertical line at 0
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral (ATE=0)')
    
    # Add statistics
    mean_ate = np.mean(ate_scores)
    median_ate = np.median(ate_scores)
    ax.axvline(mean_ate, color='orange', linestyle='--', linewidth=2, 
               label=f'Mean ({mean_ate:.3f})')
    ax.axvline(median_ate, color='green', linestyle='--', linewidth=2, 
               label=f'Median ({median_ate:.3f})')
    
    ax.set_xlabel('Average Treatment Effect (ATE)', fontsize=14)
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    
    # Add text box with statistics
    stats_text = f"Mean: {mean_ate:.4f}\nStd: {np.std(ate_scores):.4f}\nMedian: {median_ate:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved ATE histogram to {output_path}")
    plt.close()


def plot_ablation_study(
    results: Dict[str, List[float]],
    output_path: str = 'figures/ablation_study.pdf',
    title: str = 'Ablation Study: Component Contributions',
    ylabel: str = 'Task-IL Accuracy (%)'
):
    """
    Plot ablation study comparing different configurations.
    
    Args:
        results: Dictionary mapping method name to accuracy list
                 e.g., {'Baseline': [70.1, 70.5, ...], 'With Graph': [71.2, 71.8, ...]}
        output_path: Where to save
        title: Plot title
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Determine number of tasks
    num_tasks = max(len(v) for v in results.values())
    tasks = list(range(num_tasks))
    
    # Plot each configuration
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    for idx, (method_name, accuracies) in enumerate(results.items()):
        marker = markers[idx % len(markers)]
        ax.plot(tasks[:len(accuracies)], accuracies, marker=marker, linewidth=2, 
                markersize=8, label=method_name, alpha=0.8)
    
    ax.set_xlabel('Task', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, num_tasks - 0.5)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved ablation study to {output_path}")
    plt.close()


def plot_multiseed_results(
    results: Dict[str, Dict[str, np.ndarray]],
    output_path: str = 'figures/multiseed_comparison.pdf',
    title: str = 'Multi-Seed Results Comparison'
):
    """
    Plot results across multiple seeds with confidence intervals.
    
    Args:
        results: Dictionary mapping method to {'mean': np.array, 'std': np.array}
        output_path: Where to save
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    num_tasks = max(len(v['mean']) for v in results.values())
    tasks = np.arange(num_tasks)
    
    for method_name, stats in results.items():
        mean = stats['mean']
        std = stats['std']
        
        ax.plot(tasks[:len(mean)], mean, 'o-', linewidth=2, markersize=8, 
                label=method_name, alpha=0.8)
        ax.fill_between(tasks[:len(mean)], mean - std, mean + std, alpha=0.2)
    
    ax.set_xlabel('Task', fontsize=14)
    ax.set_ylabel('Task-IL Accuracy (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, num_tasks - 0.5)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved multi-seed results to {output_path}")
    plt.close()


def create_results_table(
    metrics_dict: Dict[str, Dict],
    output_path: str = 'figures/results_table.txt'
):
    """
    Create LaTeX-formatted results table.
    
    Args:
        metrics_dict: Dictionary mapping method name to metrics dict
        output_path: Where to save
    """
    # Create header
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Continual Learning Results on CIFAR-100 (10 tasks)}")
    lines.append("\\label{tab:results}")
    lines.append("\\begin{tabular}{l|cccc}")
    lines.append("\\hline")
    lines.append("Method & Avg Acc & Forgetting & BWT & FWT \\\\")
    lines.append("\\hline")
    
    # Add rows
    for method, metrics in metrics_dict.items():
        avg_acc = metrics.get('average_accuracy', 0.0)
        forgetting = metrics.get('forgetting', 0.0)
        bwt = metrics.get('backward_transfer', 0.0)
        fwt = metrics.get('forward_transfer', 0.0)
        
        line = f"{method} & {avg_acc:.2f} & {forgetting:.2f} & {bwt:+.2f} & {fwt:+.2f} \\\\"
        lines.append(line)
    
    # Close table
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Saved LaTeX table to {output_path}")


def load_metrics_from_log(log_path: str) -> Dict:
    """
    Parse metrics from log file.
    
    Args:
        log_path: Path to log file
    
    Returns:
        Dictionary with parsed metrics
    """
    task_il_accs = []
    class_il_accs = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse lines like: "Task 0: Task-IL Acc: 99.30%, Class-IL Acc: 32.80%"
            if 'Task-IL Acc:' in line and 'Class-IL Acc:' in line:
                parts = line.strip().split(',')
                
                # Extract Task-IL
                task_il_str = parts[0].split('Task-IL Acc:')[1].strip().replace('%', '')
                task_il_accs.append(float(task_il_str))
                
                # Extract Class-IL
                class_il_str = parts[1].split('Class-IL Acc:')[1].strip().replace('%', '')
                class_il_accs.append(float(class_il_str))
    
    return {
        'task_il': task_il_accs,
        'class_il': class_il_accs,
        'final_task_il': task_il_accs[-1] if task_il_accs else 0.0,
        'final_class_il': class_il_accs[-1] if class_il_accs else 0.0,
    }


if __name__ == '__main__':
    # Example usage
    print("📊 Visualization Framework Initialized")
    print("Example functions:")
    print("  - plot_accuracy_vs_task()")
    print("  - plot_causal_graph()")
    print("  - plot_ate_histogram()")
    print("  - plot_ablation_study()")
    print("  - plot_multiseed_results()")
    print("  - create_results_table()")
