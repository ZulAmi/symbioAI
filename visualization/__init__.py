"""
Visualization Package for Publication Figures
==============================================

Provides publication-quality plotting functions for continual learning
experiments.

Usage:
    from visualization.publication_figures import plot_accuracy_vs_task
    
    results = {'task_il': [...], 'class_il': [...]}
    plot_accuracy_vs_task(results, 'figures/accuracy.pdf')

Author: Symbio AI
Date: October 22, 2025
"""

__version__ = "1.0.0"

from .publication_figures import (
    plot_accuracy_vs_task,
    plot_forgetting_curve,
    plot_causal_graph,
    plot_ate_histogram,
    plot_ablation_study,
    plot_multiseed_results,
    create_results_table,
    load_metrics_from_log,
)

__all__ = [
    'plot_accuracy_vs_task',
    'plot_forgetting_curve', 
    'plot_causal_graph',
    'plot_ate_histogram',
    'plot_ablation_study',
    'plot_multiseed_results',
    'create_results_table',
    'load_metrics_from_log',
]
