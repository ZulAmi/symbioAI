"""
Tier 1: Extended Continual Learning Benchmarks
==============================================

This module implements validation for core continual learning algorithms
using the comprehensive dataset suite designed to validate SymbioAI's
COMBINED continual learning strategy.

Datasets:
- TinyImageNet: 200 classes, bridges CIFAR and ImageNet
- SVHN: Real-world digits, domain shift from MNIST
- Omniglot: Few-shot continual learning, 1,600+ classes
- Fashion-MNIST/EMNIST: Harder MNIST variants
- ImageNet-Subset: Realistic continual learning under visual shift

Focus Areas:
- Forgetting resistance
- Forward transfer
- Resource scaling
- Task-incremental learning
- Domain-incremental learning
"""

from .datasets import (
    load_tier1_dataset,
    create_continual_task_sequence,
    TinyImageNetDataset,
    OmniglotDataset,
    ImageNetSubsetDataset
)

from .validation import (
    Tier1Validator,
    ContinualLearningMetrics,
    evaluate_forgetting_resistance,
    evaluate_forward_transfer,
    evaluate_resource_scaling
)

__all__ = [
    'load_tier1_dataset',
    'create_continual_task_sequence', 
    'TinyImageNetDataset',
    'OmniglotDataset',
    'ImageNetSubsetDataset',
    'Tier1Validator',
    'ContinualLearningMetrics',
    'evaluate_forgetting_resistance',
    'evaluate_forward_transfer',
    'evaluate_resource_scaling'
]