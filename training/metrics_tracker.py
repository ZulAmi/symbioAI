"""
Comprehensive Metrics Tracking for Continual Learning
=====================================================

Tracks classical CL metrics (BWT, FWT, Forgetting) and causal metrics (ATE, attribution).
Critical for publication-ready experiments.

Author: Symbio AI
Date: October 22, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ContinualLearningMetrics:
    """
    Tracks comprehensive continual learning metrics.
    
    Classical Metrics:
    - Average Accuracy: Mean accuracy across all tasks
    - Forgetting: How much performance drops on old tasks
    - Backward Transfer (BWT): Average change in performance on past tasks
    - Forward Transfer (FWT): Performance on new task before training
    
    Causal Metrics:
    - ATE per sample: Causal effect of including sample in buffer
    - Forgetting Attribution: Which samples cause forgetting
    - Replay Gain: Benefit of causal vs random sampling
    """
    
    def __init__(self, num_tasks: int):
        """
        Initialize metrics tracker.
        
        Args:
            num_tasks: Total number of tasks in the sequence
        """
        self.num_tasks = num_tasks
        
        # Accuracy matrix: acc_matrix[i][j] = accuracy on task j after training on task i
        # Shape: (num_tasks, num_tasks)
        self.acc_matrix = np.zeros((num_tasks, num_tasks))
        self.acc_matrix[:] = np.nan  # Initialize with NaN to track what's been computed
        
        # Task-specific metrics
        self.task_accuracies: List[float] = []  # Final accuracy per task
        self.initial_accuracies: List[float] = []  # Accuracy before training each task
        
        # Causal metrics
        self.ate_scores: Dict[int, List[float]] = defaultdict(list)  # ATE per task
        self.harmful_samples: Dict[int, int] = {}  # Count of harmful samples per task
        self.beneficial_samples: Dict[int, int] = {}  # Count of beneficial samples per task
        self.replay_gains: Dict[int, float] = {}  # Replay gain per task
        
        # Current task
        self.current_task = 0
        
        logger.info(f"Initialized metrics tracker for {num_tasks} tasks")
    
    def record_accuracy(self, task_id: int, trained_up_to: int, accuracy: float):
        """
        Record accuracy on task_id after training up to trained_up_to.
        
        Args:
            task_id: Which task we're evaluating on
            trained_up_to: Which task we've trained up to (inclusive)
            accuracy: Accuracy percentage [0, 100]
        """
        self.acc_matrix[trained_up_to, task_id] = accuracy
    
    def record_initial_accuracy(self, task_id: int, accuracy: float):
        """
        Record initial accuracy on task_id before training.
        
        Args:
            task_id: Task being evaluated
            accuracy: Accuracy percentage [0, 100]
        """
        if len(self.initial_accuracies) == task_id:
            self.initial_accuracies.append(accuracy)
        else:
            logger.warning(f"Initial accuracy for task {task_id} already recorded")
    
    def record_final_accuracy(self, task_id: int, accuracy: float):
        """
        Record final accuracy on task_id after training it.
        
        Args:
            task_id: Task that was just trained
            accuracy: Accuracy percentage [0, 100]
        """
        if len(self.task_accuracies) == task_id:
            self.task_accuracies.append(accuracy)
            self.acc_matrix[task_id, task_id] = accuracy  # Diagonal
        else:
            logger.warning(f"Final accuracy for task {task_id} already recorded")
    
    def compute_average_accuracy(self, up_to_task: Optional[int] = None) -> float:
        """
        Compute average accuracy across all tasks seen so far.
        
        Args:
            up_to_task: Compute average up to this task (inclusive)
        
        Returns:
            Average accuracy [0, 100]
        """
        if up_to_task is None:
            up_to_task = self.current_task
        
        if up_to_task < 0:
            return 0.0
        
        # Get last row of accuracy matrix up to this task
        accuracies = self.acc_matrix[up_to_task, :up_to_task+1]
        valid_accs = accuracies[~np.isnan(accuracies)]
        
        if len(valid_accs) == 0:
            return 0.0
        
        return float(np.mean(valid_accs))
    
    def compute_forgetting(self, up_to_task: Optional[int] = None) -> float:
        """
        Compute average forgetting on all previous tasks.
        
        Forgetting on task j = max accuracy on j - final accuracy on j
        Average forgetting = mean over all j < current_task
        
        Args:
            up_to_task: Compute forgetting up to this task
        
        Returns:
            Average forgetting [0, 100]
        """
        if up_to_task is None:
            up_to_task = self.current_task
        
        if up_to_task <= 0:
            return 0.0
        
        forgetting_values = []
        
        for task_j in range(up_to_task):
            # Max accuracy ever achieved on task j
            max_acc = np.nanmax(self.acc_matrix[:, task_j])
            # Current accuracy on task j (after training up_to_task)
            current_acc = self.acc_matrix[up_to_task, task_j]
            
            if not np.isnan(max_acc) and not np.isnan(current_acc):
                forgetting = max_acc - current_acc
                forgetting_values.append(max(0.0, forgetting))  # Only count positive forgetting
        
        if len(forgetting_values) == 0:
            return 0.0
        
        return float(np.mean(forgetting_values))
    
    def compute_backward_transfer(self, up_to_task: Optional[int] = None) -> float:
        """
        Compute Backward Transfer (BWT).
        
        BWT = (1 / (T-1)) * Σ_{i=1}^{T-1} (acc_{T,i} - acc_{i,i})
        
        Measures how much old tasks improve (positive) or degrade (negative).
        
        Args:
            up_to_task: Compute BWT up to this task
        
        Returns:
            BWT (can be negative)
        """
        if up_to_task is None:
            up_to_task = self.current_task
        
        if up_to_task <= 0:
            return 0.0
        
        bwt_values = []
        
        for task_i in range(up_to_task):
            # Accuracy on task i after training all tasks up to up_to_task
            acc_final = self.acc_matrix[up_to_task, task_i]
            # Accuracy on task i right after training task i
            acc_initial = self.acc_matrix[task_i, task_i]
            
            if not np.isnan(acc_final) and not np.isnan(acc_initial):
                bwt = acc_final - acc_initial
                bwt_values.append(bwt)
        
        if len(bwt_values) == 0:
            return 0.0
        
        return float(np.mean(bwt_values))
    
    def compute_forward_transfer(self, up_to_task: Optional[int] = None) -> float:
        """
        Compute Forward Transfer (FWT).
        
        FWT = (1 / (T-1)) * Σ_{i=2}^{T} (acc_{i-1,i} - acc_random)
        
        Measures zero-shot performance on new tasks (before training them).
        acc_random = random guess baseline (e.g., 10% for 10-class task).
        
        Args:
            up_to_task: Compute FWT up to this task
        
        Returns:
            FWT
        """
        if up_to_task is None:
            up_to_task = self.current_task
        
        if up_to_task <= 1:
            return 0.0
        
        # Assume 10% random baseline for now (works for CIFAR-100 with 10 tasks)
        random_baseline = 100.0 / self.num_tasks
        
        fwt_values = []
        
        for task_i in range(1, up_to_task + 1):
            # Accuracy on task i before training it (after training task i-1)
            if task_i - 1 >= 0:
                acc_before = self.acc_matrix[task_i - 1, task_i]
                
                if not np.isnan(acc_before):
                    fwt = acc_before - random_baseline
                    fwt_values.append(fwt)
        
        if len(fwt_values) == 0:
            return 0.0
        
        return float(np.mean(fwt_values))
    
    def record_ate_score(self, task_id: int, ate: float):
        """
        Record ATE score for a sample from task_id.
        
        Args:
            task_id: Task the sample belongs to
            ate: Average Treatment Effect
        """
        self.ate_scores[task_id].append(ate)
    
    def record_sample_attribution(self, task_id: int, ate: float, threshold: float = 0.05):
        """
        Categorize sample as harmful/beneficial based on ATE.
        
        Args:
            task_id: Task the sample belongs to
            ate: Average Treatment Effect
            threshold: ATE threshold for classification
        """
        self.record_ate_score(task_id, ate)
        
        if ate < -threshold:
            self.harmful_samples[task_id] = self.harmful_samples.get(task_id, 0) + 1
        elif ate > threshold:
            self.beneficial_samples[task_id] = self.beneficial_samples.get(task_id, 0) + 1
    
    def record_replay_gain(self, task_id: int, gain: float):
        """
        Record replay gain (causal vs random sampling benefit).
        
        Args:
            task_id: Task ID
            gain: Percentage improvement from causal sampling
        """
        self.replay_gains[task_id] = gain
    
    def get_causal_metrics_summary(self) -> Dict:
        """
        Get summary of causal metrics.
        
        Returns:
            Dictionary with causal metrics
        """
        all_ates = []
        for ates in self.ate_scores.values():
            all_ates.extend(ates)
        
        total_harmful = sum(self.harmful_samples.values())
        total_beneficial = sum(self.beneficial_samples.values())
        avg_replay_gain = np.mean(list(self.replay_gains.values())) if self.replay_gains else 0.0
        
        return {
            'ate_mean': np.mean(all_ates) if all_ates else 0.0,
            'ate_std': np.std(all_ates) if all_ates else 0.0,
            'ate_median': np.median(all_ates) if all_ates else 0.0,
            'num_ate_samples': len(all_ates),
            'harmful_samples': total_harmful,
            'beneficial_samples': total_beneficial,
            'avg_replay_gain': avg_replay_gain,
        }
    
    def get_summary(self, task_id: Optional[int] = None) -> Dict:
        """
        Get comprehensive metrics summary.
        
        Args:
            task_id: Get summary up to this task (None = current)
        
        Returns:
            Dictionary with all metrics
        """
        if task_id is None:
            task_id = self.current_task
        
        avg_acc = self.compute_average_accuracy(task_id)
        forgetting = self.compute_forgetting(task_id)
        bwt = self.compute_backward_transfer(task_id)
        fwt = self.compute_forward_transfer(task_id)
        
        summary = {
            'task_id': task_id,
            'average_accuracy': avg_acc,
            'forgetting': forgetting,
            'backward_transfer': bwt,
            'forward_transfer': fwt,
            'final_accuracy': self.task_accuracies[task_id] if task_id < len(self.task_accuracies) else 0.0,
        }
        
        # Add causal metrics if available
        if self.ate_scores:
            summary['causal_metrics'] = self.get_causal_metrics_summary()
        
        return summary
    
    def print_summary(self, task_id: Optional[int] = None):
        """
        Print formatted metrics summary.
        
        Args:
            task_id: Print summary up to this task
        """
        summary = self.get_summary(task_id)
        
        print("\n" + "="*60)
        print(f"Metrics Summary - After Task {summary['task_id']}")
        print("="*60)
        print(f"Average Accuracy:    {summary['average_accuracy']:.2f}%")
        print(f"Final Task Accuracy: {summary['final_accuracy']:.2f}%")
        print(f"Forgetting:          {summary['forgetting']:.2f}%")
        print(f"Backward Transfer:   {summary['backward_transfer']:+.2f}%")
        print(f"Forward Transfer:    {summary['forward_transfer']:+.2f}%")
        
        if 'causal_metrics' in summary:
            cm = summary['causal_metrics']
            print("\n" + "-"*60)
            print("Causal Metrics")
            print("-"*60)
            print(f"ATE (mean ± std):    {cm['ate_mean']:.4f} ± {cm['ate_std']:.4f}")
            print(f"ATE (median):        {cm['ate_median']:.4f}")
            print(f"Samples analyzed:    {cm['num_ate_samples']}")
            print(f"Harmful samples:     {cm['harmful_samples']}")
            print(f"Beneficial samples:  {cm['beneficial_samples']}")
            if cm['avg_replay_gain'] != 0.0:
                print(f"Avg Replay Gain:     {cm['avg_replay_gain']:+.2f}%")
        
        print("="*60 + "\n")
    
    def update_current_task(self, task_id: int):
        """Update the current task being trained."""
        self.current_task = task_id
    
    def save_to_file(self, filepath: str):
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Path to save JSON
        """
        import json
        
        data = {
            'num_tasks': self.num_tasks,
            'acc_matrix': self.acc_matrix.tolist(),
            'task_accuracies': self.task_accuracies,
            'initial_accuracies': self.initial_accuracies,
            'ate_scores': {k: v for k, v in self.ate_scores.items()},
            'harmful_samples': self.harmful_samples,
            'beneficial_samples': self.beneficial_samples,
            'replay_gains': self.replay_gains,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")


def estimate_ate_for_sample(
    model: nn.Module,
    sample_data: torch.Tensor,
    sample_target: torch.Tensor,
    buffer_samples: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_reference: int = 20
) -> float:
    """
    Estimate ATE for a single sample via counterfactual removal.
    
    Question: "What would the loss be if we remove this sample from buffer?"
    
    ATE = E[Loss | with sample] - E[Loss | without sample]
    
    Positive ATE → sample prevents forgetting → beneficial
    Negative ATE → sample causes forgetting → harmful
    
    Args:
        model: Neural network
        sample_data: Sample to test [C, H, W]
        sample_target: Sample label
        buffer_samples: Other buffer samples [(data, target), ...]
        device: Device
        num_reference: How many buffer samples to use for estimation
    
    Returns:
        ATE score
    """
    if len(buffer_samples) < 5:
        return 0.0  # Not enough data
    
    with torch.no_grad():
        # Select random subset of buffer for efficiency
        if len(buffer_samples) > num_reference:
            indices = np.random.choice(len(buffer_samples), num_reference, replace=False)
            reference_samples = [buffer_samples[i] for i in indices]
        else:
            reference_samples = buffer_samples
        
        # Prepare batch WITH sample
        all_data_with = [sample_data] + [s[0] for s in reference_samples]
        all_targets_with = [sample_target] + [s[1] for s in reference_samples]
        
        batch_data_with = torch.stack(all_data_with).to(device)
        batch_targets_with = torch.stack(all_targets_with).to(device)
        
        # Compute loss with sample
        outputs_with = model(batch_data_with)
        loss_with = F.cross_entropy(outputs_with, batch_targets_with, reduction='mean')
        
        # Prepare batch WITHOUT sample
        batch_data_without = torch.stack([s[0] for s in reference_samples]).to(device)
        batch_targets_without = torch.stack([s[1] for s in reference_samples]).to(device)
        
        # Compute loss without sample
        outputs_without = model(batch_data_without)
        loss_without = F.cross_entropy(outputs_without, batch_targets_without, reduction='mean')
        
        # ATE = loss prevented by including sample
        ate = float(loss_without - loss_with)
    
    return ate


__all__ = [
    'ContinualLearningMetrics',
    'estimate_ate_for_sample',
]
