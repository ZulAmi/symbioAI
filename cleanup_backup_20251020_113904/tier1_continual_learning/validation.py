#!/usr/bin/env python3
"""
Tier 1 Validation: Extended Continual Learning Benchmarks
=========================================================

Comprehensive validation system for core continual learning algorithms
with real benchmarks measuring:

1. Forgetting Resistance - How well the model retains old knowledge
2. Forward Transfer - How well learning new tasks helps with future tasks  
3. Backward Transfer - How new tasks affect performance on old tasks
4. Resource Scaling - How computational requirements scale with tasks
5. Task Similarity Analysis - Understanding task relationships

Benchmarks based on established continual learning research:
- Average Accuracy (AA) - Overall performance across all tasks
- Forgetting Measure (FM) - Degree of catastrophic forgetting
- Learning Accuracy (LA) - Performance on new tasks
- Forward Transfer (FT) - Positive transfer to future tasks
- Backward Transfer (BT) - Impact on previous tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import time
import json
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import psutil
import gc

from .datasets import load_tier1_dataset, create_continual_task_sequence


@dataclass
class ContinualLearningMetrics:
    """Comprehensive metrics for continual learning evaluation."""
    
    # Core metrics
    average_accuracy: float = 0.0           # Overall accuracy across all tasks
    forgetting_measure: float = 0.0         # Average forgetting across tasks
    learning_accuracy: float = 0.0          # Average accuracy when learning new tasks
    forward_transfer: float = 0.0           # Positive transfer to future tasks
    backward_transfer: float = 0.0          # Impact on previous tasks
    
    # Detailed metrics
    task_accuracies: List[List[float]] = field(default_factory=list)  # Accuracy matrix [task_i][task_j]
    final_accuracies: List[float] = field(default_factory=list)       # Final accuracy per task
    initial_accuracies: List[float] = field(default_factory=list)     # Initial accuracy per task
    
    # Resource metrics
    training_times: List[float] = field(default_factory=list)         # Training time per task
    memory_usage: List[float] = field(default_factory=list)           # Memory usage per task
    model_parameters: List[int] = field(default_factory=list)         # Parameters after each task
    
    # Additional analysis
    task_similarity_matrix: Optional[np.ndarray] = None               # Task similarity analysis
    confusion_matrices: List[np.ndarray] = field(default_factory=list) # Per-task confusion matrices
    
    def compute_summary_metrics(self):
        """Compute summary metrics from detailed measurements."""
        if not self.task_accuracies:
            return
        
        n_tasks = len(self.task_accuracies)
        
        # Average Accuracy (AA)
        self.average_accuracy = np.mean([self.task_accuracies[i][i] for i in range(n_tasks)])
        
        # Forgetting Measure (FM) - average forgetting across tasks
        forgetting_scores = []
        for i in range(n_tasks - 1):
            # Maximum accuracy achieved on task i
            max_acc_i = max(self.task_accuracies[j][i] for j in range(i, n_tasks))
            # Final accuracy on task i
            final_acc_i = self.task_accuracies[-1][i]
            # Forgetting = max_accuracy - final_accuracy
            forgetting_scores.append(max_acc_i - final_acc_i)
        
        self.forgetting_measure = np.mean(forgetting_scores) if forgetting_scores else 0.0
        
        # Learning Accuracy (LA) - average accuracy when first learning each task
        self.learning_accuracy = np.mean([self.task_accuracies[i][i] for i in range(n_tasks)])
        
        # Forward Transfer (FT) - improvement due to previous learning
        if n_tasks > 1:
            forward_transfer_scores = []
            for i in range(1, n_tasks):
                # Accuracy on task i when first encountered vs random baseline
                initial_acc = self.task_accuracies[i][i]
                # Assume random baseline is 1/num_classes (we'll estimate as 0.1 for 10 classes)
                random_baseline = 0.1
                forward_transfer_scores.append(initial_acc - random_baseline)
            
            self.forward_transfer = np.mean(forward_transfer_scores)
        
        # Backward Transfer (BT) - impact on previous tasks
        if n_tasks > 1:
            backward_transfer_scores = []
            for i in range(n_tasks - 1):
                # Accuracy on task i after learning all tasks vs after learning task i
                final_acc = self.task_accuracies[-1][i]
                initial_acc = self.task_accuracies[i][i] 
                backward_transfer_scores.append(final_acc - initial_acc)
            
            self.backward_transfer = np.mean(backward_transfer_scores)


@dataclass
class Tier1ValidationResult:
    """Results from Tier 1 continual learning validation."""
    
    dataset_name: str
    num_tasks: int
    task_type: str                                    # 'class_incremental' or 'domain_incremental'
    
    # Core results
    metrics: ContinualLearningMetrics
    success_level: str = "NEEDS_WORK"                # EXCELLENT, GOOD, NEEDS_WORK
    
    # Benchmark comparisons
    baseline_comparison: Optional[Dict[str, float]] = None  # Comparison with baseline methods
    sota_comparison: Optional[Dict[str, float]] = None      # Comparison with SOTA methods
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_minutes: float = 0.0
    device_used: str = "cpu"
    
    def get_benchmark_score(self) -> float:
        """Calculate overall benchmark score (0-1)."""
        # Weighted combination of key metrics
        weights = {
            'average_accuracy': 0.4,
            'forgetting_resistance': 0.3,  # 1 - forgetting_measure
            'forward_transfer': 0.2,
            'resource_efficiency': 0.1
        }
        
        # Normalize metrics to 0-1 scale
        acc_score = min(self.metrics.average_accuracy, 1.0)
        forget_score = max(0, 1.0 - self.metrics.forgetting_measure)
        transfer_score = max(0, min(self.metrics.forward_transfer, 0.5)) / 0.5  # Normalize to 0-1
        
        # Resource efficiency (lower is better for time/memory)
        if self.metrics.training_times:
            avg_time = np.mean(self.metrics.training_times)
            resource_score = max(0, 1.0 - min(avg_time / 300.0, 1.0))  # 5 min = 0 score
        else:
            resource_score = 0.5
        
        benchmark_score = (
            weights['average_accuracy'] * acc_score +
            weights['forgetting_resistance'] * forget_score +
            weights['forward_transfer'] * transfer_score +
            weights['resource_efficiency'] * resource_score
        )
        
        return benchmark_score


class SimpleContinualLearner(nn.Module):
    """Simple baseline continual learning model for benchmarking."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class ElasticWeightConsolidation(SimpleContinualLearner):
    """EWC (Elastic Weight Consolidation) implementation for comparison."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 10, 
                 ewc_lambda: float = 1000.0):
        super().__init__(input_dim, hidden_dim, num_classes)
        self.ewc_lambda = ewc_lambda
        self.fisher_info = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, dataloader, device):
        """Compute Fisher Information Matrix for EWC."""
        self.eval()
        fisher_info = {}
        
        for name, param in self.named_parameters():
            fisher_info[name] = torch.zeros_like(param)
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            self.zero_grad()
            
            output = self(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.pow(2)
        
        # Normalize by dataset size
        for name in fisher_info:
            fisher_info[name] /= len(dataloader.dataset)
        
        self.fisher_info = fisher_info
        
        # Store optimal parameters
        self.optimal_params = {}
        for name, param in self.named_parameters():
            self.optimal_params[name] = param.clone()
    
    def ewc_loss(self):
        """Compute EWC regularization loss."""
        if not self.fisher_info:
            return 0
        
        ewc_loss = 0
        for name, param in self.named_parameters():
            if name in self.fisher_info:
                ewc_loss += (self.fisher_info[name] * 
                           (param - self.optimal_params[name]).pow(2)).sum()
        
        return self.ewc_lambda * ewc_loss


class Tier1Validator:
    """
    Comprehensive Tier 1 validation system for continual learning.
    
    Implements standard continual learning benchmarks and metrics
    following established research methodologies.
    """
    
    def __init__(self, device: Optional[str] = None, results_dir: str = "validation/results/tier1_results"):
        """Initialize Tier 1 validator."""
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧠 Tier 1 Validator initialized (device: {self.device})")
    
    def validate_continual_learning(
        self, 
        dataset_name: str = 'fashion_mnist',
        num_tasks: int = 5,
        task_type: str = 'class_incremental',
        epochs_per_task: int = 10,
        include_baselines: bool = True
    ) -> Tier1ValidationResult:
        """
        Run comprehensive continual learning validation.
        
        Args:
            dataset_name: Dataset to use for validation
            num_tasks: Number of continual learning tasks
            task_type: 'class_incremental' or 'domain_incremental'
            epochs_per_task: Training epochs per task
            include_baselines: Whether to include baseline comparisons
            
        Returns:
            Validation results with comprehensive metrics
        """
        print(f"\n{'='*80}")
        print(f"🧠 TIER 1 CONTINUAL LEARNING VALIDATION")
        print(f"{'='*80}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"🔄 Tasks: {num_tasks} ({task_type})")
        print(f"⚡ Epochs per task: {epochs_per_task}")
        print(f"🏆 Include baselines: {include_baselines}")
        
        start_time = time.time()
        
        # Load dataset and create task sequence
        print(f"\n📥 Loading dataset and creating task sequence...")
        try:
            # Define appropriate transform for dataset
            if dataset_name in ['fashion_mnist', 'emnist']:
                transform = torch.nn.Sequential(
                    torch.nn.Flatten(start_dim=1),
                )
                input_dim = 28 * 28
            elif dataset_name in ['svhn', 'tiny_imagenet']:
                transform = torch.nn.Sequential(
                    torch.nn.Flatten(start_dim=1),
                )
                input_dim = 32 * 32 * 3 if dataset_name == 'svhn' else 64 * 64 * 3
            else:
                transform = None
                input_dim = 784  # Default
            
            dataset = load_tier1_dataset(dataset_name, train=True, transform=None)
            tasks = create_continual_task_sequence(dataset, num_tasks, task_type)
            
            # Determine number of classes and input dimensions from first sample
            sample_data, sample_target = dataset[0] 
            if hasattr(sample_data, 'shape'):
                if len(sample_data.shape) == 3:  # Color image
                    input_dim = np.prod(sample_data.shape)
                else:
                    input_dim = np.prod(sample_data.shape)
            
            # Get total number of classes
            if hasattr(dataset, 'targets'):
                num_classes = len(set(dataset.targets))
            else:
                targets = [dataset[i][1] for i in range(min(1000, len(dataset)))]
                num_classes = len(set(targets))
            
            print(f"✅ Created {len(tasks)} tasks")
            print(f"📊 Input dim: {input_dim}, Classes: {num_classes}")
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            # Return failed result
            metrics = ContinualLearningMetrics()
            return Tier1ValidationResult(
                dataset_name=dataset_name,
                num_tasks=num_tasks,
                task_type=task_type,
                metrics=metrics,
                success_level="NEEDS_WORK",
                duration_minutes=0.0,
                device_used=str(self.device)
            )
        
        # Run main validation
        print(f"\n🚀 Running continual learning validation...")
        metrics = self._run_continual_learning_benchmark(
            tasks, input_dim, num_classes, epochs_per_task
        )
        
        # Run baseline comparisons if requested
        baseline_comparison = None
        if include_baselines:
            print(f"\n🏆 Running baseline comparisons...")
            baseline_comparison = self._run_baseline_comparisons(
                tasks, input_dim, num_classes, epochs_per_task
            )
        
        # Calculate duration
        duration_minutes = (time.time() - start_time) / 60
        
        # Determine success level
        benchmark_score = self._calculate_benchmark_score(metrics)
        
        if benchmark_score >= 0.8:
            success_level = "EXCELLENT"
        elif benchmark_score >= 0.6:
            success_level = "GOOD"
        else:
            success_level = "NEEDS_WORK"
        
        # Create result
        result = Tier1ValidationResult(
            dataset_name=dataset_name,
            num_tasks=num_tasks,
            task_type=task_type,
            metrics=metrics,
            success_level=success_level,
            baseline_comparison=baseline_comparison,
            duration_minutes=duration_minutes,
            device_used=str(self.device)
        )
        
        # Print summary
        self._print_validation_summary(result)
        
        # Save results
        self._save_validation_results(result)
        
        return result
    
    def _run_continual_learning_benchmark(
        self, 
        tasks: List[torch.utils.data.Dataset],
        input_dim: int,
        num_classes: int,
        epochs_per_task: int
    ) -> ContinualLearningMetrics:
        """Run the main continual learning benchmark."""
        
        metrics = ContinualLearningMetrics()
        
        # Initialize model
        model = SimpleContinualLearner(input_dim, hidden_dim=512, num_classes=num_classes)
        model.to(self.device)
        
        # Track accuracy matrix: task_accuracies[i][j] = accuracy on task j after learning task i
        task_accuracies = []
        
        print(f"\n🔄 Training on {len(tasks)} tasks sequentially...")
        
        for task_idx, task in enumerate(tasks):
            print(f"\n{'─'*60}")
            print(f"📚 Learning Task {task_idx + 1}/{len(tasks)}")
            print(f"{'─'*60}")
            
            # Create data loader for current task
            task_loader = torch.utils.data.DataLoader(task, batch_size=128, shuffle=True, num_workers=0)
            
            # Train on current task
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            self._train_task(model, task_loader, epochs_per_task)
            
            training_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            metrics.training_times.append(training_time)
            metrics.memory_usage.append(memory_usage)
            metrics.model_parameters.append(sum(p.numel() for p in model.parameters()))
            
            print(f"✅ Task {task_idx + 1} trained in {training_time:.2f}s")
            
            # Evaluate on all tasks seen so far
            current_task_accuracies = []
            
            for eval_task_idx in range(task_idx + 1):
                eval_task = tasks[eval_task_idx]
                eval_loader = torch.utils.data.DataLoader(eval_task, batch_size=128, shuffle=False, num_workers=0)
                
                accuracy = self._evaluate_task(model, eval_loader)
                current_task_accuracies.append(accuracy)
                
                print(f"   Task {eval_task_idx + 1} accuracy: {accuracy:.4f}")
            
            # Pad with zeros for tasks not yet seen
            while len(current_task_accuracies) < len(tasks):
                current_task_accuracies.append(0.0)
            
            task_accuracies.append(current_task_accuracies)
        
        # Store results in metrics
        metrics.task_accuracies = task_accuracies
        metrics.final_accuracies = [task_accuracies[-1][i] for i in range(len(tasks))]
        metrics.initial_accuracies = [task_accuracies[i][i] for i in range(len(tasks))]
        
        # Compute summary metrics
        metrics.compute_summary_metrics()
        
        return metrics
    
    def _train_task(self, model: nn.Module, task_loader: torch.utils.data.DataLoader, epochs: int):
        """Train model on a single task."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(task_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Flatten data if needed
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            
            if epoch % (epochs // 2) == 0 or epoch == epochs - 1:
                accuracy = correct / total
                print(f"     Epoch {epoch + 1}/{epochs}: Loss {epoch_loss/len(task_loader):.4f}, Acc {accuracy:.4f}")
    
    def _evaluate_task(self, model: nn.Module, task_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate model on a single task."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in task_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Flatten data if needed
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def _run_baseline_comparisons(
        self,
        tasks: List[torch.utils.data.Dataset],
        input_dim: int,
        num_classes: int,
        epochs_per_task: int
    ) -> Dict[str, float]:
        """Run comparisons with baseline methods."""
        
        print(f"🏆 Running baseline comparisons...")
        
        baselines = {}
        
        # 1. Fine-tuning baseline (catastrophic forgetting)
        print(f"   🔄 Fine-tuning baseline...")
        ft_metrics = self._run_finetuning_baseline(tasks, input_dim, num_classes, epochs_per_task)
        baselines['finetuning'] = {
            'average_accuracy': ft_metrics.average_accuracy,
            'forgetting_measure': ft_metrics.forgetting_measure
        }
        
        # 2. Multi-task learning upper bound
        print(f"   🔄 Multi-task learning upper bound...")
        mt_accuracy = self._run_multitask_upperbound(tasks, input_dim, num_classes, epochs_per_task)
        baselines['multitask_upperbound'] = {
            'average_accuracy': mt_accuracy,
            'forgetting_measure': 0.0  # No forgetting in multi-task learning
        }
        
        # 3. EWC baseline (if time permits)
        print(f"   🔄 EWC baseline...")
        try:
            ewc_metrics = self._run_ewc_baseline(tasks, input_dim, num_classes, epochs_per_task)
            baselines['ewc'] = {
                'average_accuracy': ewc_metrics.average_accuracy,
                'forgetting_measure': ewc_metrics.forgetting_measure
            }
        except Exception as e:
            print(f"     ⚠️  EWC baseline failed: {e}")
            baselines['ewc'] = {'average_accuracy': 0.0, 'forgetting_measure': 1.0}
        
        return baselines
    
    def _run_finetuning_baseline(self, tasks, input_dim, num_classes, epochs_per_task) -> ContinualLearningMetrics:
        """Run fine-tuning baseline (standard neural network)."""
        model = SimpleContinualLearner(input_dim, num_classes=num_classes)
        model.to(self.device)
        
        metrics = ContinualLearningMetrics()
        task_accuracies = []
        
        for task_idx, task in enumerate(tasks):
            task_loader = torch.utils.data.DataLoader(task, batch_size=128, shuffle=True, num_workers=0)
            self._train_task(model, task_loader, epochs_per_task)
            
            # Evaluate on all tasks
            current_accuracies = []
            for eval_task_idx in range(len(tasks)):
                if eval_task_idx <= task_idx:
                    eval_task = tasks[eval_task_idx]
                    eval_loader = torch.utils.data.DataLoader(eval_task, batch_size=128, shuffle=False, num_workers=0)
                    accuracy = self._evaluate_task(model, eval_loader)
                    current_accuracies.append(accuracy)
                else:
                    current_accuracies.append(0.0)
            
            task_accuracies.append(current_accuracies)
        
        metrics.task_accuracies = task_accuracies
        metrics.final_accuracies = [task_accuracies[-1][i] for i in range(len(tasks))]
        metrics.initial_accuracies = [task_accuracies[i][i] for i in range(len(tasks))]
        metrics.compute_summary_metrics()
        
        return metrics
    
    def _run_multitask_upperbound(self, tasks, input_dim, num_classes, epochs_per_task) -> float:
        """Run multi-task learning upper bound."""
        # Combine all tasks into one dataset
        combined_data = []
        combined_targets = []
        
        for task in tasks:
            for i in range(len(task)):
                data, target = task[i]
                combined_data.append(data)
                combined_targets.append(target)
        
        # Create combined dataset
        class CombinedDataset(torch.utils.data.Dataset):
            def __init__(self, data, targets):
                self.data = data
                self.targets = targets
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        combined_dataset = CombinedDataset(combined_data, combined_targets)
        combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=128, shuffle=True, num_workers=0)
        
        # Train model on combined dataset
        model = SimpleContinualLearner(input_dim, num_classes=num_classes)  
        model.to(self.device)
        
        self._train_task(model, combined_loader, epochs_per_task * len(tasks))
        
        # Evaluate on combined dataset
        accuracy = self._evaluate_task(model, combined_loader)
        
        return accuracy
    
    def _run_ewc_baseline(self, tasks, input_dim, num_classes, epochs_per_task) -> ContinualLearningMetrics:
        """Run EWC (Elastic Weight Consolidation) baseline."""
        model = ElasticWeightConsolidation(input_dim, num_classes=num_classes, ewc_lambda=1000.0)
        model.to(self.device)
        
        metrics = ContinualLearningMetrics()
        task_accuracies = []
        
        for task_idx, task in enumerate(tasks):
            task_loader = torch.utils.data.DataLoader(task, batch_size=128, shuffle=True, num_workers=0)
            
            # Train with EWC regularization
            self._train_task_ewc(model, task_loader, epochs_per_task)
            
            # Compute Fisher Information after training this task
            if task_idx < len(tasks) - 1:  # Not the last task
                model.compute_fisher_information(task_loader, self.device)
            
            # Evaluate on all tasks
            current_accuracies = []
            for eval_task_idx in range(len(tasks)):
                if eval_task_idx <= task_idx:
                    eval_task = tasks[eval_task_idx]
                    eval_loader = torch.utils.data.DataLoader(eval_task, batch_size=128, shuffle=False, num_workers=0)
                    accuracy = self._evaluate_task(model, eval_loader)
                    current_accuracies.append(accuracy)
                else:
                    current_accuracies.append(0.0)
            
            task_accuracies.append(current_accuracies)
        
        metrics.task_accuracies = task_accuracies
        metrics.final_accuracies = [task_accuracies[-1][i] for i in range(len(tasks))]
        metrics.initial_accuracies = [task_accuracies[i][i] for i in range(len(tasks))]
        metrics.compute_summary_metrics()
        
        return metrics
    
    def _train_task_ewc(self, model: ElasticWeightConsolidation, task_loader, epochs: int):
        """Train EWC model with regularization."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for data, target in task_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Standard loss
                loss = criterion(output, target)
                
                # Add EWC regularization
                ewc_loss = model.ewc_loss()
                total_loss = loss + ewc_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
    
    def _calculate_benchmark_score(self, metrics: ContinualLearningMetrics) -> float:
        """Calculate overall benchmark score."""
        # Weighted combination of metrics
        weights = {
            'accuracy': 0.4,
            'forgetting': 0.3,
            'transfer': 0.2,
            'efficiency': 0.1
        }
        
        acc_score = min(metrics.average_accuracy, 1.0)
        forget_score = max(0, 1.0 - metrics.forgetting_measure)
        transfer_score = max(0, min(metrics.forward_transfer + 0.5, 1.0))  # Normalize
        
        if metrics.training_times:
            avg_time = np.mean(metrics.training_times)
            eff_score = max(0, 1.0 - min(avg_time / 300.0, 1.0))
        else:
            eff_score = 0.5
        
        return (weights['accuracy'] * acc_score + 
                weights['forgetting'] * forget_score +
                weights['transfer'] * transfer_score + 
                weights['efficiency'] * eff_score)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _print_validation_summary(self, result: Tier1ValidationResult):
        """Print validation summary."""
        print(f"\n{'='*80}")
        print(f"📊 TIER 1 VALIDATION SUMMARY: {result.success_level}")
        print(f"{'='*80}")
        
        metrics = result.metrics
        
        print(f"🎯 Core Metrics:")
        print(f"   Average Accuracy: {metrics.average_accuracy:.4f}")
        print(f"   Forgetting Measure: {metrics.forgetting_measure:.4f}")
        print(f"   Forward Transfer: {metrics.forward_transfer:.4f}")
        print(f"   Backward Transfer: {metrics.backward_transfer:.4f}")
        
        print(f"\n📈 Performance Analysis:")
        if metrics.final_accuracies:
            print(f"   Final Accuracies: {[f'{acc:.3f}' for acc in metrics.final_accuracies]}")
        
        print(f"\n⚡ Resource Usage:")
        if metrics.training_times:
            print(f"   Avg Training Time: {np.mean(metrics.training_times):.2f}s per task")
        if metrics.memory_usage:
            print(f"   Avg Memory Usage: {np.mean(metrics.memory_usage):.2f}MB")
        
        if result.baseline_comparison:
            print(f"\n🏆 Baseline Comparisons:")
            for method, scores in result.baseline_comparison.items():
                print(f"   {method}: Acc={scores['average_accuracy']:.4f}, Forget={scores['forgetting_measure']:.4f}")
        
        benchmark_score = result.get_benchmark_score()
        print(f"\n🎯 Overall Benchmark Score: {benchmark_score:.4f}")
        
        # Success level interpretation
        if result.success_level == "EXCELLENT":
            print(f"✅ EXCELLENT: Ready for academic publication and funding proposals")
        elif result.success_level == "GOOD":  
            print(f"⚠️  GOOD: Core capabilities working, some improvements needed")
        else:
            print(f"❌ NEEDS_WORK: Significant improvements required")
    
    def _save_validation_results(self, result: Tier1ValidationResult):
        """Save validation results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"tier1_continual_learning_{result.dataset_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        data = {
            'dataset_name': result.dataset_name,
            'num_tasks': result.num_tasks,
            'task_type': result.task_type,
            'success_level': result.success_level,
            'duration_minutes': result.duration_minutes,
            'device_used': result.device_used,
            'timestamp': result.timestamp,
            'benchmark_score': result.get_benchmark_score(),
            
            'metrics': {
                'average_accuracy': result.metrics.average_accuracy,
                'forgetting_measure': result.metrics.forgetting_measure,
                'learning_accuracy': result.metrics.learning_accuracy,
                'forward_transfer': result.metrics.forward_transfer,
                'backward_transfer': result.metrics.backward_transfer,
                'task_accuracies': result.metrics.task_accuracies,
                'final_accuracies': result.metrics.final_accuracies,
                'initial_accuracies': result.metrics.initial_accuracies,
                'training_times': result.metrics.training_times,
                'memory_usage': result.metrics.memory_usage,
                'model_parameters': result.metrics.model_parameters,
            },
            
            'baseline_comparison': result.baseline_comparison,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")


def evaluate_forgetting_resistance(metrics: ContinualLearningMetrics) -> Dict[str, float]:
    """Evaluate forgetting resistance from metrics."""
    return {
        'forgetting_measure': metrics.forgetting_measure,
        'forgetting_resistance': max(0, 1.0 - metrics.forgetting_measure),
        'stability_score': 1.0 - np.std(metrics.final_accuracies) if metrics.final_accuracies else 0.0
    }


def evaluate_forward_transfer(metrics: ContinualLearningMetrics) -> Dict[str, float]:
    """Evaluate forward transfer capabilities."""
    return {
        'forward_transfer': metrics.forward_transfer,
        'learning_efficiency': metrics.learning_accuracy,
        'positive_transfer_ratio': max(0, metrics.forward_transfer) / 0.5 if metrics.forward_transfer > 0 else 0.0
    }


def evaluate_resource_scaling(metrics: ContinualLearningMetrics) -> Dict[str, float]:
    """Evaluate resource scaling characteristics."""
    if not metrics.training_times:
        return {'time_scaling': 0.0, 'memory_scaling': 0.0, 'parameter_scaling': 0.0}
    
    # Calculate scaling trends
    tasks = list(range(1, len(metrics.training_times) + 1))
    
    # Time scaling
    time_trend = np.polyfit(tasks, metrics.training_times, 1)[0] if len(tasks) > 1 else 0
    time_scaling = max(0, 1.0 - time_trend / 60.0)  # Penalize if time increases >60s per task
    
    # Memory scaling  
    if metrics.memory_usage:
        memory_trend = np.polyfit(tasks, metrics.memory_usage, 1)[0] if len(tasks) > 1 else 0
        memory_scaling = max(0, 1.0 - memory_trend / 100.0)  # Penalize if memory increases >100MB per task
    else:
        memory_scaling = 0.5
    
    # Parameter scaling
    if metrics.model_parameters:
        param_trend = np.polyfit(tasks, metrics.model_parameters, 1)[0] if len(tasks) > 1 else 0
        param_scaling = max(0, 1.0 - param_trend / 1000000.0)  # Penalize if params increase >1M per task
    else:
        param_scaling = 0.5
    
    return {
        'time_scaling': time_scaling,
        'memory_scaling': memory_scaling, 
        'parameter_scaling': param_scaling,
        'overall_efficiency': (time_scaling + memory_scaling + param_scaling) / 3
    }


if __name__ == '__main__':
    """Test Tier 1 validation system."""
    print("🧪 Testing Tier 1 Validation System...")
    
    validator = Tier1Validator()
    
    # Test with Fashion-MNIST (quick validation)
    result = validator.validate_continual_learning(
        dataset_name='fashion_mnist',
        num_tasks=3,
        task_type='class_incremental', 
        epochs_per_task=2,  # Quick test
        include_baselines=True
    )
    
    print(f"\n✅ Tier 1 validation test complete!")
    print(f"   Dataset: {result.dataset_name}")
    print(f"   Success Level: {result.success_level}")
    print(f"   Benchmark Score: {result.get_benchmark_score():.4f}")
    print(f"   Duration: {result.duration_minutes:.2f} minutes")