"""
Continual Learning Benchmarks for Research Publication
Phase 1: Build Publication Record - Benchmark Infrastructure

This module provides comprehensive benchmarking for continual learning systems,
implementing standard benchmarks used in NeurIPS/ICML/ICLR publications:

- Split CIFAR-100 (20 tasks, 5 classes each)
- Split MNIST (5 tasks, 2 classes each)  
- Permuted MNIST (10 tasks, different pixel permutations)
- Split ImageNet (subset for faster evaluation)

Compares our combined approach vs individual methods and baselines.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.continual_learning import (
    ContinualLearningEngine,
    Task,
    TaskType,
    ForgettingPreventionStrategy,
    create_continual_learning_engine
)


@dataclass
class BenchmarkConfig:
    """Configuration for continual learning benchmarks."""
    dataset_name: str
    num_tasks: int
    classes_per_task: int
    total_classes: int
    num_epochs_per_task: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_results: bool = True
    plot_results: bool = True


@dataclass
class BenchmarkResult:
    """Results from a continual learning benchmark run."""
    strategy: str
    config: BenchmarkConfig
    task_accuracies: List[List[float]]  # [task_id][evaluation_step] 
    final_accuracies: List[float]  # Final accuracy per task
    average_accuracy: float
    forgetting_measure: float
    forward_transfer: float
    backward_transfer: float
    total_parameters: int
    training_time: float
    memory_usage: float
    timestamp: str


class SimpleConvNet(nn.Module):
    """Simple CNN for CIFAR-100/MNIST benchmarks."""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ContinualLearningBenchmarkSuite:
    """Comprehensive benchmarking suite for continual learning research."""
    
    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategies to benchmark
        self.strategies = {
            "naive_finetuning": {"strategy": "ewc", "ewc_lambda": 0.0},
            "ewc": {"strategy": "ewc", "ewc_lambda": 10000.0},
            "experience_replay": {"strategy": "replay", "replay_buffer_size": 5000},
            "progressive_nets": {"strategy": "progressive", "use_progressive_nets": True},
            "adapters": {"strategy": "adapters", "use_adapters": True},
            "combined": {"strategy": "combined", "ewc_lambda": 5000.0, "replay_buffer_size": 2000, "use_adapters": True}
        }
        
        self.benchmark_configs = {
            "split_cifar100": BenchmarkConfig(
                dataset_name="split_cifar100",
                num_tasks=20,
                classes_per_task=5,
                total_classes=100,
                num_epochs_per_task=50,
                batch_size=128
            ),
            "split_mnist": BenchmarkConfig(
                dataset_name="split_mnist", 
                num_tasks=5,
                classes_per_task=2,
                total_classes=10,
                num_epochs_per_task=20,
                batch_size=256
            ),
            "permuted_mnist": BenchmarkConfig(
                dataset_name="permuted_mnist",
                num_tasks=10,
                classes_per_task=10,
                total_classes=10,
                num_epochs_per_task=20,
                batch_size=256
            )
        }
    
    def create_split_cifar100_tasks(self, config: BenchmarkConfig) -> Tuple[List[Task], List[DataLoader]]:
        """Create Split CIFAR-100 benchmark tasks."""
        print(f"📥 Loading CIFAR-100 dataset...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform
        )
        
        tasks = []
        dataloaders = []
        
        for task_id in range(config.num_tasks):
            # Select classes for this task
            start_class = task_id * config.classes_per_task
            end_class = start_class + config.classes_per_task
            task_classes = list(range(start_class, end_class))
            
            # Filter datasets
            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in task_classes]
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]
            
            train_subset = Subset(train_dataset, train_indices)
            test_subset = Subset(test_dataset, test_indices)
            
            train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False)
            
            task = Task(
                task_id=f"cifar100_task_{task_id}",
                task_name=f"CIFAR-100 Task {task_id} (Classes {start_class}-{end_class-1})",
                task_type=TaskType.CLASSIFICATION,
                input_dim=3*32*32,
                output_dim=config.classes_per_task,
                dataset_size=len(train_indices)
            )
            
            tasks.append(task)
            dataloaders.append((train_loader, test_loader))
        
        return tasks, dataloaders
    
    def create_split_mnist_tasks(self, config: BenchmarkConfig) -> Tuple[List[Task], List[DataLoader]]:
        """Create Split MNIST benchmark tasks."""
        print(f"📥 Loading MNIST dataset...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        tasks = []
        dataloaders = []
        
        for task_id in range(config.num_tasks):
            start_class = task_id * config.classes_per_task
            end_class = start_class + config.classes_per_task
            task_classes = list(range(start_class, end_class))
            
            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in task_classes]
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]
            
            train_subset = Subset(train_dataset, train_indices)
            test_subset = Subset(test_dataset, test_indices)
            
            train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False)
            
            task = Task(
                task_id=f"mnist_task_{task_id}",
                task_name=f"MNIST Task {task_id} (Classes {start_class}-{end_class-1})",
                task_type=TaskType.CLASSIFICATION,
                input_dim=1*28*28,
                output_dim=config.classes_per_task,
                dataset_size=len(train_indices)
            )
            
            tasks.append(task)
            dataloaders.append((train_loader, test_loader))
        
        return tasks, dataloaders
    
    def create_permuted_mnist_tasks(self, config: BenchmarkConfig) -> Tuple[List[Task], List[DataLoader]]:
        """Create Permuted MNIST benchmark tasks."""
        print(f"📥 Loading MNIST dataset for permutation...")
        
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=base_transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=base_transform
        )
        
        tasks = []
        dataloaders = []
        permutations = []
        
        for task_id in range(config.num_tasks):
            # Generate random permutation for this task
            if task_id == 0:
                # First task uses identity permutation
                perm = torch.arange(28*28)
            else:
                perm = torch.randperm(28*28)
            
            permutations.append(perm)
            
            # Create custom datasets with permutation
            class PermutedDataset(torch.utils.data.Dataset):
                def __init__(self, dataset, permutation):
                    self.dataset = dataset
                    self.permutation = permutation
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    img, label = self.dataset[idx]
                    img_flat = img.view(-1)
                    img_permuted = img_flat[self.permutation].view(1, 28, 28)
                    return img_permuted, label
            
            train_permuted = PermutedDataset(train_dataset, perm)
            test_permuted = PermutedDataset(test_dataset, perm)
            
            train_loader = DataLoader(train_permuted, batch_size=config.batch_size, shuffle=True)
            test_loader = DataLoader(test_permuted, batch_size=config.batch_size, shuffle=False)
            
            task = Task(
                task_id=f"pmnist_task_{task_id}",
                task_name=f"Permuted MNIST Task {task_id}",
                task_type=TaskType.CLASSIFICATION,
                input_dim=1*28*28,
                output_dim=10,
                dataset_size=len(train_dataset)
            )
            
            tasks.append(task)
            dataloaders.append((train_loader, test_loader))
        
        return tasks, dataloaders
    
    def evaluate_model(self, model: nn.Module, test_loaders: List[DataLoader], device: str) -> List[float]:
        """Evaluate model on all tasks."""
        model.eval()
        task_accuracies = []
        
        with torch.no_grad():
            for test_loader in test_loaders:
                correct = 0
                total = 0
                
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                
                accuracy = correct / total if total > 0 else 0.0
                task_accuracies.append(accuracy)
        
        return task_accuracies
    
    def run_benchmark(self, benchmark_name: str, strategy_name: str) -> BenchmarkResult:
        """Run a single benchmark with specified strategy."""
        print(f"\n🔬 Running {benchmark_name} with {strategy_name}")
        print(f"{'='*60}")
        
        config = self.benchmark_configs[benchmark_name]
        strategy_config = self.strategies[strategy_name]
        
        # Create tasks and dataloaders
        if benchmark_name == "split_cifar100":
            tasks, dataloaders = self.create_split_cifar100_tasks(config)
            model = SimpleConvNet(input_channels=3, num_classes=config.total_classes)
        elif benchmark_name == "split_mnist":
            tasks, dataloaders = self.create_split_mnist_tasks(config)
            model = SimpleConvNet(input_channels=1, num_classes=config.total_classes)
        elif benchmark_name == "permuted_mnist":
            tasks, dataloaders = self.create_permuted_mnist_tasks(config)
            model = SimpleConvNet(input_channels=1, num_classes=10)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        model = model.to(config.device)
        
        # Create continual learning engine
        engine = create_continual_learning_engine(**strategy_config)
        
        # Register all tasks
        for task in tasks:
            engine.register_task(task)
        
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Track results
        task_accuracies = []  # [task_id][evaluation_step]
        all_test_loaders = [dl[1] for dl in dataloaders]
        
        start_time = time.time()
        
        # Sequential task training
        for task_idx, (task, (train_loader, test_loader)) in enumerate(zip(tasks, dataloaders)):
            print(f"\n📚 Training on {task.task_name}")
            
            # Prepare for new task
            prep_info = engine.prepare_for_task(task, model, train_loader)
            print(f"   Preparation: {prep_info['components_activated']}")
            
            # Train on current task
            model.train()
            for epoch in range(config.num_epochs_per_task):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(config.device), target.to(config.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Add continual learning regularization (non-inplace)
                    train_info = engine.train_step(model, (data, target), optimizer, task)
                    if 'additional_loss' in train_info:
                        loss = loss + train_info['additional_loss']
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                if epoch % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Finish task training
            final_acc = self.evaluate_model(model, [test_loader], config.device)[0]
            finish_info = engine.finish_task_training(task, model, train_loader, final_acc)
            print(f"   Final accuracy: {final_acc:.4f}")
            print(f"   Finalization: {finish_info['components_finalized']}")
            
            # Evaluate on all tasks seen so far
            current_accuracies = self.evaluate_model(
                model, all_test_loaders[:task_idx+1], config.device
            )
            task_accuracies.append(current_accuracies)
            
            print(f"   All task accuracies: {[f'{acc:.3f}' for acc in current_accuracies]}")
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        final_accuracies = task_accuracies[-1] if task_accuracies else []
        average_accuracy = np.mean(final_accuracies) if final_accuracies else 0.0
        
        # Forgetting measure (average of max accuracy - final accuracy)
        forgetting_measures = []
        for i in range(len(final_accuracies)):
            max_acc = max([task_acc[i] for task_acc in task_accuracies if i < len(task_acc)])
            forgetting_measures.append(max_acc - final_accuracies[i])
        forgetting_measure = np.mean(forgetting_measures) if forgetting_measures else 0.0
        
        # Forward transfer (simplified)
        forward_transfer = 0.0  # Would need baseline random initialization
        
        # Backward transfer (average forgetting)
        backward_transfer = -forgetting_measure
        
        # Model info
        total_parameters = sum(p.numel() for p in model.parameters())
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        result = BenchmarkResult(
            strategy=strategy_name,
            config=config,
            task_accuracies=task_accuracies,
            final_accuracies=final_accuracies,
            average_accuracy=average_accuracy,
            forgetting_measure=forgetting_measure,
            forward_transfer=forward_transfer,
            backward_transfer=backward_transfer,
            total_parameters=total_parameters,
            training_time=training_time,
            memory_usage=memory_usage,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"\n✅ Benchmark completed!")
        print(f"   Average accuracy: {average_accuracy:.4f}")
        print(f"   Forgetting measure: {forgetting_measure:.4f}")
        print(f"   Training time: {training_time:.1f}s")
        print(f"   Parameters: {total_parameters:,}")
        
        return result
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Run all benchmarks with all strategies."""
        print(f"\n🚀 Starting comprehensive continual learning benchmarks")
        print(f"{'='*80}")
        print(f"Benchmarks: {list(self.benchmark_configs.keys())}")
        print(f"Strategies: {list(self.strategies.keys())}")
        
        all_results = {}
        
        for benchmark_name in self.benchmark_configs.keys():
            print(f"\n🎯 Benchmark Suite: {benchmark_name.upper()}")
            benchmark_results = {}
            
            for strategy_name in self.strategies.keys():
                try:
                    result = self.run_benchmark(benchmark_name, strategy_name)
                    benchmark_results[strategy_name] = result
                    
                    # Save individual result
                    if result.config.save_results:
                        self.save_result(result, benchmark_name)
                        
                except Exception as e:
                    print(f"❌ Error running {benchmark_name} with {strategy_name}: {e}")
                    continue
            
            all_results[benchmark_name] = benchmark_results
        
        # Generate comparison analysis
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def save_result(self, result: BenchmarkResult, benchmark_name: str):
        """Save individual benchmark result."""
        filename = f"{benchmark_name}_{result.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        # Convert result to dict for JSON serialization
        result_dict = asdict(result)
        result_dict['config'] = asdict(result.config)
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"💾 Saved results to {filepath}")
    
    def generate_comparison_report(self, all_results: Dict[str, Dict[str, BenchmarkResult]]):
        """Generate comprehensive comparison report for publication."""
        print(f"\n📊 Generating comparison report...")
        
        # Create summary tables
        summary_data = []
        
        for benchmark_name, benchmark_results in all_results.items():
            for strategy_name, result in benchmark_results.items():
                summary_data.append({
                    'Benchmark': benchmark_name,
                    'Strategy': strategy_name,
                    'Average Accuracy': result.average_accuracy,
                    'Forgetting Measure': result.forgetting_measure,
                    'Backward Transfer': result.backward_transfer,
                    'Parameters': result.total_parameters,
                    'Training Time (s)': result.training_time,
                    'Memory Usage (MB)': result.memory_usage
                })
        
        df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_path = self.results_dir / f"continual_learning_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved comparison table to {csv_path}")
        
        # Generate plots
        self.plot_comparison_results(all_results)
        
        # Print summary
        print(f"\n📋 COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        # Highlight best performers
        print(f"\n🏆 BEST PERFORMERS")
        print(f"{'='*40}")
        
        for benchmark in df['Benchmark'].unique():
            benchmark_df = df[df['Benchmark'] == benchmark]
            best_accuracy = benchmark_df.loc[benchmark_df['Average Accuracy'].idxmax()]
            best_forgetting = benchmark_df.loc[benchmark_df['Forgetting Measure'].idxmin()]
            
            print(f"\n{benchmark.upper()}:")
            print(f"  Best Accuracy: {best_accuracy['Strategy']} ({best_accuracy['Average Accuracy']:.4f})")
            print(f"  Best Forgetting: {best_forgetting['Strategy']} ({best_forgetting['Forgetting Measure']:.4f})")
    
    def plot_comparison_results(self, all_results: Dict[str, Dict[str, BenchmarkResult]]):
        """Generate publication-quality plots."""
        if not all_results:
            return
        
        # Set style for publication
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Continual Learning Benchmark Results', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        plot_data = []
        for benchmark_name, benchmark_results in all_results.items():
            for strategy_name, result in benchmark_results.items():
                plot_data.append({
                    'Benchmark': benchmark_name.replace('_', ' ').title(),
                    'Strategy': strategy_name.replace('_', ' ').title(),
                    'Average Accuracy': result.average_accuracy,
                    'Forgetting Measure': result.forgetting_measure,
                    'Training Time': result.training_time / 60,  # Convert to minutes
                    'Parameters (M)': result.total_parameters / 1e6
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot 1: Average Accuracy
        sns.barplot(data=plot_df, x='Benchmark', y='Average Accuracy', hue='Strategy', ax=axes[0,0])
        axes[0,0].set_title('Average Accuracy Across Tasks')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Forgetting Measure
        sns.barplot(data=plot_df, x='Benchmark', y='Forgetting Measure', hue='Strategy', ax=axes[0,1])
        axes[0,1].set_title('Catastrophic Forgetting (Lower is Better)')
        axes[0,1].set_ylabel('Forgetting Measure')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Training Time
        sns.barplot(data=plot_df, x='Benchmark', y='Training Time', hue='Strategy', ax=axes[1,0])
        axes[1,0].set_title('Training Time')
        axes[1,0].set_ylabel('Time (minutes)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Model Complexity
        sns.barplot(data=plot_df, x='Benchmark', y='Parameters (M)', hue='Strategy', ax=axes[1,1])
        axes[1,1].set_title('Model Complexity')
        axes[1,1].set_ylabel('Parameters (Millions)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"continual_learning_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path.with_suffix('.pdf'), bbox_inches='tight')
        
        print(f"📊 Saved comparison plots to {plot_path}")
        plt.show()


if __name__ == "__main__":
    # Run comprehensive benchmarks
    benchmark_suite = ContinualLearningBenchmarkSuite()
    
    print("🚀 Starting Continual Learning Benchmark Suite for Research Publication")
    print("This will run comprehensive comparisons for Phase 1 grant applications")
    
    # Run all benchmarks (this will take several hours)
    results = benchmark_suite.run_all_benchmarks()
    
    print("\n✅ Benchmark suite completed!")
    print("Results saved for NeurIPS/ICML/ICLR paper submission")