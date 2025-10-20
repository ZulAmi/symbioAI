#!/usr/bin/env python3
"""
Causal-DER Benchmark

Runs Causal-DER with exact same setup as DER++ for fair comparison.
Only difference: causal importance weighting in buffer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from training.causal_der import CausalDEREngine


class SimpleResNet18(nn.Module):
    """Simple ResNet-18 for CIFAR-100."""
    
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def create_split_cifar100(num_tasks=10, data_root='../../data'):
    """Create Split CIFAR-100 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                           (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=False, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=False, transform=transform_test
    )
    
    classes_per_task = 100 // num_tasks
    tasks_train = []
    tasks_test = []
    
    # Use label lists directly to avoid iterating through transformed samples
    train_targets = train_dataset.targets if hasattr(train_dataset, 'targets') else [y for _, y in train_dataset]
    test_targets = test_dataset.targets if hasattr(test_dataset, 'targets') else [y for _, y in test_dataset]

    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task

        task_indices = [i for i, label in enumerate(train_targets)
                        if start_class <= label < end_class]
        tasks_train.append(Subset(train_dataset, task_indices))

        task_indices = [i for i, label in enumerate(test_targets)
                        if start_class <= label < end_class]
        tasks_test.append(Subset(test_dataset, task_indices))
    
    return tasks_train, tasks_test


def compute_accuracy(model, dataloader, device):
    """Compute accuracy on dataloader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    model.train()
    return correct / total if total > 0 else 0.0


def run_causal_der_benchmark(seed=42, data_root='../../data', num_tasks=10, epochs_per_task=50):
    """
    Run Causal-DER benchmark.
    
    Exact same hyperparameters as DER++:
    - Learning rate: 0.03
    - Optimizer: SGD with momentum 0.9
    - Batch size: 32
    - Epochs per task: 50
    - Buffer size: 2000
    - Alpha (distillation): 0.5
    
    Only difference: Causal importance weighting in buffer
    """
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')
    
    print(f"Running Causal-DER on {device}")
    print(f"Seed: {seed}")
    print(f"=" * 60)
    
    # Create model
    model = SimpleResNet18(num_classes=100).to(device)
    
    # Causal-DER engine (same alpha/beta and buffer as DER++)
    causal_der_engine = CausalDEREngine(alpha=0.5, beta=0.5, buffer_size=2000)
    
    # Optimizer (same as DER++)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.03,
        momentum=0.9,
        weight_decay=0.0
    )
    
    # Data
    print("Loading CIFAR-100...")
    tasks_train, tasks_test = create_split_cifar100(num_tasks=num_tasks, data_root=data_root)
    classes_per_task = 100 // num_tasks
    print(f"Created {num_tasks} tasks with {classes_per_task} classes/task")
    
    # Training
    all_task_accuracies = []
    task_times = []
    accmean_per_task = []
    
    for task_id in range(num_tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_id + 1}/{num_tasks} (Classes {task_id*classes_per_task}-{(task_id+1)*classes_per_task-1})")
        print(f"{'='*60}")
        
        task_start = time.time()
        
        # Data loaders
        train_loader = DataLoader(tasks_train[task_id], 
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=0)
        
        # Train for epochs_per_task epochs
        for epoch in range(epochs_per_task):
            epoch_loss = 0.0
            epoch_current = 0.0
            epoch_replay_mse = 0.0
            epoch_replay_ce = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Forward
                output = model(data)
                
                # Causal-DER loss (includes causal importance estimation)
                loss, info = causal_der_engine.compute_loss(
                    model, data, target, output, task_id
                )
                
                # Backward
                loss.backward()
                optimizer.step()
                
                # Store in buffer with causal importance
                with torch.no_grad():
                    logits = model(data)
                causal_der_engine.store(data, target, logits, task_id)
                
                epoch_loss += loss.item()
                epoch_current += info['current_loss']
                epoch_replay_mse += info.get('replay_mse', 0.0)
                epoch_replay_ce += info.get('replay_ce', 0.0)
            
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs_per_task:
                avg_loss = epoch_loss / len(train_loader)
                avg_current = epoch_current / len(train_loader)
                avg_replay_mse = epoch_replay_mse / len(train_loader)
                avg_replay_ce = epoch_replay_ce / len(train_loader)
                print(f"  Epoch {epoch+1:2d}/{epochs_per_task} - Loss: {avg_loss:.4f} "
                      f"(Current: {avg_current:.4f}, ReplayMSE: {avg_replay_mse:.4f}, ReplayCE: {avg_replay_ce:.4f})")
        
        task_time = time.time() - task_start
        task_times.append(task_time)
        
        # Evaluate on ALL tasks seen so far
        print(f"\n  Evaluating on tasks 1-{task_id+1}:")
        task_accs = []
        for eval_task_id in range(task_id + 1):
            test_loader = DataLoader(tasks_test[eval_task_id],
                                    batch_size=128,
                                    shuffle=False,
                                    num_workers=0)
            acc = compute_accuracy(model, test_loader, device)
            task_accs.append(acc)
            print(f"    Task {eval_task_id + 1}: {acc:.4f} ({acc*100:.2f}%)")
        
        all_task_accuracies.append(task_accs)
        print(f"  Task {task_id+1} completed in {task_time:.2f}s")
        
        # Show buffer stats (including causal importance info)
        stats = causal_der_engine.get_statistics()
        buf = stats.get('buffer', {})
        print(f"  Buffer: {buf.get('size', 0)}/{buf.get('capacity', 0)} samples")
        print(f"  Avg Causal Importance: {buf.get('avg_causal_importance', 0.0):.4f}")
        # AccMean up to this task (class-IL mean over seen tasks)
        accmean = float(np.mean(task_accs)) if task_accs else 0.0
        accmean_per_task.append(accmean)
        print(f"  AccMean@Task{task_id+1}: {accmean*100:.2f}%")
    
    # Compute standard metrics
    
    # Average Accuracy
    avg_accuracy = np.mean([
        all_task_accuracies[i][j]
        for i in range(num_tasks)
        for j in range(i+1)
    ])
    
    # Final Accuracy
    final_accuracy = np.mean(all_task_accuracies[-1])
    
    # Forgetting Measure
    forgetting = 0.0
    for i in range(num_tasks - 1):
        max_acc = max([all_task_accuracies[j][i] for j in range(i, num_tasks)])
        final_acc = all_task_accuracies[-1][i]
        forgetting += max(0, max_acc - final_acc)
    forgetting /= max(1, num_tasks - 1)
    
    # Forward Transfer
    forward_transfer = np.mean([
        all_task_accuracies[i][i]
        for i in range(num_tasks)
    ])
    
    results = {
        'method': 'Causal-DER',
        'seed': seed,
        'average_accuracy': float(avg_accuracy),
        'final_accuracy': float(final_accuracy),
        'forgetting_measure': float(forgetting),
        'forward_transfer': float(forward_transfer),
        'task_accuracies': [[float(x) for x in row] for row in all_task_accuracies],
        'task_times': [float(x) for x in task_times],
        'total_time': float(sum(task_times)),
        'buffer_stats': causal_der_engine.get_statistics(),
        'alpha': 0.5,
        'beta': 0.5,
        'num_tasks': num_tasks,
        'epochs_per_task': epochs_per_task,
        'accmean_per_task': accmean_per_task
    }
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS (STANDARD METRICS)")
    print(f"{'='*60}")
    print(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"Final Accuracy:   {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Forgetting:       {forgetting:.4f} ({forgetting*100:.2f}%)")
    print(f"Forward Transfer: {forward_transfer:.4f} ({forward_transfer*100:.2f}%)")
    print(f"Total Time:       {sum(task_times):.2f}s")
    
    # Save results
    # Save next to validation/results folder at repo root for consistency
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_file = results_dir / f'causal_der_seed{seed}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Causal-DER benchmark')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--data_root', type=str, default='../../data', help='Data directory')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks (CIFAR-100 uses 10 for class-IL)')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per task')
    args = parser.parse_args()
    
    all_results = []
    for run in range(args.num_runs):
        seed = args.seed + run
        print(f"\n\n{'#'*60}")
        print(f"RUN {run+1}/{args.num_runs} - Seed {seed}")
        print(f"{'#'*60}\n")
        results = run_causal_der_benchmark(seed=seed, data_root=args.data_root, num_tasks=args.num_tasks, epochs_per_task=args.epochs)
        all_results.append(results)
    
    if len(all_results) > 1:
        # Aggregate results
        avg_acc = np.mean([r['average_accuracy'] for r in all_results])
        std_acc = np.std([r['average_accuracy'] for r in all_results])
        avg_forg = np.mean([r['forgetting_measure'] for r in all_results])
        std_forg = np.std([r['forgetting_measure'] for r in all_results])
        
        print(f"\n\n{'='*60}")
        print(f"AGGREGATE RESULTS ({len(all_results)} runs)")
        print(f"{'='*60}")
        print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f} ({avg_acc*100:.2f}% ± {std_acc*100:.2f}%)")
        print(f"Forgetting:       {avg_forg:.4f} ± {std_forg:.4f} ({avg_forg*100:.2f}% ± {std_forg*100:.2f}%)")
