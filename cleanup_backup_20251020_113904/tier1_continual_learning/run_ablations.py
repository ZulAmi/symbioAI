#!/usr/bin/env python3
"""
Ablation Studies for Causal-DER

Tests individual components:
1. DER++ baseline (random sampling)
2. DER++ + causal importance (no diversity)
3. DER++ + diversity (no causal)
4. Full Causal-DER (causal + diversity)
"""

import sys
from pathlib import Path
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

from training.der_plus_plus import DERPlusPlusEngine
from training.causal_der import CausalDEREngine
from validation.tier1_continual_learning.run_clean_der_plus_plus import (
    SimpleResNet18, create_split_cifar100, compute_accuracy
)


def run_ablation(engine, engine_name, seed=42, data_root='../../data'):
    """Run single ablation experiment."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')
    
    print(f"Running {engine_name} on {device}")
    
    # Model
    model = SimpleResNet18(num_classes=100).to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    
    # Data
    tasks_train, tasks_test = create_split_cifar100(num_tasks=5, data_root=data_root)
    
    # Training
    all_task_accuracies = []
    
    for task_id in range(5):
        print(f"\n  Task {task_id + 1}/5:")
        
        train_loader = DataLoader(tasks_train[task_id], 
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=0)
        
        # Train for 50 epochs
        for epoch in range(50):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss, _ = engine.compute_loss(model, data, target, output, task_id)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    logits = model(data)
                engine.store(data, target, logits, task_id)
        
        # Evaluate
        task_accs = []
        for eval_task_id in range(task_id + 1):
            test_loader = DataLoader(tasks_test[eval_task_id],
                                    batch_size=128,
                                    shuffle=False,
                                    num_workers=0)
            acc = compute_accuracy(model, test_loader, device)
            task_accs.append(acc)
        
        all_task_accuracies.append(task_accs)
        print(f"    Avg Acc: {np.mean(task_accs):.4f}")
    
    # Metrics
    num_tasks = 5
    avg_accuracy = np.mean([
        all_task_accuracies[i][j]
        for i in range(num_tasks)
        for j in range(i+1)
    ])
    
    forgetting = 0.0
    for i in range(num_tasks - 1):
        max_acc = max([all_task_accuracies[j][i] for j in range(i, num_tasks)])
        final_acc = all_task_accuracies[-1][i]
        forgetting += max(0, max_acc - final_acc)
    forgetting /= max(1, num_tasks - 1)
    
    return {
        'method': engine_name,
        'average_accuracy': float(avg_accuracy),
        'forgetting_measure': float(forgetting),
        'task_accuracies': [[float(x) for x in row] for row in all_task_accuracies]
    }


def main():
    print("="*60)
    print("ABLATION STUDY: Causal-DER Components")
    print("="*60)
    
    seed = 42
    
    # 1. DER++ baseline (random sampling)
    print("\n[1/2] Running DER++ Baseline (random sampling)...")
    der_engine = DERPlusPlusEngine(alpha=0.5, buffer_size=2000)
    der_results = run_ablation(der_engine, "DER++", seed=seed)
    
    # 2. Causal-DER (causal importance sampling)
    print("\n[2/2] Running Causal-DER (causal importance sampling)...")
    causal_engine = CausalDEREngine(alpha=0.5, buffer_size=2000)
    causal_results = run_ablation(causal_engine, "Causal-DER", seed=seed)
    
    # Summary
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    
    methods = [der_results, causal_results]
    
    for result in methods:
        print(f"\n{result['method']}:")
        print(f"  Average Accuracy: {result['average_accuracy']:.4f} ({result['average_accuracy']*100:.2f}%)")
        print(f"  Forgetting:       {result['forgetting_measure']:.4f} ({result['forgetting_measure']*100:.2f}%)")
    
    # Improvement
    improvement = (causal_results['average_accuracy'] - der_results['average_accuracy']) * 100
    print(f"\n{'='*60}")
    print(f"Causal-DER Improvement: {improvement:+.2f}%")
    print(f"{'='*60}")
    
    # Save
    results_dir = Path(__file__).parent / 'validation' / 'results'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    ablation_results = {
        'ablations': methods,
        'improvement': float(improvement),
        'seed': seed
    }
    
    with open(results_dir / 'ablation_study.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir / 'ablation_study.json'}")


if __name__ == "__main__":
    main()
