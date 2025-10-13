#!/usr/bin/env python3
"""
Simple Working Benchmark - Bypasses the in-place operation issues
This runs real continual learning benchmarks with actual data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import time
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader, Subset

class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def create_task_data(dataset, task_classes, samples_per_class=500):
    """Create task-specific dataset"""
    indices = []
    for class_id in task_classes:
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]
        indices.extend(class_indices[:samples_per_class])
    
    return Subset(dataset, indices)

def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total

def run_ewc_training(model, task_loader, device, epochs=5, lr=0.001, ewc_lambda=1000):
    """Simple EWC training without problematic operations"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Store previous parameters for EWC (simplified)
    prev_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(task_loader):
            if batch_idx > 20:  # Limit batches for speed
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Standard cross-entropy loss
            ce_loss = criterion(output, target)
            
            # Simple EWC regularization (without Fisher Information for now)
            ewc_loss = 0
            if len(prev_params) > 0:
                for name, param in model.named_parameters():
                    if name in prev_params:
                        ewc_loss += ewc_lambda * ((param - prev_params[name]) ** 2).sum()
            
            total_loss = ce_loss + ewc_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_count += 1
        
        if epoch % 2 == 0:
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return model

def run_experience_replay_training(model, task_loader, device, epochs=5, lr=0.001, replay_buffer=None):
    """Simple Experience Replay training"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create simple replay buffer
    if replay_buffer is None:
        replay_buffer = []
    
    # Add current task data to replay buffer
    for data, target in task_loader:
        if len(replay_buffer) < 1000:  # Limit buffer size
            replay_buffer.extend(list(zip(data, target)))
        break  # Just take first batch for replay
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(task_loader):
            if batch_idx > 20:  # Limit batches for speed
                break
                
            data, target = data.to(device), target.to(device)
            
            # Mix current task with replay data
            if replay_buffer and len(replay_buffer) > 16:
                replay_samples = np.random.choice(len(replay_buffer), 16, replace=False)
                replay_data = [replay_buffer[i] for i in replay_samples]
                replay_x = torch.stack([x for x, y in replay_data]).to(device)
                replay_y = torch.stack([y for x, y in replay_data]).to(device)
                
                # Combine current and replay data
                combined_data = torch.cat([data[:16], replay_x], dim=0)
                combined_target = torch.cat([target[:16], replay_y], dim=0)
            else:
                combined_data = data
                combined_target = target
            
            optimizer.zero_grad()
            output = model(combined_data)
            loss = criterion(output, combined_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        if epoch % 2 == 0:
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return model, replay_buffer

def run_naive_finetuning(model, task_loader, device, epochs=5, lr=0.001):
    """Simple fine-tuning without continual learning"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(task_loader):
            if batch_idx > 20:  # Limit batches for speed
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        if epoch % 2 == 0:
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return model

def run_benchmarks():
    print("🚀 REAL CONTINUAL LEARNING BENCHMARKS")
    print("================================================================================")
    
    device = torch.device('cpu')  # Use CPU for Mac
    print(f"Device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Define tasks for Split MNIST (2 classes per task)
    tasks = [
        [0, 1],  # Task 0: digits 0, 1
        [2, 3],  # Task 1: digits 2, 3
        [4, 5],  # Task 2: digits 4, 5
        [6, 7],  # Task 3: digits 6, 7
        [8, 9],  # Task 4: digits 8, 9
    ]
    
    strategies = {
        'naive_finetuning': run_naive_finetuning,
        'ewc': run_ewc_training,
        'experience_replay': run_experience_replay_training,
    }
    
    results = {
        'mode': 'real_local',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'device': str(device),
        'config': {
            'num_tasks': len(tasks),
            'epochs_per_task': 5,
            'strategies': list(strategies.keys())
        },
        'results': {}
    }
    
    for strategy_name, strategy_fn in strategies.items():
        print(f"\n🔬 RUNNING STRATEGY: {strategy_name.upper()}")
        print("=" * 80)
        
        # Create fresh model for each strategy
        model = SimpleModel().to(device)
        replay_buffer = None
        task_accuracies = []
        all_task_loaders = []
        
        start_time = time.time()
        
        # Train on each task sequentially
        for task_id, task_classes in enumerate(tasks):
            print(f"\n📚 Task {task_id}: Classes {task_classes}")
            
            # Create task data
            task_data = create_task_data(train_dataset, task_classes, samples_per_class=300)
            task_loader = DataLoader(task_data, batch_size=64, shuffle=True)
            all_task_loaders.append(task_loader)
            
            # Train on this task
            if strategy_name == 'ewc':
                model = strategy_fn(model, task_loader, device)
            elif strategy_name == 'experience_replay':
                model, replay_buffer = strategy_fn(model, task_loader, device, replay_buffer=replay_buffer)
            else:  # naive_finetuning
                model = strategy_fn(model, task_loader, device)
            
            # Evaluate on all tasks seen so far
            current_task_accuracies = []
            for eval_task_id in range(task_id + 1):
                eval_classes = tasks[eval_task_id]
                eval_data = create_task_data(test_dataset, eval_classes, samples_per_class=100)
                eval_loader = DataLoader(eval_data, batch_size=64, shuffle=False)
                
                accuracy = evaluate_model(model, eval_loader, device)
                current_task_accuracies.append(accuracy)
                print(f"  Task {eval_task_id} accuracy: {accuracy:.3f}")
            
            task_accuracies.append(current_task_accuracies.copy())
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        final_accuracies = task_accuracies[-1]  # Accuracies after training on all tasks
        avg_accuracy = sum(final_accuracies) / len(final_accuracies)
        
        # Calculate forgetting (simplified)
        forgetting_scores = []
        for task_id in range(len(tasks) - 1):
            initial_acc = task_accuracies[task_id][task_id]  # Accuracy right after training on task
            final_acc = final_accuracies[task_id]  # Accuracy after training on all tasks
            forgetting = max(0, initial_acc - final_acc)
            forgetting_scores.append(forgetting)
        
        avg_forgetting = sum(forgetting_scores) / len(forgetting_scores) if forgetting_scores else 0
        
        results['results'][strategy_name] = {
            'average_accuracy': avg_accuracy,
            'forgetting': avg_forgetting,
            'backward_transfer': -avg_forgetting,  # Simplified metric
            'final_accuracies': final_accuracies,
            'task_accuracies': task_accuracies,
            'training_time': training_time,
            'status': 'completed'
        }
        
        print(f"\n✅ {strategy_name.upper()} COMPLETED!")
        print(f"   Average Accuracy: {avg_accuracy:.3f}")
        print(f"   Average Forgetting: {avg_forgetting:.3f}")
        print(f"   Training Time: {training_time:.1f} seconds")
    
    # Save results
    results_file = f'experiments/results/real_benchmark_{results["timestamp"]}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n🎉 ALL BENCHMARKS COMPLETED!")
    print("=" * 80)
    print("📊 FINAL RESULTS:")
    for strategy, result in results['results'].items():
        print(f"  {strategy:20} | Acc: {result['average_accuracy']:.3f} | Forgetting: {result['forgetting']:.3f}")
    
    print(f"\n💾 Results saved to: {results_file}")
    return results_file

if __name__ == "__main__":
    run_benchmarks()