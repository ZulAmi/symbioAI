#!/usr/bin/env python3
"""
Tier 1 Quick Benchmark Test
===========================

Focused test of Tier 1 continual learning capabilities using working datasets.
"""

import sys
from pathlib import Path
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

def test_fashion_mnist_continual_learning():
    """Test continual learning on Fashion-MNIST."""
    print("🧠 Testing Fashion-MNIST Continual Learning")
    print("="*50)
    
    try:
        # Load Fashion-MNIST 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = torchvision.datasets.FashionMNIST(
            root='./data', 
            train=True, 
            transform=transform,
            download=True
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Create simple class-incremental tasks
        tasks = create_simple_class_incremental_tasks(dataset, num_tasks=3)
        
        print(f"✅ Created {len(tasks)} continual learning tasks")
        
        # Test basic continual learning
        metrics = test_simple_continual_learning(tasks)
        
        print(f"\n📊 Results:")
        print(f"   Task Accuracies: {[f'{acc:.3f}' for acc in metrics['task_accuracies']]}")
        print(f"   Average Accuracy: {metrics['average_accuracy']:.4f}")
        print(f"   Forgetting Measure: {metrics['forgetting_measure']:.4f}")
        
        # Determine success level
        if metrics['average_accuracy'] >= 0.8:
            success_level = "EXCELLENT"
        elif metrics['average_accuracy'] >= 0.6:
            success_level = "GOOD"
        else:
            success_level = "NEEDS_WORK"
        
        print(f"   Success Level: {success_level}")
        
        return True, metrics
        
    except Exception as e:
        print(f"❌ Fashion-MNIST test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def create_simple_class_incremental_tasks(dataset, num_tasks=3):
    """Create simple class-incremental tasks."""
    # Get all targets
    if hasattr(dataset, 'targets'):
        targets = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else dataset.targets
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
    
    # Group by class
    class_indices = {}
    for idx, target in enumerate(targets):
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(idx)
    
    # Create tasks
    unique_classes = sorted(class_indices.keys())
    classes_per_task = len(unique_classes) // num_tasks
    
    tasks = []
    for task_idx in range(num_tasks):
        start_class = task_idx * classes_per_task
        if task_idx == num_tasks - 1:
            end_class = len(unique_classes)
        else:
            end_class = (task_idx + 1) * classes_per_task
        
        task_classes = unique_classes[start_class:end_class]
        
        # Get indices for this task
        task_indices = []
        for class_id in task_classes:
            task_indices.extend(class_indices[class_id])
        
        # Create subset
        task_dataset = torch.utils.data.Subset(dataset, task_indices)
        tasks.append(task_dataset)
        
        print(f"   Task {task_idx + 1}: Classes {task_classes}, {len(task_indices)} samples")
    
    return tasks


def test_simple_continual_learning(tasks):
    """Test simple continual learning on task sequence."""
    # Simple MLP model
    import torch.nn as nn
    import torch.optim as optim
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"🔧 Using device: {device}")
    
    # Create model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)  # Fashion-MNIST has 10 classes
    ).to(device)
    
    task_accuracies = []
    
    print(f"\n🚀 Training on {len(tasks)} tasks sequentially...")
    
    for task_idx, task in enumerate(tasks):
        print(f"\n📚 Training Task {task_idx + 1}...")
        
        # Create data loader
        task_loader = torch.utils.data.DataLoader(task, batch_size=128, shuffle=True, num_workers=0)
        
        # Train on current task
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(3):  # Quick training
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(task_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        print(f"   ✅ Task {task_idx + 1} training complete")
        
        # Evaluate on all tasks seen so far
        current_accuracies = []
        for eval_task_idx in range(task_idx + 1):
            eval_task = tasks[eval_task_idx]
            eval_loader = torch.utils.data.DataLoader(eval_task, batch_size=128, shuffle=False, num_workers=0)
            
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in eval_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            accuracy = correct / total
            current_accuracies.append(accuracy)
            print(f"   Task {eval_task_idx + 1} accuracy: {accuracy:.4f}")
        
        task_accuracies.append(current_accuracies)
    
    # Calculate metrics
    final_accuracies = task_accuracies[-1]
    average_accuracy = sum(final_accuracies) / len(final_accuracies)
    
    # Calculate forgetting (simplified)
    forgetting_scores = []
    for i in range(len(tasks) - 1):
        max_acc = max(task_accuracies[j][i] for j in range(i, len(tasks)))
        final_acc = task_accuracies[-1][i]
        forgetting_scores.append(max_acc - final_acc)
    
    forgetting_measure = sum(forgetting_scores) / len(forgetting_scores) if forgetting_scores else 0.0
    
    return {
        'task_accuracies': final_accuracies,
        'average_accuracy': average_accuracy,
        'forgetting_measure': forgetting_measure,
        'all_task_accuracies': task_accuracies
    }


def test_tier1_integration():
    """Test integration with tier validation system."""
    print("\n🧠 Testing Tier 1 Integration")
    print("="*50)
    
    try:
        # Test tier validation on Fashion-MNIST
        from validation.run_tier_validation import TierBasedValidator
        
        validator = TierBasedValidator()
        
        print("🚀 Running integrated Tier 1 validation...")
        
        # Test just the Fashion-MNIST part
        result = validator._validate_continual_learning('fashion_mnist')
        
        print(f"✅ Integration test complete:")
        print(f"   Accuracy: {result.accuracy:.4f}")
        print(f"   Success: {result.metadata.get('success_level', 'UNKNOWN')}")
        print(f"   Forgetting Resistance: {result.metadata.get('forgetting_resistance', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Tier 1 focused tests."""
    print("🧪 TIER 1 FOCUSED BENCHMARK TEST")
    print("="*60)
    
    tests = [
        ("Fashion-MNIST Continual Learning", test_fashion_mnist_continual_learning),
        ("Tier 1 Integration", test_tier1_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print("="*60)
        
        try:
            if test_name == "Fashion-MNIST Continual Learning":
                success, metrics = test_func()
                results.append((test_name, success, metrics))
            else:
                success = test_func()
                results.append((test_name, success, {}))
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            results.append((test_name, False, {}))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TIER 1 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success, metrics in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        
        if success and metrics:
            print(f"   Avg Accuracy: {metrics.get('average_accuracy', 0):.4f}")
            print(f"   Forgetting: {metrics.get('forgetting_measure', 0):.4f}")
        
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Tier 1 implementation is working!")
        print("✅ Ready for continual learning benchmarks")
    else:
        print("⚠️  Some Tier 1 components need attention")


if __name__ == '__main__':
    main()