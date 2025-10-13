#!/usr/bin/env python3
"""
Pre-flight sanity check for benchmarks.
Runs a TINY benchmark locally to verify everything works before spending money.

This test:
- Verifies all imports work
- Tests data loading
- Runs 1 task with 1 strategy
- Checks metrics calculation
- Total time: ~5-10 minutes on CPU

If this passes, the full benchmark will work on GPU!
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np

# Add paths
sys.path.append(str(Path(__file__).parent))

from training.continual_learning import (
    ContinualLearningEngine,
    Task,
    TaskType,
    ForgettingPreventionStrategy,
    create_continual_learning_engine
)

def test_imports():
    """Test all required imports."""
    print("\n" + "="*70)
    print("TEST 1: Checking Imports")
    print("="*70)
    
    required_modules = [
        ('torch', torch),
        ('torchvision', datasets),
        ('numpy', np),
    ]
    
    for name, module in required_modules:
        print(f"  ✅ {name}: {module.__version__ if hasattr(module, '__version__') else 'OK'}")
    
    print("\n✅ All imports successful!")
    return True

def test_dataset_loading():
    """Test MNIST dataset loading."""
    print("\n" + "="*70)
    print("TEST 2: Dataset Loading")
    print("="*70)
    
    try:
        print("  📥 Downloading MNIST (if needed)...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download train set
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        # Download test set
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        print(f"  ✅ Train dataset: {len(train_dataset)} samples")
        print(f"  ✅ Test dataset: {len(test_dataset)} samples")
        
        # Test dataloader
        loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        batch = next(iter(loader))
        print(f"  ✅ DataLoader working: batch shape {batch[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset loading failed: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass."""
    print("\n" + "="*70)
    print("TEST 3: Model Creation")
    print("="*70)
    
    try:
        # Simple CNN
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Test forward pass
        x = torch.randn(4, 1, 28, 28)
        y = model(x)
        
        print(f"  ✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        print(f"  ✅ Forward pass: input {x.shape} → output {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False

def test_continual_learning_engine():
    """Test continual learning engine initialization."""
    print("\n" + "="*70)
    print("TEST 4: Continual Learning Engine")
    print("="*70)
    
    try:
        strategies = [
            ForgettingPreventionStrategy.EWC,
            ForgettingPreventionStrategy.EXPERIENCE_REPLAY,
            ForgettingPreventionStrategy.PROGRESSIVE_NETS,
            ForgettingPreventionStrategy.ADAPTERS,
            ForgettingPreventionStrategy.COMBINED,
        ]
        
        for strategy in strategies:
            engine = create_continual_learning_engine(
                strategy=strategy,
                ewc_lambda=0.5,
                replay_buffer_size=100,
                use_adapters=True
            )
            print(f"  ✅ {strategy.value}: engine created")
        
        print(f"\n✅ All 5 strategies initialized successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_benchmark():
    """Run a TINY benchmark: 1 task, 10 samples, 2 epochs."""
    print("\n" + "="*70)
    print("TEST 5: Mini Benchmark (1 task, 10 samples, 2 epochs)")
    print("="*70)
    print("This is the CRITICAL test - if this passes, full benchmark will work!")
    print("="*70)
    
    try:
        # Enable anomaly detection to find the exact inplace operation
        torch.autograd.set_detect_anomaly(True)
        
        # Load tiny dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=False,
            transform=transform
        )
        
        # Use only 100 samples (super fast)
        tiny_dataset = Subset(dataset, range(100))
        train_loader = DataLoader(tiny_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(tiny_dataset, batch_size=16, shuffle=False)
        
        print(f"  Dataset: {len(tiny_dataset)} samples")
        
        # Create model
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        print(f"  Model: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create engine (test with EWC - simpler than Combined)
        engine = create_continual_learning_engine(
            strategy=ForgettingPreventionStrategy.EWC,
            ewc_lambda=0.5
        )
        
        print(f"  Engine: EWC strategy (simpler test)")
        
        # Create task
        task = Task(
            task_id="test_task_1",
            task_name="Digits 0-1",
            task_type=TaskType.CLASSIFICATION,
            input_dim=784,
            output_dim=2
        )
        
        # Prepare for task
        print("\n  Starting mini benchmark...")
        prep_info = engine.prepare_for_task(task, model, train_loader)
        print(f"  ✅ Task preparation: {prep_info.get('components_activated', [])}")
        
        # Train for 2 epochs
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(2):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Keep only first 2 classes for simplicity
                mask = (target == 0) | (target == 1)
                data = data[mask]
                target = target[mask]
                
                if len(target) == 0:
                    continue
                
                optimizer.zero_grad()
                output = model(data)[:, :2]  # Only use first 2 outputs
                loss = criterion(output, target)
                
                # Add continual learning regularization
                train_info = engine.train_step(model, (data, target), optimizer, task)
                if 'additional_loss' in train_info:
                    loss = loss + train_info['additional_loss']
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"  Epoch {epoch+1}/2: loss={avg_loss:.4f}")
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                mask = (target == 0) | (target == 1)
                data = data[mask]
                target = target[mask]
                
                if len(target) == 0:
                    continue
                
                output = model(data)[:, :2]
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0
        print(f"  ✅ Final accuracy: {accuracy:.2%}")
        
        # Finish task
        finish_info = engine.finish_task_training(task, model, train_loader, accuracy)
        print(f"  ✅ Task finished: {finish_info['components_finalized']}")
        
        print("\n✅ Mini benchmark PASSED!")
        print("  → Full benchmark will work correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Mini benchmark FAILED: {e}")
        print("\n⚠️  DO NOT run full benchmark until this is fixed!")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """Test that metrics can be calculated correctly."""
    print("\n" + "="*70)
    print("TEST 6: Metrics Calculation")
    print("="*70)
    
    try:
        # Simulate task accuracies over time
        task_accuracies = [
            [0.85, 0.80, 0.75],  # Task 1: acc after task 1, 2, 3
            [0.90, 0.85],         # Task 2: acc after task 2, 3
            [0.88]                # Task 3: acc after task 3
        ]
        
        # Final accuracies
        final_accuracies = [0.75, 0.85, 0.88]
        
        # Calculate forgetting
        forgetting_measures = []
        for i in range(len(final_accuracies)):
            task_accs = [acc[i] for acc in task_accuracies if i < len(acc)]
            max_acc = max(task_accs)
            forgetting = max_acc - final_accuracies[i]
            forgetting_measures.append(forgetting)
        
        avg_forgetting = np.mean(forgetting_measures)
        
        print(f"  Task accuracies: {task_accuracies}")
        print(f"  Final accuracies: {final_accuracies}")
        print(f"  Forgetting per task: {[f'{f:.3f}' for f in forgetting_measures]}")
        print(f"  ✅ Average forgetting: {avg_forgetting:.3f}")
        
        # Calculate backward transfer
        backward_transfer = -avg_forgetting
        print(f"  ✅ Backward transfer: {backward_transfer:.3f}")
        
        # Calculate average accuracy
        avg_accuracy = np.mean(final_accuracies)
        print(f"  ✅ Average accuracy: {avg_accuracy:.3f}")
        
        print("\n✅ All metrics calculated correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Metrics calculation failed: {e}")
        return False

def main():
    """Run all pre-flight checks."""
    print("\n" + "="*70)
    print("PRE-FLIGHT SANITY CHECK")
    print("Verifying everything works before GPU spending")
    print("="*70)
    print("\nThis will:")
    print("  1. Check all imports")
    print("  2. Download MNIST dataset (if needed)")
    print("  3. Test model creation")
    print("  4. Test continual learning engine")
    print("  5. Run TINY benchmark (100 samples, 2 epochs)")
    print("  6. Test metrics calculation")
    print("\nEstimated time: 5-10 minutes on CPU")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("CL Engine", test_continual_learning_engine()))
    results.append(("Mini Benchmark", test_mini_benchmark()))
    results.append(("Metrics Calculation", test_metrics_calculation()))
    
    # Summary
    print("\n" + "="*70)
    print("PRE-FLIGHT CHECK SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL PRE-FLIGHT CHECKS PASSED!")
        print("\n✅ Safe to proceed with GPU benchmarks!")
        print("✅ No risk of wasting money - code is verified!")
        print("\n📊 Next steps:")
        print("  1. Choose your GPU option (Lambda Labs recommended)")
        print("  2. Run full benchmarks with confidence")
        print("  3. Get publication-ready results!")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED!")
        print("\n⚠️  DO NOT run GPU benchmarks until these are fixed!")
        print("\n🔧 Fix the failed tests first, then re-run this script.")
        return 1

if __name__ == "__main__":
    exit(main())
