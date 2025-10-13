#!/usr/bin/env python3
"""
Quick test to verify inplace operation fixes.
Tests Fisher Information computation and basic benchmark flow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))

from training.continual_learning import (
    ContinualLearningEngine,
    Task,
    TaskType,
    ForgettingPreventionStrategy,
    create_continual_learning_engine
)

def create_simple_model():
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

def create_simple_dataset(num_samples=100):
    """Create a simple random dataset."""
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)

def test_fisher_computation():
    """Test Fisher Information computation (was causing inplace errors)."""
    print("\n" + "="*60)
    print("TEST 1: Fisher Information Computation")
    print("="*60)
    
    try:
        # Create model and data
        model = create_simple_model()
        dataset = create_simple_dataset(100)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Create engine with EWC
        engine = create_continual_learning_engine(
            strategy=ForgettingPreventionStrategy.EWC,
            ewc_lambda=0.5
        )
        
        # Create task
        task = Task(
            task_id="test_task_1",
            task_name="Test Task 1",
            task_type=TaskType.CLASSIFICATION,
            input_dim=10,
            output_dim=2
        )
        
        # Compute Fisher (this was causing inplace errors)
        print("Computing Fisher Information...")
        fisher = engine.compute_fisher_information(
            model=model,
            task=task,
            dataloader=dataloader,
            num_samples=50
        )
        
        print(f"✅ Fisher computation successful!")
        print(f"   Task ID: {fisher.task_id}")
        print(f"   Samples used: {fisher.num_samples}")
        print(f"   Parameters tracked: {len(fisher.fisher_diagonal)}")
        
        return True
        
    except RuntimeError as e:
        if "in-place" in str(e).lower() or "inplace" in str(e).lower():
            print(f"❌ INPLACE ERROR: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_ewc_loss_computation():
    """Test EWC loss computation with multiple tasks."""
    print("\n" + "="*60)
    print("TEST 2: EWC Loss Computation")
    print("="*60)
    
    try:
        model = create_simple_model()
        
        # Create engine
        engine = create_continual_learning_engine(
            strategy=ForgettingPreventionStrategy.EWC,
            ewc_lambda=0.5
        )
        
        # Simulate two tasks
        for task_num in [1, 2]:
            dataset = create_simple_dataset(100)
            dataloader = DataLoader(dataset, batch_size=16)
            
            task = Task(
                task_id=f"test_task_{task_num}",
                task_name=f"Test Task {task_num}",
                task_type=TaskType.CLASSIFICATION,
                input_dim=10,
                output_dim=2
            )
            
            print(f"\nTask {task_num}:")
            
            # Start task
            engine.start_task_training(task, model)
            
            # Compute Fisher
            fisher = engine.compute_fisher_information(model, task, dataloader, num_samples=50)
            print(f"  ✅ Fisher computed: {fisher.num_samples} samples")
            
            # Test EWC loss computation
            X, y = next(iter(dataloader))
            output = model(X)
            base_loss = nn.CrossEntropyLoss()(output, y)
            
            # This should add regularization from previous task
            total_loss = engine.compute_ewc_loss(model, base_loss)
            
            print(f"  ✅ Loss computed: base={base_loss.item():.4f}, total={total_loss.item():.4f}")
            
            # Backward pass (critical test for inplace operations)
            total_loss.backward()
            print(f"  ✅ Backward pass successful (no inplace errors)")
            
            # Finish task
            engine.finish_task_training(task, model, dataloader, accuracy=0.8)
        
        print("\n✅ All EWC tests passed!")
        return True
        
    except RuntimeError as e:
        if "in-place" in str(e).lower() or "inplace" in str(e).lower():
            print(f"\n❌ INPLACE ERROR: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_combined_strategy():
    """Test combined strategy with all components."""
    print("\n" + "="*60)
    print("TEST 3: Combined Strategy (EWC + Replay + Adapters)")
    print("="*60)
    
    try:
        model = create_simple_model()
        
        # Create engine with combined strategy
        engine = create_continual_learning_engine(
            strategy=ForgettingPreventionStrategy.COMBINED,
            ewc_lambda=0.4,
            replay_buffer_size=50,
            use_adapters=True
        )
        
        # Run 3 tasks
        for task_num in range(1, 4):
            dataset = create_simple_dataset(100)
            dataloader = DataLoader(dataset, batch_size=16)
            
            task = Task(
                task_id=f"task_{task_num}",
                task_name=f"Task {task_num}",
                task_type=TaskType.CLASSIFICATION,
                input_dim=10,
                output_dim=2
            )
            
            print(f"\nTask {task_num}:")
            
            # Start task
            start_info = engine.start_task_training(task, model)
            print(f"  Started: {start_info['components_initialized']}")
            
            # Train for a few steps
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            for batch_idx, (X, y) in enumerate(dataloader):
                if batch_idx >= 3:  # Just test a few batches
                    break
                
                optimizer.zero_grad()
                output = model(X)
                base_loss = nn.CrossEntropyLoss()(output, y)
                
                # Add continual learning regularization
                train_info = engine.train_step(model, (X, y), optimizer, task)
                
                if 'additional_loss' in train_info:
                    total_loss = base_loss + train_info['additional_loss']
                else:
                    total_loss = base_loss
                
                # Backward pass (tests for inplace errors)
                total_loss.backward()
                optimizer.step()
                
                print(f"  Batch {batch_idx+1}: loss={total_loss.item():.4f}")
            
            # Finish task
            finish_info = engine.finish_task_training(task, model, dataloader, accuracy=0.85)
            print(f"  Finished: {finish_info['components_finalized']}")
        
        print("\n✅ Combined strategy test passed!")
        return True
        
    except RuntimeError as e:
        if "in-place" in str(e).lower() or "inplace" in str(e).lower():
            print(f"\n❌ INPLACE ERROR: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("INPLACE OPERATION FIX VERIFICATION")
    print("Testing: Fisher computation, EWC loss, Combined strategy")
    print("="*60)
    
    results = []
    
    # Test 1: Fisher Information
    results.append(("Fisher Computation", test_fisher_computation()))
    
    # Test 2: EWC Loss
    results.append(("EWC Loss", test_ewc_loss_computation()))
    
    # Test 3: Combined Strategy
    results.append(("Combined Strategy", test_combined_strategy()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Inplace operations fixed successfully.")
        print("\n✅ Ready to run real benchmarks!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED. Inplace issues remain.")
        return 1

if __name__ == "__main__":
    exit(main())
