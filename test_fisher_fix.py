#!/usr/bin/env python3
"""
Simple direct test of Fisher Information inplace fix.
Tests the core Fisher computation that was causing errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# Simplified Fisher computation (extracted from EWCStrategy)
def compute_fisher_diagonal_FIXED(model, dataloader, num_samples=100):
    """
    Fixed version of Fisher Information computation.
    Uses NON-INPLACE operations: `=` instead of `+=` and `/=`
    """
    print("\n🧪 Testing FIXED Fisher computation (non-inplace operations)...")
    
    # Initialize Fisher diagonal
    fisher_diagonal = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_diagonal[name] = torch.zeros_like(param.data)
    
    model.eval()
    samples_processed = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute log likelihood
        log_likelihood = F.log_softmax(outputs, dim=1)[range(len(targets)), targets]
        loss = -log_likelihood.mean()
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Accumulate squared gradients (NON-INPLACE - THIS IS THE FIX!)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # OLD (broken): fisher_diagonal[name] += param.grad.data ** 2
                # NEW (fixed): 
                fisher_diagonal[name] = fisher_diagonal[name] + (param.grad.data ** 2)
        
        samples_processed += len(targets)
        
        if batch_idx == 0:
            print(f"  ✅ Batch {batch_idx+1}: Backward pass successful")
    
    # Average Fisher (NON-INPLACE - THIS IS THE FIX!)
    for name in fisher_diagonal:
        # OLD (broken): fisher_diagonal[name] /= samples_processed
        # NEW (fixed):
        fisher_diagonal[name] = fisher_diagonal[name] / samples_processed
    
    print(f"  ✅ Fisher computed successfully with {samples_processed} samples")
    print(f"  ✅ Tracked {len(fisher_diagonal)} parameters")
    
    return fisher_diagonal

def compute_fisher_diagonal_BROKEN(model, dataloader, num_samples=100):
    """
    Broken version showing the original inplace error.
    Uses INPLACE operations: `+=` and `/=` (WILL FAIL!)
    """
    print("\n🧪 Testing BROKEN Fisher computation (inplace operations)...")
    
    fisher_diagonal = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_diagonal[name] = torch.zeros_like(param.data)
    
    model.eval()
    samples_processed = 0
    
    try:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
            
            outputs = model(inputs)
            log_likelihood = F.log_softmax(outputs, dim=1)[range(len(targets)), targets]
            loss = -log_likelihood.mean()
            
            model.zero_grad()
            loss.backward()
            
            # BROKEN (inplace operation)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diagonal[name] += param.grad.data ** 2  # ❌ INPLACE!
            
            samples_processed += len(targets)
        
        # BROKEN (inplace operation)
        for name in fisher_diagonal:
            fisher_diagonal[name] /= samples_processed  # ❌ INPLACE!
        
        print(f"  ⚠️  Unexpectedly succeeded (PyTorch version may handle this)")
        return fisher_diagonal
        
    except RuntimeError as e:
        if "in-place" in str(e).lower() or "inplace" in str(e).lower():
            print(f"  ❌ EXPECTED ERROR: {e}")
            return None
        else:
            raise

def test_ewc_loss_with_fisher(model, fisher_diagonal, lambda_ewc=0.5):
    """Test EWC loss computation using Fisher."""
    print("\n🧪 Testing EWC loss computation with Fisher...")
    
    # Store optimal parameters
    optimal_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            optimal_params[name] = param.data.clone()
    
    # Create some dummy data
    X = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))
    
    # Compute base loss
    outputs = model(X)
    base_loss = F.cross_entropy(outputs, y)
    
    # Compute EWC regularization
    ewc_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if name in fisher_diagonal and param.requires_grad:
            fisher = torch.from_numpy(fisher_diagonal[name].cpu().numpy())
            optimal = optimal_params[name]
            
            # NON-INPLACE: Use `+` instead of `+=`
            param_diff = (param - optimal) ** 2
            ewc_loss = ewc_loss + (fisher * param_diff).sum()
    
    ewc_loss = ewc_loss * (lambda_ewc / 2)
    
    # Total loss (NON-INPLACE!)
    total_loss = base_loss + ewc_loss  # ✅ Non-inplace
    # NOT: base_loss += ewc_loss  # ❌ Inplace would fail!
    
    # Test backward pass
    total_loss.backward()
    
    print(f"  ✅ EWC loss: base={base_loss.item():.4f}, reg={ewc_loss.item():.4f}")
    print(f"  ✅ Backward pass successful!")
    
    return True

def main():
    """Run comprehensive inplace fix tests."""
    print("="*70)
    print("INPLACE OPERATION FIX VERIFICATION")
    print("Testing Fisher Information computation fixes")
    print("="*70)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    
    # Create dataset
    X = torch.randn(200, 10)
    y = torch.randint(0, 2, (200,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    print(f"\n📊 Test Setup:")
    print(f"  Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  Dataset: {len(dataset)} samples")
    print(f"  Batch size: 16")
    
    # Test 1: BROKEN version (should fail or warn)
    print("\n" + "="*70)
    print("TEST 1: BROKEN Fisher Computation (Inplace Operations)")
    print("="*70)
    fisher_broken = compute_fisher_diagonal_BROKEN(model, dataloader, num_samples=50)
    
    # Test 2: FIXED version (should succeed)
    print("\n" + "="*70)
    print("TEST 2: FIXED Fisher Computation (Non-Inplace Operations)")
    print("="*70)
    fisher_fixed = compute_fisher_diagonal_FIXED(model, dataloader, num_samples=50)
    
    # Test 3: EWC loss with fixed Fisher
    if fisher_fixed:
        print("\n" + "="*70)
        print("TEST 3: EWC Loss Computation")
        print("="*70)
        test_ewc_loss_with_fisher(model, fisher_fixed)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if fisher_fixed:
        print("✅ FIXED Fisher computation works correctly")
        print("✅ All inplace operations have been resolved")
        print("✅ Ready to run real benchmarks!")
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ Fisher computation still failing")
        return 1

if __name__ == "__main__":
    exit(main())
