#!/usr/bin/env python3
"""
ULTRA-FAST PRE-FLIGHT TEST: Just verify code paths work.

Tests ONLY the first few batches to ensure:
1. Graph mode initializes correctly
2. TRUE mode initializes correctly
3. No immediate crashes

Takes ~30 seconds instead of 5 minutes.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mammoth'))
sys.path.insert(0, os.path.dirname(__file__))

def test_graph_mode():
    """Test Graph Heuristic mode initialization."""
    print("Testing Graph mode imports and initialization...")
    
    try:
        # Add mammoth to path
        mammoth_path = os.path.join(os.path.dirname(__file__), 'mammoth')
        if mammoth_path not in sys.path:
            sys.path.insert(0, mammoth_path)
        
        from argparse import Namespace
        from backbone.ResNetBlock import resnet18
        from datasets.seq_cifar100 import SequentialCIFAR100
        
        print("  ✅ Mammoth imports SUCCESS")
        
        # Try importing the causal model
        try:
            from training.derpp_causal import DerppCausal
            print("  ✅ Import from training.derpp_causal SUCCESS")
        except ImportError:
            # Fallback to models path (for RunPod)
            models_path = os.path.join(os.path.dirname(__file__), 'mammoth', 'models')
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            from derpp_causal import DerppCausal
            print("  ✅ Import from models.derpp_causal SUCCESS (RunPod path)")
        
        # Create minimal args with CORRECT lr_milestones
        args = Namespace(
            dataset='seq-cifar100',
            buffer_size=100,
            n_epochs=1,
            lr=0.03,
            alpha=0.1,  # DER++ official
            beta=0.5,   # DER++ official
            lr_milestones=[3, 4],  # CRITICAL: Correct for 5-epoch training
            lr_scheduler='multisteplr',
            seed=42,
            batch_size=32,
            minibatch_size=32,
            enable_causal_graph_learning=1,
            use_causal_sampling=1,
            num_tasks=10,
            temperature=2.0,
            debug=1,
            # Add all other required args
            optim_wd=0,
            optim_mom=0,
            optim_nesterov=0,
        )
        
        print("  ✅ Args created")
        
        # Try to create model instance
        backbone = resnet18(num_classes=100)
        print("  ✅ Backbone created")
        
        dataset = SequentialCIFAR100(args)
        print("  ✅ Dataset created")
        
        # This will test if derpp_causal initializes correctly
        from torch.nn import CrossEntropyLoss
        model = DerppCausal(
            backbone=backbone,
            loss=CrossEntropyLoss(),
            args=args,
            transform=dataset.get_transform(),
            dataset=dataset
        )
        print("  ✅ Model initialized")
        
        # Check causal components
        assert hasattr(model, 'enable_causal_graph'), "Missing enable_causal_graph"
        assert hasattr(model, 'use_causal_sampling'), "Missing use_causal_sampling"
        assert model.enable_causal_graph == 1, "Graph learning not enabled"
        assert model.use_causal_sampling == 1, "Causal sampling not enabled"
        
        print("  ✅ Graph mode configuration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Graph mode FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_true_causality_mode():
    """Test TRUE Interventional mode initialization."""
    print("Testing TRUE causality mode imports and initialization...")
    
    try:
        # Add mammoth to path
        mammoth_path = os.path.join(os.path.dirname(__file__), 'mammoth')
        if mammoth_path not in sys.path:
            sys.path.insert(0, mammoth_path)
        
        from argparse import Namespace
        from backbone.ResNetBlock import resnet18
        from datasets.seq_cifar100 import SequentialCIFAR100
        
        print("  ✅ Mammoth imports SUCCESS")
        
        # Try importing the causal model
        try:
            from training.derpp_causal import DerppCausal
            print("  ✅ Import from training.derpp_causal SUCCESS")
        except ImportError:
            # Fallback to models path (for RunPod)
            models_path = os.path.join(os.path.dirname(__file__), 'mammoth', 'models')
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            from derpp_causal import DerppCausal
            print("  ✅ Import from models.derpp_causal SUCCESS (RunPod path)")
        
        # Create minimal args for TRUE mode with CORRECT lr_milestones
        args = Namespace(
            dataset='seq-cifar100',
            buffer_size=100,
            n_epochs=1,
            lr=0.03,
            alpha=0.1,  # DER++ official
            beta=0.5,   # DER++ official
            lr_milestones=[3, 4],  # CRITICAL: Correct for 5-epoch training
            lr_scheduler='multisteplr',
            seed=42,
            batch_size=32,
            minibatch_size=32,
            enable_causal_graph_learning=0,  # OFF for TRUE mode
            use_causal_sampling=3,  # TRUE interventional
            num_tasks=10,
            temperature=2.0,
            debug=1,
            causal_num_interventions=10,
            causal_eval_interval=1,
            # Add all other required args
            optim_wd=0,
            optim_mom=0,
            optim_nesterov=0,
        )
        
        print("  ✅ Args created")
        
        # Try to create model instance
        backbone = resnet18(num_classes=100)
        print("  ✅ Backbone created")
        
        dataset = SequentialCIFAR100(args)
        print("  ✅ Dataset created")
        
        # This will test if derpp_causal initializes correctly
        from torch.nn import CrossEntropyLoss
        model = DerppCausal(
            backbone=backbone,
            loss=CrossEntropyLoss(),
            args=args,
            transform=dataset.get_transform(),
            dataset=dataset
        )
        print("  ✅ Model initialized")
        
        # Check TRUE causality components
        assert hasattr(model, 'use_true_causality'), "Missing use_true_causality"
        assert model.use_true_causality == 3, f"TRUE causality not enabled (got {model.use_true_causality})"
        assert hasattr(model, 'causal_forgetting_detector'), "Missing causal_forgetting_detector"
        
        print("  ✅ TRUE mode configuration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ TRUE mode FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("ULTRA-FAST PRE-FLIGHT TEST")
    print("="*60)
    print("Testing initialization only (~30 seconds)")
    print()
    
    graph_ok = test_graph_mode()
    true_ok = test_true_causality_mode()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if graph_ok:
        print("✅ GRAPH HEURISTIC - Initialization successful")
    else:
        print("❌ GRAPH HEURISTIC - Initialization failed")
    
    if true_ok:
        print("✅ TRUE INTERVENTIONAL - Initialization successful")
    else:
        print("❌ TRUE INTERVENTIONAL - Initialization failed")
    
    print()
    
    if graph_ok and true_ok:
        print("🎉 ALL TESTS PASSED - CODE PATHS VERIFIED")
        print()
        print("="*60)
        print("READY FOR RUNPOD - Use these commands:")
        print("="*60)
        print()
        print("# 1. Vanilla DER++ baseline:")
        print("python utils/main.py --model derpp --dataset seq-cifar100 \\")
        print("  --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 \\")
        print("  --lr_milestones 3 4 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/vanilla_5ep.log")
        print()
        print("# 2. Graph Heuristic:")
        print("python utils/main.py --model derpp-causal --dataset seq-cifar100 \\")
        print("  --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 \\")
        print("  --lr_milestones 3 4 --enable_causal_graph_learning 1 \\")
        print("  --use_causal_sampling 1 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/graph_5ep.log")
        print()
        print("# 3. TRUE Interventional:")
        print("python utils/main.py --model derpp-causal --dataset seq-cifar100 \\")
        print("  --buffer_size 500 --alpha 0.1 --beta 0.5 --n_epochs 5 --lr 0.03 \\")
        print("  --lr_milestones 3 4 --use_causal_sampling 3 --debug 1 \\")
        print("  --temperature 2.0 --seed 1 2>&1 | grep -v NNPACK | tee /workspace/true_5ep.log")
        print()
        print("="*60)
        print("IMPORTANT: All 3 use alpha=0.1, beta=0.5 (DER++ official)")
        print("           lr_milestones=3,4 (CORRECT for 5-epoch training)")
        print("="*60)
        return 0
    else:
        print("❌ TESTS FAILED - Fix initialization errors before RunPod")
        return 1

if __name__ == '__main__':
    sys.exit(main())
