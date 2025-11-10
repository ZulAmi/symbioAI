#!/usr/bin/env python3
"""
Optimized TRUE Causality Training Script
=========================================

Runs optimized TRUE interventional causality with Phase 1 optimizations:
- true_micro_steps: 2 → 1 (2x speedup)
- causal_hybrid_candidates: 200 → 100 (2x speedup)
- causal_eval_interval: 5 (amortization)

Expected: ~3x speedup vs baseline (43 min → 15 min for 5 epochs)

Usage:
    # 5-epoch test (default, for RunPod)
    python3 run_optimized_true_causality.py
    
    # 1-epoch quick test (for local Mac)
    python3 run_optimized_true_causality.py --n_epochs 1
    
    # Custom seed
    python3 run_optimized_true_causality.py --seed 42
    
    # Override optimizations (testing)
    python3 run_optimized_true_causality.py --true_micro_steps 2 --causal_hybrid_candidates 200

Author: Zulhilmi Rahmat
Date: November 7, 2025
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_environment():
    """Check Python, PyTorch, and CUDA availability."""
    print("\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    # Check Python version
    print(f"Python: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("WARNING: PyTorch not installed!")
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        import torch
        print(f"PyTorch: {torch.__version__} (just installed)")
    
    print("=" * 60)
    print()


def check_directory():
    """Verify we're in the correct directory."""
    # Check for key files
    required_files = ["utils/main.py", "models/derpp_causal.py", "training/causal_inference.py"]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"\nERROR: {file} not found!")
            print(f"Current directory: {os.getcwd()}")
            print("\nPlease run this script from the repository root:")
            print("  /workspace/mammoth  (on RunPod)")
            print("  or")
            print("  .../Symbio AI  (on Mac)")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run optimized TRUE interventional causality training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 5-epoch test (RunPod)
  python3 run_optimized_true_causality.py
  
  # Quick 1-epoch test (local Mac)
  python3 run_optimized_true_causality.py --n_epochs 1 --seed 42
  
  # Test without optimizations (baseline)
  python3 run_optimized_true_causality.py --true_micro_steps 2 --causal_hybrid_candidates 200

Baseline (true.log):
  Runtime: 43 minutes (5 epochs, seed 1)
  Class-IL: 24.04%
  Config: true_micro_steps=2, causal_hybrid_candidates=200

Optimized (default):
  Expected runtime: ~15 minutes (5 epochs)
  Target Class-IL: ≥23.5%
  Config: true_micro_steps=1, causal_hybrid_candidates=100
        """
    )
    
    # Core experiment settings
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='Number of epochs per task (default: 5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1 for comparison with baseline)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    
    # Optimization parameters
    parser.add_argument('--true_micro_steps', type=int, default=1,
                        help='Micro-steps for TRUE intervention (default: 1, baseline: 2)')
    parser.add_argument('--causal_hybrid_candidates', type=int, default=100,
                        help='Candidates for TRUE causality (default: 100, baseline: 200)')
    parser.add_argument('--causal_eval_interval', type=int, default=5,
                        help='Reuse causal selection interval (default: 5)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='validation/results/optimized',
                        help='Output directory for logs (default: validation/results/optimized)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output log filename (default: auto-generated)')
    
    # Advanced settings (rarely changed)
    parser.add_argument('--buffer_size', type=int, default=500,
                        help='Replay buffer size (default: 500)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='DER++ alpha (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='DER++ beta (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Learning rate (default: 0.03)')
    parser.add_argument('--debug', type=int, default=1,
                        help='Debug logging (default: 1)')
    
    args = parser.parse_args()
    
    # Check environment
    check_directory()
    check_environment()
    
    # Print configuration
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Epochs per task: {args.n_epochs}")
    print(f"Random seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print()
    print("Optimization Settings:")
    print(f"  • true_micro_steps: {args.true_micro_steps} (baseline: 2)")
    print(f"  • causal_hybrid_candidates: {args.causal_hybrid_candidates} (baseline: 200)")
    print(f"  • causal_eval_interval: {args.causal_eval_interval}")
    
    # Calculate expected speedup
    speedup_micro = 2 / args.true_micro_steps
    speedup_candidates = 200 / args.causal_hybrid_candidates
    speedup_total = speedup_micro * speedup_candidates
    
    print()
    print(f"Expected speedup: ~{speedup_total:.1f}x")
    if args.n_epochs == 5:
        expected_time = 43 / speedup_total
        print(f"Expected runtime: ~{expected_time:.0f} minutes (vs 43 min baseline)")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if args.output_name:
        output_file = output_dir / args.output_name
    else:
        output_file = output_dir / f"{args.n_epochs}epoch_seed{args.seed}_optimized.log"
    
    print(f"Output log: {output_file}")
    print()
    
    # Build command
    cmd = [
        sys.executable,
        "utils/main.py",
        "--model", "derpp-causal",
        "--dataset", "seq-cifar100",
        "--buffer_size", str(args.buffer_size),
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--n_epochs", str(args.n_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lr_scheduler", "multisteplr",
        "--lr_milestones", "3", "4",
        "--use_causal_sampling", "3",
        "--temperature", "2.0",
        "--true_micro_steps", str(args.true_micro_steps),
        "--true_temp_lr", "0.05",
        "--causal_hybrid_candidates", str(args.causal_hybrid_candidates),
        "--causal_eval_interval", str(args.causal_eval_interval),
        "--causal_num_interventions", "50",
        "--causal_effect_threshold", "0.05",
        "--causal_blend_ratio", "0.3",
        "--causal_batch_size", "16",
        "--causal_warmup_tasks", "5",
        "--use_batched_causality", "1",
        "--debug", str(args.debug),
        "--seed", str(args.seed),
    ]
    
    # Print command for debugging
    print("=" * 60)
    print("Running Command:")
    print("=" * 60)
    print(" \\\n  ".join(cmd))
    print("=" * 60)
    print()
    
    # Run training with output to both stdout and file
    with open(output_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output to both terminal and file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
    
    # Print summary
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Exit code: {process.returncode}")
    print(f"Results saved to: {output_file}")
    print()
    
    # Try to extract final accuracy
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Class-IL" in line and "%" in line:
                    print("Final Result:")
                    print(f"  {line.strip()}")
                    break
    except Exception as e:
        print(f"Could not extract final accuracy: {e}")
    
    print()
    print("To view full results:")
    print(f"  cat {output_file}")
    print()
    print("To compare with baseline:")
    print(f"  grep 'Class-IL' validation/results/new5ep/true.log | tail -1")
    print(f"  grep 'Class-IL' {output_file} | tail -1")
    print()
    
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
