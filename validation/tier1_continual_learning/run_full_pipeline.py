#!/usr/bin/env python3
"""
Master Experimental Pipeline for Causal-DER

Runs complete experimental pipeline:
1. Baseline validation (DER++ with 5 seeds)
2. Causal-DER experiments (5 seeds)
3. Ablation studies
4. Statistical analysis and visualization

Usage:
    python run_full_pipeline.py --quick  # Quick test (1 seed each)
    python run_full_pipeline.py          # Full pipeline (5 seeds)
"""

import sys
from pathlib import Path
import argparse
import time

sys.path.append(str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Run complete Causal-DER experimental pipeline')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test with 1 seed (for debugging)')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline validation (if already run)')
    parser.add_argument('--skip_causal', action='store_true',
                       help='Skip Causal-DER experiments')
    parser.add_argument('--skip_ablation', action='store_true',
                       help='Skip ablation studies')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*80)
    print("CAUSAL-DER EXPERIMENTAL PIPELINE")
    print("="*80)
    print()
    
    # Phase 1: Baseline Validation
    if not args.skip_baseline:
        print("\n" + "="*80)
        print("PHASE 1: BASELINE VALIDATION (DER++)")
        print("="*80)
        
        if args.quick:
            print("Running QUICK baseline (1 seed)...")
            from run_clean_der_plus_plus import run_der_plus_plus_benchmark
            run_der_plus_plus_benchmark(seed=42)
        else:
            print("Running FULL baseline (5 seeds)...")
            from run_baseline_validation import main as run_baseline
            run_baseline()
        
        print("\n✅ Baseline validation complete!")
    
    # Phase 2: Causal-DER Experiments
    if not args.skip_causal:
        print("\n" + "="*80)
        print("PHASE 2: CAUSAL-DER EXPERIMENTS")
        print("="*80)
        
        from run_causal_der_benchmark import run_causal_der_benchmark
        
        if args.quick:
            print("Running QUICK Causal-DER (1 seed)...")
            run_causal_der_benchmark(seed=42)
        else:
            print("Running FULL Causal-DER (5 seeds)...")
            seeds = [42, 123, 456, 789, 1337]
            for i, seed in enumerate(seeds):
                print(f"\n[{i+1}/5] Running Causal-DER with seed {seed}...")
                run_causal_der_benchmark(seed=seed)
        
        print("\n✅ Causal-DER experiments complete!")
    
    # Phase 3: Ablation Studies
    if not args.skip_ablation and not args.quick:
        print("\n" + "="*80)
        print("PHASE 3: ABLATION STUDIES")
        print("="*80)
        
        from run_ablations import main as run_ablations
        run_ablations()
        
        print("\n✅ Ablation studies complete!")
    
    # Phase 4: Statistical Analysis
    print("\n" + "="*80)
    print("PHASE 4: STATISTICAL ANALYSIS")
    print("="*80)
    
    try:
        from statistical_analysis import main as run_stats
        run_stats()
        print("\n✅ Statistical analysis complete!")
    except Exception as e:
        print(f"\n⚠️  Statistical analysis failed: {e}")
        print("   (May need more results files)")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print()
    print("Results saved in: validation/results/")
    print("  - der_plus_plus_baseline.json")
    print("  - causal_der_seed*.json")
    print("  - ablation_study.json")
    print("  - statistical_analysis.json")
    print("  - plots/")
    print()
    print("Next steps:")
    print("  1. Check statistical_analysis.json for significance")
    print("  2. Review plots in results/plots/")
    print("  3. If significant improvement, prepare paper!")
    print("="*80)


if __name__ == "__main__":
    main()
