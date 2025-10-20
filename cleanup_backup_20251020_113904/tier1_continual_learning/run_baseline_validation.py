#!/usr/bin/env python3
"""
Baseline Validation: DER++ on CIFAR-100

Run DER++ with exact paper settings to establish baseline.
This is the number Causal-DER must beat.

Expected Results: 70-72% average accuracy
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from validation.tier1_continual_learning.run_clean_der_plus_plus import run_der_plus_plus_benchmark


def main():
    print("="*60)
    print("BASELINE VALIDATION: DER++")
    print("Running 5 seeds for statistical validation")
    print("="*60)
    
    # Run 5 seeds for statistical validation
    results = []
    seeds = [42, 123, 456, 789, 1337]
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running DER++ with seed {seed}")
        print(f"{'='*60}\n")
        
        result = run_der_plus_plus_benchmark(seed=seed)
        results.append(result)
        
        print(f"\nSeed {seed} Results:")
        print(f"  Average Accuracy: {result['average_accuracy']:.4f}")
        print(f"  Forgetting: {result['forgetting_measure']:.4f}")
    
    # Aggregate
    avg_acc_mean = np.mean([r['average_accuracy'] for r in results])
    avg_acc_std = np.std([r['average_accuracy'] for r in results])
    forg_mean = np.mean([r['forgetting_measure'] for r in results])
    forg_std = np.std([r['forgetting_measure'] for r in results])
    final_acc_mean = np.mean([r['final_accuracy'] for r in results])
    final_acc_std = np.std([r['final_accuracy'] for r in results])
    
    print("\n" + "="*60)
    print("BASELINE RESULTS (5 seeds)")
    print("="*60)
    print(f"DER++ Average Accuracy: {avg_acc_mean:.4f} ± {avg_acc_std:.4f} ({avg_acc_mean*100:.2f}% ± {avg_acc_std*100:.2f}%)")
    print(f"DER++ Final Accuracy:   {final_acc_mean:.4f} ± {final_acc_std:.4f} ({final_acc_mean*100:.2f}% ± {final_acc_std*100:.2f}%)")
    print(f"DER++ Forgetting:       {forg_mean:.4f} ± {forg_std:.4f} ({forg_mean*100:.2f}% ± {forg_std*100:.2f}%)")
    print("="*60)
    print("\n✅ This is the baseline Causal-DER must beat!")
    print(f"✅ Target: >{avg_acc_mean*100:.1f}% average accuracy")
    
    # Save aggregate results
    import json
    results_dir = Path(__file__).parent / 'validation' / 'results'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    aggregate = {
        'method': 'DER++',
        'num_runs': len(results),
        'seeds': seeds,
        'average_accuracy_mean': float(avg_acc_mean),
        'average_accuracy_std': float(avg_acc_std),
        'final_accuracy_mean': float(final_acc_mean),
        'final_accuracy_std': float(final_acc_std),
        'forgetting_mean': float(forg_mean),
        'forgetting_std': float(forg_std),
        'individual_results': results
    }
    
    with open(results_dir / 'der_plus_plus_baseline.json', 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    print(f"\nAggregate results saved to: {results_dir / 'der_plus_plus_baseline.json'}")
    
    return results


if __name__ == "__main__":
    main()
