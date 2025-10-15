# Causal-DER Experimental Pipeline

This directory contains the complete experimental pipeline for validating Causal-DER against the DER++ baseline.

## Quick Start

### Option 1: Quick Test (1 seed, ~30 minutes)

```bash
python run_full_pipeline.py --quick
```

### Option 2: Full Validation (5 seeds, ~3-4 hours)

```bash
python run_full_pipeline.py
```

### Option 3: Step-by-Step

1. **Baseline Validation** (DER++ with 5 seeds)

   ```bash
   python run_baseline_validation.py
   ```

2. **Causal-DER Experiments** (5 seeds)

   ```bash
   python run_causal_der_benchmark.py --num_runs 5
   ```

3. **Ablation Studies**

   ```bash
   python run_ablations.py
   ```

4. **Statistical Analysis**
   ```bash
   python statistical_analysis.py
   ```

## Files

### Experiment Scripts

- `run_full_pipeline.py` - Master script that runs everything
- `run_baseline_validation.py` - DER++ baseline (5 seeds)
- `run_clean_der_plus_plus.py` - Single DER++ run
- `run_causal_der_benchmark.py` - Causal-DER experiments
- `run_ablations.py` - Ablation studies
- `statistical_analysis.py` - Statistical tests and plots

### Configuration

- `config.yaml` - Training configuration (UPDATED with correct hyperparameters!)
  - Learning rate: 0.03 (was 0.001)
  - Batch size: 32 (was 128)
  - Epochs: 50 (was 20)
  - Optimizer: SGD (was AdamW)

### Utilities

- `industry_standard_benchmarks.py` - Benchmark infrastructure
- `validation.py` - Validation utilities

## Expected Results

Based on DER++ paper (Buzzega et al., NeurIPS 2020):

- **DER++ baseline**: 70.5% ± 1.2% average accuracy
- **Target for Causal-DER**: 73-75% (2-4% improvement)
- **Statistical significance**: p < 0.05

## Hyperparameters (DER++ Paper Settings)

All experiments use these exact settings:

```yaml
Learning rate: 0.03
Optimizer: SGD (momentum=0.9)
Batch size: 32
Epochs per task: 50
Buffer size: 2000
Alpha (distillation): 0.5
```

## Output

Results are saved in `validation/results/`:

- `der_plus_plus_baseline.json` - Aggregated DER++ results
- `causal_der_seed{N}.json` - Individual Causal-DER runs
- `ablation_study.json` - Ablation results
- `statistical_analysis.json` - Statistical tests
- `plots/` - Comparison visualizations

## What Changed?

### ✅ Fixed Hyperparameters

- **Learning rate**: 0.001 → 0.03 (30x increase!)
- **Batch size**: 128 → 32 (4x decrease)
- **Epochs**: 20 → 50 (2.5x increase)
- **Optimizer**: AdamW → SGD with momentum

### ✅ Removed Custom Metrics

- No more `overall_score` (always 0.5)
- Using standard CL metrics only:
  - Average Accuracy
  - Forgetting Measure
  - Forward Transfer
  - Backward Transfer

### ✅ Clean Implementations

- `run_clean_der_plus_plus.py` - Exact DER++ replication
- `run_causal_der_benchmark.py` - Fair comparison
- No wrappers, no extra complexity

## Publication Readiness Checklist

- ✅ DER++ baseline with exact paper settings
- ✅ 5-seed validation for statistical power
- ✅ Ablation studies (random vs causal sampling)
- ✅ Statistical significance tests (t-test, Cohen's d)
- ✅ Standard metrics (no custom scores)
- ✅ Publication-quality plots

## Timeline

- **Week 1**: Run full pipeline (5 seeds each)
- **Week 2**: Analyze results, tune if needed
- **Week 3**: Write paper draft
- **Week 4**: Submit to ICLR/NeurIPS/ICML

## Troubleshooting

### If accuracy is still low (~51%):

1. Check that config.yaml has correct hyperparameters
2. Make sure you're running the new scripts (not old ones)
3. Verify data is loaded correctly (CIFAR-100, 50k train samples)

### If memory issues:

1. Reduce batch size (but keep 32 for fair comparison)
2. Use CPU instead of GPU
3. Clear buffer between runs

### If results are not significant:

1. Run more seeds (10 instead of 5)
2. Tune learning rate slightly (0.025-0.035)
3. Try different buffer management strategies

## Next Steps

After running the pipeline:

1. **Check statistical_analysis.json**:

   - Is p-value < 0.05?
   - Is improvement > 2%?
   - Is Cohen's d > 0.5?

2. **If YES** → Write paper!
3. **If NO** → Debug and tune

Good luck! 🚀
