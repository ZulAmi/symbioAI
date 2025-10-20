# 🚨 URGENT: Causal-DER Performance Crisis

**Date**: October 19, 2025 23:05 UTC  
**Status**: ❌ CRITICAL FAILURE - Causal-DER underperforms DER++ by 16.6x

---

## Executive Summary

**Result**: After completing both Causal-DER and DER++ baseline experiments, **Causal-DER performs dramatically worse**:

| Metric       | DER++ Baseline | Causal-DER | Performance        |
| ------------ | -------------- | ---------- | ------------------ |
| **Class-IL** | 14.92%         | 0.90%      | **16.6x WORSE** ❌ |
| **Task-IL**  | 61.45%         | 11.25%     | **5.5x WORSE** ❌  |
| **Task 10**  | 75.80%         | 3.00%      | **25x WORSE** ❌   |
| **Time**     | ~53 min        | ~90 min    | **40% SLOWER** ❌  |

**Conclusion**: Every metric shows Causal-DER is significantly worse than the baseline it's supposed to improve.

---

## Critical Issues Identified

### 1. **Current Task Learning Failure** (Most Severe)

- DER++ learns Task 10 well: 75.8% accuracy
- Causal-DER FAILS on Task 10: 3.0% accuracy
- **This is the task we JUST trained on!**
- Suggests fundamental issue in loss computation or optimization

### 2. **Gradient Clipping Too Strict**

- Causal-DER: `clip_grad=1.0` (needed to prevent NaN)
- DER++: NO gradient clipping (Mammoth framework limitation)
- **Hypothesis**: 1.0 is too conservative, prevents learning
- Standard practice: 5.0-10.0 for continual learning

### 3. **Causal Sampling May Harm Performance**

- Causal importance (avg=0.44) filters samples
- **Hypothesis**: Discarding high-loss "hard" samples needed for learning
- DER++ uses uniform random sampling (no filtering)
- Need ablation: `use_causal_sampling=0` to test

### 4. **Temperature Too High**

- Causal-DER: T=2.0 (soft distillation)
- DER++ equivalent: T=1.0 (uses MSE, not KL)
- **Hypothesis**: Weak distillation signal harms retention
- Literature uses T=1.0-1.5 for continual learning

### 5. **Computational Overhead**

- 30-edge causal graph learning
- MIR 3x candidate pool sampling
- Feature hooks (even with `store_features=0`?)
- **Result**: 40% slower with no performance benefit

---

## Immediate Action Plan

### 🔧 Priority 1 Fixes (Apply NOW)

Create new script: `run_causal_der_5epoch_fixed.sh`

```bash
python3 mammoth/utils/main.py \
  --model causal_der \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --alpha 0.3 \
  --beta 0.5 \
  --n_epochs 5 \
  --batch_size 32 \
  --lr 0.005 \
  --optim_wd 0.0 \
  --optim_mom 0.0 \
  --clip_grad 10.0 \              # CHANGED: 1.0 → 10.0 (less strict)
  --temperature 1.0 \              # CHANGED: 2.0 → 1.0 (stronger distillation)
  --use_causal_sampling 0 \        # CHANGED: 1 → 0 (disable causal filtering)
  --use_mir_sampling 0 \           # CHANGED: 1 → 0 (disable MIR overhead)
  --seed 42 \
  --dataset_config xder
```

**Expected Result**: Should match DER++ ~15% Class-IL

### 🔍 Priority 2 Ablation Study

If Priority 1 works, test features individually:

1. **Baseline** (all causal OFF): Should match DER++ exactly
2. **+Gradient clip only** (10.0): Safety without too much harm
3. **+Temperature 1.0**: Standard distillation
4. **+Causal sampling**: Does it help or hurt?
5. **+MIR sampling**: Does it help or hurt?
6. **All features ON**: Current config (we know this fails)

### 📊 Priority 3 Extended Validation

**ONLY if Priority 1+2 succeed:**

- Increase to 50 epochs
- Multiple seeds (42, 43, 44)
- Additional datasets

---

## Root Cause Hypotheses

### Most Likely:

1. ✅ **Gradient clipping too strict** (1.0 vs DER++'s none)
2. ✅ **Causal sampling discards useful samples** (filtering hurts)
3. ✅ **Temperature too high** (weak distillation signal)

### Possible:

4. ⚠️ **Feature distillation overhead** (even with weight=0.0?)
5. ⚠️ **MIR sampling overhead** (3x pool without benefit)
6. ⚠️ **Causal graph overhead** (30 edges, no value in 5 epochs)

### Unlikely:

7. ❓ **Bug in engine implementation** (test with all features OFF)
8. ❓ **Buffer corruption** (seems healthy: 500/500 samples)
9. ❓ **LR too low** (0.005 works for DER++, shouldn't be issue)

---

## Decision Tree

```
START: Causal-DER underperforms (0.90% vs 14.92%)
  │
  ├─→ Apply Priority 1 fixes → Re-run 5 epochs
  │   │
  │   ├─→ SUCCESS (matches DER++ ~15%)
  │   │   │
  │   │   ├─→ Run Priority 2 ablation
  │   │   │   │
  │   │   │   ├─→ Find features that help
  │   │   │   │   └─→ Extend to 50 epochs → PAPER ✅
  │   │   │   │
  │   │   │   └─→ All features hurt
  │   │   │       └─→ Publish negative results 📝
  │   │   │
  │   │   └─→ Skip ablation, extend to 50 epochs
  │   │       └─→ If competitive → PAPER ✅
  │   │
  │   └─→ FAILURE (still underperforms)
  │       │
  │       ├─→ Deep debug (bug in engine?)
  │       │   └─→ Fix bug → Retry
  │       │
  │       └─→ Abandon causal approach
  │           └─→ Simple DER++ improvements instead
  │
  └─→ Give up on causal
      └─→ Focus on other research directions
```

---

## What NOT To Do

❌ **DO NOT run 50-epoch experiment yet**

- Would waste ~15 hours of compute
- Current config is fundamentally broken
- Fix 5-epoch first, then scale up

❌ **DO NOT publish current results as positive**

- 16.6x worse than baseline is NOT publishable
- Would damage reputation

❌ **DO NOT add more causal features**

- Current features already hurt performance
- Complexity is the problem, not the solution

❌ **DO NOT ignore the comparison**

- "5 epochs is too few" is NOT an excuse
- DER++ works fine with 5 epochs (14.92%)
- Something is fundamentally wrong with Causal-DER

---

## Communication Strategy

### If Asked About Progress:

**Honest Response**:

> "We completed the 5-epoch validation experiments. The good news:
> training is stable with no NaN errors. The bad news: Causal-DER
> currently underperforms the DER++ baseline by a significant margin
> (0.90% vs 14.92% Class-IL). We've identified several critical issues
> (gradient clipping too strict, causal sampling possibly harmful,
> temperature suboptimal) and are applying fixes now. We expect a
> re-run with corrections to match or exceed the baseline."

**Optimistic Spin** (if needed):

> "The experiments revealed important insights about integrating causal
> reasoning into continual learning. We've identified specific
> hyperparameter issues and are iterating rapidly toward a solution."

**Academic Honesty** (preferred):

> "Initial results show our causal approach underperforms the baseline.
> This is actually valuable scientific information - we're learning what
> DOESN'T work, which will guide better designs. We're conducting
> systematic ablation studies to understand which components help vs hurt."

---

## Timeline

**Today (Oct 19)**:

- ✅ Completed both experiments
- ✅ Identified critical issues
- ✅ Documented findings

**Tomorrow (Oct 20)**:

- 🔧 Apply Priority 1 fixes
- 🔄 Re-run 5-epoch experiment (~90 min)
- 📊 Compare results

**Oct 21-22**:

- 🔍 Ablation study (if fixes work)
- 📈 Extend to 50 epochs (if ablation positive)

**Oct 23-25**:

- ✍️ Write paper (if results good)
- OR pivot to alternative approach

---

## Key Takeaways

1. **Stability ≠ Performance**: Causal-DER is stable (no NaN) but performs terribly
2. **Complexity is expensive**: 40% slower with no benefit
3. **Baselines matter**: Without DER++ comparison, we'd think 0.9% was just "low-epoch effect"
4. **Early testing pays off**: Found issues at 5 epochs, not after wasting weeks on 50-epoch runs
5. **Negative results are results**: This is still valuable scientific information

---

## Files Updated

- ✅ `RESULTS_5EPOCH.md` - Full comparison analysis
- ✅ `URGENT_FINDINGS.md` - This document
- ✅ `run_derpp_5epoch_baseline.sh` - Working baseline script
- ⏳ `run_causal_der_5epoch_fixed.sh` - TODO: Create with fixes

---

## Next Steps

1. **Read this document carefully**
2. **Decide on approach** (fix vs pivot vs abandon)
3. **If fixing**: Create `run_causal_der_5epoch_fixed.sh` with Priority 1 changes
4. **If pivoting**: Decide on alternative research direction
5. **If abandoning**: Document lessons learned for future work

**Recommended**: Fix and retry (Priority 1). We've invested significant effort and the issues are tractable.

---

**Author**: Symbio AI / GitHub Copilot  
**Date**: October 19, 2025  
**Status**: CRITICAL - REQUIRES IMMEDIATE ATTENTION
