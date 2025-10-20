# Causal-DER: 5-Epoch Experimental Results

## Experiment Configuration

**Goal**: Validate Causal-DER approach with reliable 5-epoch training for paper submission.

**Date**: October 19, 2025  
**Hardware**: Apple Silicon (MPS)  
**Framework**: Mammoth v2024  
**Dataset**: Split CIFAR-100 (10 tasks, 10 classes each)

---

## Experimental Settings

### Common Hyperparameters (Both Methods)

| Parameter                   | Value    | Source                            |
| --------------------------- | -------- | --------------------------------- |
| **Buffer Size**             | 500      | Standard for CIFAR-100            |
| **Alpha (MSE weight)**      | 0.3      | DER++ config                      |
| **Beta (CE replay weight)** | 0.5      | DER++ config                      |
| **Epochs per Task**         | 5        | For stable validation             |
| **Batch Size**              | 32       | Standard                          |
| **Learning Rate**           | 0.005    | **Causal-DER stable** (6x lower)  |
| **Weight Decay**            | 0.0      | DER++ verified                    |
| **Gradient Clip**           | 1.0      | **Causal-DER stable** (10x lower) |
| **Mixed Precision**         | Disabled | For stability                     |
| **Seed**                    | 42       | Reproducibility                   |

### Causal-DER Specific Features

| Feature                   | Status      | Description                                       |
| ------------------------- | ----------- | ------------------------------------------------- |
| **SCM Importance**        | ✅ Enabled  | Structural causal model for sample prioritization |
| **Causal Graph Learning** | ✅ Enabled  | Discover task-to-task relationships               |
| **Intervention Sampling** | ✅ Enabled  | Do-calculus based replay                          |
| **Enhanced IRM**          | ❌ Disabled | For stability                                     |
| **VAE Counterfactuals**   | ❌ Disabled | For stability                                     |
| **Neural NOTEARS**        | ❌ Disabled | For stability                                     |
| **Adaptive Controller**   | ❌ Disabled | For stability                                     |
| **Task-Free Streaming**   | ❌ Disabled | For stability                                     |

---

## Results

### Final Accuracy (After Task 10)

| Method                | Class-IL (%) | Task-IL (%) | Improvement              |
| --------------------- | ------------ | ----------- | ------------------------ |
| **DER++ (Baseline)**  | 14.92        | 61.45       | -                        |
| **Causal-DER (Ours)** | 0.90         | 11.25       | **-14.02% / -50.20%** ❌ |

### Per-Task Accuracy (Class-IL)

| Task        | DER++ | Causal-DER | Delta         |
| ----------- | ----- | ---------- | ------------- |
| Task 1      | 7.50  | 6.00       | -1.50         |
| Task 2      | 5.30  | 0.00       | -5.30         |
| Task 3      | 11.20 | 0.00       | -11.20        |
| Task 4      | 4.30  | 0.00       | -4.30         |
| Task 5      | 9.90  | 0.00       | -9.90         |
| Task 6      | 9.30  | 0.00       | -9.30         |
| Task 7      | 12.60 | 0.00       | -12.60        |
| Task 8      | 4.80  | 0.00       | -4.80         |
| Task 9      | 8.50  | 0.00       | -8.50         |
| Task 10     | 75.80 | 3.00       | -72.80        |
| **Average** | 14.92 | 0.90       | **-14.02** ❌ |

### 🚨 CRITICAL FINDING: Causal-DER UNDERPERFORMS DER++ Baseline

**Result**: Causal-DER achieves **0.90% Class-IL** vs DER++ **14.92%** = **16.6x WORSE performance**

**Root Cause Analysis:**

1. **No Gradient Clipping in DER++**: Standard Mammoth DER++ lacks `--clip_grad` argument

   - Causal-DER has `clip_grad=1.0` (very strict)
   - DER++ has NO gradient clipping → more aggressive optimization
   - This may actually HELP DER++ converge faster in 5 epochs

2. **Task 10 Dominance**: Both methods show recency bias but DER++ is much better:

   - DER++: 75.8% on Task 10 (current task)
   - Causal-DER: 3.0% on Task 10 (current task)
   - **DER++ is 25x better even on the CURRENT task!**

3. **Potential Issues in Causal-DER**:
   - ❌ Causal importance sampling may HARM performance (filtering useful samples)
   - ❌ MIR sampling overhead without benefit in 5 epochs
   - ❌ Temperature-scaled KL may be too soft (T=2.0)
   - ❌ Feature distillation disabled but still overhead?
   - ❌ Causal graph learning consuming compute without value

**Hypothesis**: Causal mechanisms ADD OVERHEAD without adding VALUE in low-epoch regime

### Forgetting Metrics

| Metric                 | DER++  | Causal-DER | Improvement |
| ---------------------- | ------ | ---------- | ----------- |
| **Average Forgetting** | [FILL] | [FILL]     | [FILL]      |
| **Maximum Forgetting** | [FILL] | [FILL]     | [FILL]      |

---

## Training Stability

### Causal-DER Diagnostics

| Metric                       | Value          | Status |
| ---------------------------- | -------------- | ------ |
| **NaN Detected**             | 0 (None!)      | ✅     |
| **Training Completed**       | Yes            | ✅     |
| **Total Training Time**      | ~90 min        | -      |
| **Samples Stored**           | 500 (full)     | ✅     |
| **Causal Graph Learned**     | Yes (30 edges) | ✅     |
| **Harmful Samples Filtered** | 0              | -      |
| **Avg Causal Importance**    | 0.4393         | -      |
| **Buffer Distribution**      | 50 per task    | ✅     |

---

## Analysis

### Key Findings

1. **Stability**: ✅ **YES! Training completed without ANY NaN errors** after fixing hyperparameters (LR=0.005, clip_grad=1.0)

2. **Performance**: ❌ **CRITICAL FAILURE: Causal-DER is 16.6x WORSE than DER++ baseline**

   - DER++: 14.92% Class-IL, 61.45% Task-IL
   - Causal-DER: 0.90% Class-IL, 11.25% Task-IL
   - Even on Task 10 (current): DER++ 75.8% vs Causal-DER 3.0% (25x worse!)

3. **Causal Features**: ✅ **Technically working** but ❌ **HARMING performance**

   - Causal graph learned (30 edges) but adds overhead
   - SCM importance filtering may discard useful samples
   - MIR sampling overhead without benefit

4. **Root Cause**: Gradient clipping TOO STRICT (1.0) + causal overhead + 5 epochs insufficient
   - DER++ has NO gradient clipping → faster convergence
   - Causal mechanisms need more epochs to show benefit

### Observations

**Comparison Analysis (DER++ vs Causal-DER):**

| Aspect            | DER++ Baseline | Causal-DER     | Winner   |
| ----------------- | -------------- | -------------- | -------- |
| Class-IL Accuracy | 14.92%         | 0.90%          | DER++ ✅ |
| Task-IL Accuracy  | 61.45%         | 11.25%         | DER++ ✅ |
| Task 10 (current) | 75.80%         | 3.00%          | DER++ ✅ |
| Training Time     | ~53 min        | ~90 min        | DER++ ✅ |
| Gradient Clipping | None           | 1.0 (strict)   | ?        |
| Complexity        | Simple         | Complex        | DER++ ✅ |
| NaN Stability     | No issues      | Required fixes | DER++ ✅ |

**Critical Issues Identified:**

1. **Causal Overhead Without Benefit:**

   - 30-edge causal graph learned but adds 40% training time overhead
   - SCM importance scoring (avg 0.44) may be filtering GOOD samples
   - MIR candidate selection (3x pool) adds compute without value

2. **Gradient Clipping Too Strict:**

   - clip_grad=1.0 was needed to prevent NaN, but may be TOO conservative
   - DER++ converges faster with NO clipping
   - Need to find middle ground (e.g., clip_grad=5.0?)

3. **Current Task Performance Disaster:**
   - DER++ learns Task 10 well (75.8%)
   - Causal-DER FAILS on Task 10 (3.0%) despite just training on it!
   - Suggests fundamental issue in loss computation or optimization

**Hypothesis for Poor Performance:**

- Causal importance sampling DISCARDS high-loss samples (needed for learning)
- Temperature=2.0 makes KL loss too soft (weak distillation signal)
- Strict gradient clipping prevents model from learning current task
- Feature distillation weight=0.05 may interfere even though features not stored

---

## 🔧 Action Plan to Fix Performance

### Immediate Fixes (Priority 1 - Critical)

1. **Relax Gradient Clipping**: Change from 1.0 → 5.0 or 10.0

   - Current: Too conservative, prevents learning
   - DER++ has NONE and works better
   - Target: Find sweet spot between stability and performance

2. **Disable Causal Importance Sampling**: Set `use_causal_sampling=0`

   - Hypothesis: Filtering is DISCARDING useful hard samples
   - Test pure DER++ with our engine's improvements only
   - Ablation: Does causal help or hurt?

3. **Lower Temperature**: Change from 2.0 → 1.0

   - Hypothesis: Too soft, weak distillation signal
   - Standard DER++ uses MSE (equivalent to T=1.0 for KL)

4. **Remove Feature Distillation Overhead**: Set `feature_kd_weight=0.0` (already done)
   - Ensure no feature hooks running
   - Check if `store_features=0` actually disables overhead

### Medium Priority Fixes (Priority 2 - Testing)

5. **Increase Epochs**: 5 → 50 epochs (standard benchmark)

   - Causal mechanisms may need more time to show value
   - Fair comparison with literature results

6. **Disable MIR Sampling**: Set `use_mir_sampling=0`

   - Hypothesis: 3x candidate pool overhead without benefit
   - Test if MIR helps or hurts in low-epoch regime

7. **Increase Alpha/Beta**: From 0.3/0.5 → 0.5/0.5 or 0.7/0.5
   - Stronger replay signal may help retention

### Debug Experiments (Priority 3 - Understanding)

8. **Pure DER++ Replication**: Disable ALL causal features

   ```bash
   --use_causal_sampling 0
   --use_mir_sampling 0
   --temperature 1.0
   --clip_grad 10.0
   --feature_kd_weight 0.0
   ```

   - Should match standard DER++ performance
   - If not, there's a bug in the engine

9. **Ablation Study**: Test each feature independently

   - Baseline: Pure DER++
   - +Causal sampling only
   - +MIR only
   - +Both (current config)

10. **Gradient Analysis**: Log gradient norms per module
    - Is clipping actually being triggered?
    - Which modules have large gradients?

---

## Updated Next Steps for Paper

1. ❌ ~~Run DER++ baseline~~ (COMPLETED - shows Causal-DER underperforms 16.6x)
2. 🔧 **FIX CRITICAL ISSUES** (gradient clip, causal sampling, temperature)
3. 🔄 **Re-run 5-epoch with fixes** (expect ~15% Class-IL, matching DER++)
4. ✅ **If fixed**: Extend to 50 epochs for SOTA claim
5. ❌ **If still broken**: Debug engine, may need to revert to standard DER++

**DO NOT proceed with 50-epoch run until 5-epoch matches DER++ baseline!**

---

## Paper-Ready Claims (Updated After Baseline Comparison)

### ❌ CURRENT STATUS: NOT READY FOR PUBLICATION

**Critical Finding**: Causal-DER **underperforms** DER++ by 16.6x (0.90% vs 14.92%)

### What You CANNOT Claim (Until Fixed):

❌ "Causal-DER improves over DER++" (currently FALSE - performs WORSE)  
❌ "Causal importance sampling helps retention" (appears to HURT)  
❌ "The method achieves competitive performance" (16.6x worse than baseline)  
❌ "Causal reasoning benefits continual learning" (no evidence yet)

### What You CAN Claim (Negative Results):

✅ "We implemented structural causal models for replay buffer management"  
✅ "Causal graph learning is computationally feasible during training"  
✅ "Initial results show causal overhead without performance benefit in 5-epoch regime"  
✅ "Gradient clipping requirements differ between standard and causal-enhanced methods"

### Honest Framing (If Publishing Negative Results):

```
"We investigated whether structural causal models can improve
continual learning replay strategies. On Split CIFAR-100 with
5-epoch training, our Causal-DER approach achieved 0.90% Class-IL
accuracy compared to DER++'s 14.92%, suggesting that:

1. Causal importance filtering may discard useful samples
2. Additional computational overhead (40% slower) does not justify
   the complexity in low-epoch regimes
3. Further investigation needed with extended training (50 epochs)
   and ablation studies

We provide this as a cautionary tale about the challenges of
integrating causal reasoning into deep learning systems."
```

### Path Forward:

**Option A - Fix and Retry:**

1. Debug issues (gradient clip, sampling strategy, temperature)
2. Re-run with fixes targeting DER++ parity
3. If successful, extend to 50 epochs and claim improvement

**Option B - Pivot to Ablation Study:**

1. Accept that naive causal integration doesn't work
2. Systematic ablation to find WHAT specifically hurts
3. Publish lessons learned as "What NOT to do"

**Option C - Abandon and Simplify:**

1. Strip out all causal features
2. Focus on simple improvements to DER++ (better sampling, temperature tuning)
3. Claim modest incremental gains (safer bet)

---

## Revised Research Plan

### Phase 1: Debug & Fix (THIS WEEK)

1. ✅ **Complete 5-epoch validation** (DONE - found critical issues)
2. 🔧 **Fix critical bugs** (gradient clip, causal sampling, temperature)
3. 🔄 **Re-run 5-epoch** (target: match DER++ ~15% Class-IL)
4. 🔍 **Ablation study** (isolate what hurts performance)

### Phase 2: Extended Validation (IF PHASE 1 SUCCEEDS)

5. ⏳ **Extend to 50 epochs** (for SOTA claim)
6. ⏳ **Run multiple seeds** (42, 43, 44 for statistical significance)
7. ⏳ **Additional datasets** (Split CIFAR-10, Tiny ImageNet)

### Phase 3: Publication (IF PHASE 2 SUCCEEDS)

8. ⏳ **Write full paper** (ICLR/CVPR submission)
9. ⏳ **Code release** (GitHub with reproducibility)

### STOP CONDITIONS:

❌ **If Phase 1 fails** (can't match DER++ baseline):

- Publish negative results as workshop paper
- OR pivot to simpler improvements
- OR abandon causal approach entirely

❌ **If Phase 2 shows no improvement over DER++**:

- Honest reporting: "Causal doesn't help"
- Educational contribution: "Lessons learned"

---

## Target Venues

### With 5-Epoch Results:

| Venue              | Acceptance Likelihood | Notes                       |
| ------------------ | --------------------- | --------------------------- |
| **CVPR Workshop**  | High                  | Accepts preliminary work    |
| **ICCV Workshop**  | High                  | Continual Learning Workshop |
| **arXiv Preprint** | Guaranteed            | No peer review              |

### With 50-Epoch Results:

| Venue           | Acceptance Likelihood | Notes               |
| --------------- | --------------------- | ------------------- |
| **ICLR**        | Medium                | Need strong results |
| **CVPR (Main)** | Medium                | Need ablations      |
| **NeurIPS**     | Low-Medium            | Very competitive    |

---

## Citation (Draft)

```bibtex
@inproceedings{yourname2025causalder,
  title={Causal-DER: Structural Causal Models for Continual Learning},
  author={Your Name and Collaborator Names},
  booktitle={CVPR Workshop on Continual Learning},
  year={2026}
}
```

---

## Commands Reference

### Run 5-Epoch Experiment:

```bash
cd /Users/zulhilmirahmat/Development/programming/Symbio\ AI
chmod +x run_5epoch_stable.sh
./run_5epoch_stable.sh
```

### Check Results:

```bash
# Causal-DER results
tail -50 ~/causal_der_5epoch_stable.log | grep -E "Class-IL|Task-IL"

# DER++ baseline
tail -50 ~/derpp_5epoch_baseline.log | grep -E "Class-IL|Task-IL"
```

### Extract Final Numbers:

```bash
# Class-IL accuracy
grep "Class-IL" ~/causal_der_5epoch_stable.log | tail -1
grep "Class-IL" ~/derpp_5epoch_baseline.log | tail -1

# Task-IL accuracy
grep "Task-IL" ~/causal_der_5epoch_stable.log | tail -1
grep "Task-IL" ~/derpp_5epoch_baseline.log | tail -1
```

---

## Log Files

- **Causal-DER**: `~/causal_der_5epoch_stable.log`
- **DER++ Baseline**: `~/derpp_5epoch_baseline.log`

---

## Status

- [x] Experiment started
- [x] Causal-DER completed (0.90% Class-IL - UNDERPERFORMS)
- [x] DER++ baseline completed (14.92% Class-IL - 16.6x BETTER)
- [x] Results extracted and compared
- [x] Analysis completed - CRITICAL ISSUES IDENTIFIED
- [ ] **FIXES APPLIED** (gradient clip, causal sampling, temperature)
- [ ] **RE-RUN with fixes** (target: ~15% Class-IL)
- [ ] Paper draft written (BLOCKED until performance fixed)

---

## 🚨 CRITICAL DECISION POINT

**Current Status**: Causal-DER is **BROKEN** - performs 16.6x worse than baseline

**Options:**

1. **Fix and retry** (recommended) - Apply Priority 1 fixes and re-run
2. **Debug deeply** - Systematic ablation to find root cause
3. **Abandon causal** - Revert to simple DER++ improvements
4. **Publish negative** - Submit to workshop as "lessons learned"

**Recommendation**: Apply fixes from Priority 1 list and re-run before proceeding with any 50-epoch experiments.

---

**Last Updated**: October 19, 2025 23:01 UTC  
**Experiment Status**: ❌ FAILED - Need critical fixes before proceeding  
**Next Action**: Apply Priority 1 fixes and re-run 5-epoch validation
