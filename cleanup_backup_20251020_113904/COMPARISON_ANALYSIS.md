# DER++ vs Causal-DER: Comparison Analysis

**Date**: October 20, 2025  
**Status**: 🚨 **CRITICAL ISSUE FOUND** - Causal-DER produces 0% accuracy

---

## 📊 Results Summary

### Your Mammoth DER++ Run (2 epochs, lr=0.03, momentum=0.9)
```
Final Results after 10 tasks:
  Class-IL: 15.36%
  Task-IL:  56.06%

Task-by-task Task-IL accuracy:
[55.3, 44.8, 51.9, 47.3, 54.5, 56.3, 54.8, 58.4, 62.3, 75.0]
```

### Causal-DER with ALL Features Disabled (2 epochs, lr=0.03, momentum=0.9)
```
Final Results after 10 tasks:
  Class-IL: 0.0%
  Task-IL:  0.0%

Task-by-task Task-IL accuracy:
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

**Verdict**: ❌ **CATASTROPHIC FAILURE** - Causal-DER engine produces 0% accuracy even with all causal features disabled

---

## 🔬 Official Mammoth Hyperparameters (from `utils/best_args.py`)

### DER++ on seq-cifar100 (buffer_size=500)
```python
{
    'lr': 0.03,
    'optim_mom': 0,        # ← NO MOMENTUM
    'optim_wd': 0,         # ← NO WEIGHT DECAY
    'alpha': 0.1,          # ← Official alpha (you used 0.3)
    'beta': 0.5
}
```

### Your Test Runs Used
| Setting | DER++ Baseline | Causal-DER Test |
|---------|----------------|-----------------|
| **lr** | 0.03 ✅ | 0.03 ✅ |
| **momentum** | **0.9** ⚠️ | **0.9** ⚠️ |
| **weight_decay** | 0.0 ✅ | 0.0 ✅ |
| **alpha** | **0.3** ⚠️ | **0.3** ⚠️ |
| **beta** | 0.5 ✅ | 0.5 ✅ |
| **n_epochs** | 2 ⚠️ | 2 ⚠️ |

**Key Differences**:
1. ✅ You matched lr=0.03 correctly
2. ⚠️ You used momentum=0.9 (official is 0.0)
3. ⚠️ You used alpha=0.3 (official is 0.1)
4. ⚠️ You ran 2 epochs (official is 50)

---

## 📚 Are Your Results Matching the Published Paper?

### Short Answer
**Your DER++ run (15.36% Class-IL, 56.06% Task-IL) is NOT directly comparable to published papers because:**
1. You only ran **2 epochs** instead of the standard **50 epochs**
2. You used **momentum=0.9** instead of **0.0**
3. You used **alpha=0.3** instead of **0.1**

### Expected Performance (from literature)
The DER++ paper and Mammoth benchmarks typically report results after **50 epochs** per task. For comparison:

- **DER++ on Split CIFAR-100 (50 epochs, buffer=500)**:
  - Class-IL: ~35-40% (approximate from various papers)
  - Task-IL: ~70-75% (approximate)

Your 2-epoch run achieved:
- Class-IL: 15.36% (~40-45% of expected full training)
- Task-IL: 56.06% (~75-80% of expected full training)

**This is actually REASONABLE for only 2 epochs!** Task-IL at 56% shows the model is learning the current task well.

---

## 🚨 The Critical Problem: Causal-DER Returns 0%

### What We Tested
We ran Causal-DER with **ALL causal features disabled**:
```bash
--use_causal_sampling 0
--use_mir_sampling 0
--importance_weight_replay 0
--use_enhanced_irm 0
--use_ate_pruning 0
--use_neural_causal_discovery 0
--use_counterfactual_replay 0
--use_task_free_streaming 0
--use_adaptive_controller 0
```

This should have behaved **identically** to vanilla DER++.

### What We Got
**0% accuracy on ALL tasks** - complete learning failure.

### What This Means
There is a **fundamental bug** in the Causal-DER engine that prevents it from learning, even when all causal features are disabled. Possible causes:

1. **Loss computation bug** in `training/causal_der.py`:
   - Cross-entropy loss may be computed incorrectly
   - KL distillation temperature handling may be broken
   - Logits may be sanitized incorrectly (nan_to_num converting valid gradients to zeros)

2. **Buffer storage/retrieval bug**:
   - Logits may be stored incorrectly (wrong shape, wrong values)
   - Buffer sampling may return corrupted data
   - Transform pipeline may be breaking the data

3. **Optimizer integration bug**:
   - Gradients may not be flowing correctly
   - Optimizer step may not be updating weights
   - Learning rate decay may be too aggressive

4. **Over-sanitization**:
   - `torch.nan_to_num` may be converting valid small logits to zero
   - Gradient clipping (clip_grad=10.0) may be too strict
   - Output clamping may be preventing learning

---

## 🔍 Diagnostic Observations

### From Causal-DER Logs
```
Task 7 - Epoch 2: 100%|██████████| 314/314 [03:14<00:00, 1.61it/s, loss=4.8, lr=0.03]
Task 8 - Epoch 2: 100%|██████████| 314/314 [03:14<00:00, 1.61it/s, loss=4.96, lr=0.03]
Task 9 - Epoch 2: 100%|██████████| 314/314 [03:15<00:00, 1.60it/s, loss=4.77, lr=0.03]
Task 10 - Epoch 2: 100%|██████████| 314/314 [03:18<00:00, 1.58it/s, loss=4.61, lr=0.03]
```

**Loss values hover around 4.6-5.2** throughout training:
- For 100-class classification, random guessing has loss ≈ ln(100) ≈ 4.605
- **The model is NOT learning at all** - it's stuck at random baseline!

### Causal Graph Learning (Even with Features Disabled!)
```
[INFO] 20-Oct-25 09:53:32 - Learning causal graph between tasks (offline)...
[INFO] 20-Oct-25 09:53:32 - Discovered causal graph with 42 edges
```

**This should NOT be happening!** With all causal features disabled, the causal graph learning should be skipped entirely. This suggests:
- The feature flags are not being respected
- Causal code is still running in the background
- This overhead may be interfering with learning

---

## 🔧 Immediate Action Plan

### Priority 1: Verify Basic Learning Works
Run a **minimal test** to check if the base model can learn at all:

1. **Disable ALL causal features** (already done)
2. **Remove all sanitization**:
   - Remove `torch.nan_to_num` from loss computation
   - Remove output clamping
   - Use standard PyTorch CE loss directly
3. **Test on Task 1 only** (1 epoch, 10 classes):
   ```bash
   # Expected: Should achieve >80% accuracy on Task 1 alone
   ```

### Priority 2: Compare Code Paths
1. **Read Mammoth's original `models/derpp.py`**
2. **Compare with `training/causal_der.py` line-by-line**
3. **Identify ALL differences** in:
   - Loss computation
   - Buffer storage/retrieval
   - Optimizer integration
   - Gradient flow

### Priority 3: Add Debug Logging
Insert logging in `training/causal_der.py` to track:
```python
logging.info(f"Step {step}: loss={loss.item():.4f}, "
             f"output_mean={outputs.mean():.4f}, "
             f"output_std={outputs.std():.4f}, "
             f"grad_norm={grad_norm:.4f}")
```

Track:
- Are outputs reasonable? (should vary, not constant)
- Are gradients flowing? (grad_norm > 0)
- Is loss decreasing? (should drop from ~4.6 toward 0)

---

## 📋 Next Steps

### Recommended Order:
1. ✅ **(DONE)** Run Mammoth DER++ baseline → **Works! (15.36% / 56.06%)**
2. ✅ **(DONE)** Run Causal-DER with features disabled → **FAILED! (0% / 0%)**
3. **🔜 (DO NEXT)** Debug why Causal-DER returns 0%:
   - Option A: Remove all sanitization and re-run
   - Option B: Compare code with Mammoth's `models/derpp.py`
   - Option C: Add verbose logging to track loss/gradients
4. **Fix the bug** and re-run
5. **If fixed**: Run full 50-epoch experiment
6. **If not fixed**: Consider reverting to standard Mammoth DER++

---

## 📝 Conclusion

### What We Learned
1. ✅ **Mammoth DER++ works correctly** (15.36% Class-IL in 2 epochs is reasonable)
2. ❌ **Causal-DER has a critical bug** that prevents ANY learning (0% accuracy)
3. ⚠️ **Your hyperparameters don't match the official paper**:
   - Use momentum=0.0 (not 0.9)
   - Use alpha=0.1 (not 0.3)
   - Run 50 epochs (not 2)

### Recommendation
**DO NOT proceed with 50-epoch runs until Causal-DER can match Mammoth DER++ at 2 epochs.**

The 0% result is not a hyperparameter issue - it's a **fundamental bug** in the engine that must be fixed before any further experimentation.

---

## 🔗 References

- Mammoth Official Best Args: `mammoth/utils/best_args.py` line 861-904
- DER++ Paper: Buzzega et al., "Dark Experience for General Continual Learning", NeurIPS 2020
- X-DER Paper: Boschini et al., "Class-Incremental Continual Learning into the eXtended DER-verse", TPAMI 2022

---

**Last Updated**: October 20, 2025  
**Status**: 🔴 **BLOCKED** - Critical bug in Causal-DER engine prevents learning
