# Causal-DER Fix Results & Comparison

**Date**: October 20, 2025  
**Status**: 🔧 **PARTIALLY FIXED** - Causal-DER now runs without crashing, but performance is still far below DER++

---

## 📊 Results Comparison

### Mammoth DER++ Baseline (2 epochs)
```
Hyperparameters:
  lr: 0.03
  momentum: 0.9
  alpha: 0.3
  beta: 0.5
  n_epochs: 2
  buffer_size: 500

Final Results after 10 tasks:
  Class-IL: 15.36%
  Task-IL:  56.06%

Task-by-task Task-IL accuracy:
[55.3, 44.8, 51.9, 47.3, 54.5, 56.3, 54.8, 58.4, 62.3, 75.0]
```
✅ **WORKING** - Model learns successfully

---

### Causal-DER BEFORE Fixes (2 epochs, all features disabled)
```
Hyperparameters:
  lr: 0.03
  momentum: 0.9
  alpha: 0.3
  beta: 0.5
  n_epochs: 2
  buffer_size: 500
  [ALL causal features disabled]

Final Results after 10 tasks:
  Class-IL: 0.0%
  Task-IL:  0.0%

Loss values: 4.6-5.2 (stuck at random baseline)
```
❌ **CATASTROPHIC FAILURE** - Complete learning failure

---

### Causal-DER AFTER Fixes (1 epoch, all features disabled)
```
Hyperparameters:
  lr: 0.03
  momentum: 0.9
  alpha: 0.3
  beta: 0.5
  n_epochs: 1
  buffer_size: 500
  buffer_dtype: float32  ← CRITICAL FIX
  store_logits_as: logits32  ← CRITICAL FIX
  [ALL causal features disabled]

Final Results after 10 tasks:
  Class-IL: 1.0%
  Task-IL:  10.0%

Task-by-task Task-IL accuracy:
[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

Loss values during training: 8.79 → 2.0 (learning!)
```
⚠️ **RUNS BUT PERFORMS POORLY** - Model learns during training but final accuracy is still random (10%)

---

## 🔧 Fixes Applied

### Fix #1: Added Pure DER++ Bypass
**Problem**: Complex Causal-DER code path ran even with all features disabled  
**Solution**: Added bypass at start of `compute_loss()` to use simple DER++ implementation when all causal features are disabled

```python
# Check if ALL causal features are disabled
use_pure_derpp = (
    not self.use_causal_sampling and
    not self.use_mir_sampling and
    not self.use_counterfactual_replay and
    not self.use_neural_causal_discovery and
    not self.use_task_free_streaming and
    not self.use_enhanced_irm and
    self.feature_kd_weight == 0.0
)

if use_pure_derpp:
    # SIMPLE DER++ IMPLEMENTATION
    # Loss = CE(current) + alpha * MSE(replay_logits) + beta * CE(buffer)
    ...
```

**Result**: ✅ Bypasses complex causal code path successfully

---

### Fix #2: Fixed Float16 Precision Issues
**Problem**: Buffer stored data as float16 by default, causing NaN when reloaded  
**Solution**: Switched to float32 storage for both data and logits

```bash
--buffer_dtype float32
--store_logits_as logits32
```

**Result**: ✅ Eliminated NaN errors during replay

---

### Fix #3: Added Missing Attributes
**Problem**: `AttributeError: 'CausalDEREngine' object has no attribute 'current_task_id'`  
**Solution**: Added initialization in `__init__`:

```python
self.global_step = 0
self.current_task_id = -1  # Track current task
self.task_step = 0         # Steps within current task
```

**Result**: ✅ No more AttributeError

---

### Fix #4: Added NaN Safety Guards
**Problem**: Model outputs on replay data sometimes contained NaN  
**Solution**: Added safety checks before MSE loss:

```python
# Safety check: if buffer logits contain NaN/Inf, skip replay
if torch.isnan(buf_logits_mse).any() or torch.isinf(buf_logits_mse).any():
    print(f"[ERROR] NaN/Inf detected in stored logits from buffer!")
    return current_loss, info

# Safety check: if model outputs contain NaN/Inf, skip replay
if torch.isnan(buf_outputs_mse).any() or torch.isinf(buf_outputs_mse).any():
    print(f"[ERROR] NaN/Inf detected in model outputs on replay data!")
    return current_loss, info
```

**Result**: ✅ Gracefully handles NaN cases

---

## 🔍 Analysis: Why Still 10% (Random) Performance?

### Observations from Training Logs

1. **Loss DOES decrease during training**:
   ```
   Task 2 - Epoch 1: loss=8.79 → 2.0
   ```
   This proves the model IS learning within each task!

2. **But final accuracy is random (10%)**:
   ```
   Task-IL: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
   ```
   10% = 1/10 classes = random guessing for 10-class problems

3. **Buffer is nearly empty**:
   ```
   Buffer size: 10/500
   Samples per task: {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}
   ```
   Only 1 sample per task stored! This is WAY too few for effective replay.

### Root Cause Hypothesis

**The buffer is not being filled properly!**

In DER++, the `store()` method should be called after EVERY batch to add samples to the buffer. With:
- 10 tasks
- ~5000 samples per task
- Batch size 32
- Buffer capacity 500

We should have ~50 samples per task in the buffer, not just 1!

**Possible causes**:
1. The `store()` method is not being called correctly
2. The reservoir sampling is rejecting too many samples
3. There's a bug in the buffer add logic

---

## 🔧 Required Fix: Investigate Buffer Storage

### Next Steps

1. **Add logging to `store()` method**:
   ```python
   def store(self, data, target, logits, task_id, model=None):
       logging.info(f"Store called: batch_size={data.size(0)}, task_id={task_id}, "
                   f"buffer_size={len(self.buffer.buffer)}/{self.buffer.capacity}")
       ...
   ```

2. **Check if `store()` is being called**:
   - Should be called ~157 times per task (5000 samples / 32 batch_size)
   - Should add ~50 samples per task to buffer

3. **Check Mammoth's `models/derpp.py`** to see how it calls `store()`:
   ```python
   # In observe() method:
   self.buffer.add_data(examples=inputs, labels=labels, logits=outputs.data)
   ```

4. **Compare with Causal-DER's approach** in `mammoth/models/causal_der.py`

---

## 📊 Performance Gap Analysis

| Metric | DER++ (2 epochs) | Causal-DER Fixed (1 epoch) | Gap |
|--------|------------------|---------------------------|-----|
| **Class-IL** | 15.36% | 1.0% | **-14.36%** ❌ |
| **Task-IL** | 56.06% | 10.0% | **-46.06%** ❌ |
| **Loss (final)** | ~2.5 | ~2.0 | Similar ✅ |
| **Buffer usage** | ~500/500 | **10/500** ❌ |

**Key Finding**: The loss values are similar, but the buffer is nearly empty in Causal-DER!

---

## 🎯 Immediate Action Items

### Priority 1: Fix Buffer Storage
- [x] Causal-DER runs without crashing
- [ ] **Debug why only 10 samples stored (should be ~500)**
- [ ] Compare `store()` implementation with Mammoth DER++
- [ ] Ensure `store()` is called after every training batch

### Priority 2: Match DER++ Exactly
- [ ] Verify double buffer sampling (MSE batch ≠ CE batch)
- [ ] Verify logits are stored correctly (no corruption)
- [ ] Verify importance weighting is disabled (should be uniform)

### Priority 3: Run Fair Comparison
Once buffer is fixed:
- [ ] Run Causal-DER for 2 epochs (match DER++ baseline)
- [ ] Compare Task-IL (target: ~56% like DER++)
- [ ] If matched: Proceed with causal features enabled
- [ ] If not matched: Continue debugging

---

## 💡 Key Insights

### What Works Now ✅
1. Pure DER++ bypass successfully routes to simple loss computation
2. Float32 storage prevents NaN errors
3. Loss decreases during training (proves learning works)
4. No crashes or AttributeErrors

### What's Still Broken ❌
1. **Buffer is nearly empty** (10/500 instead of ~500/500)
2. **Final accuracy is random** (10% = random guessing)
3. **No retention across tasks** (each task starts from scratch)

### What This Tells Us
The **core learning mechanism works** (loss decreases), but the **replay mechanism is broken** (buffer not filled, no continual learning).

---

## 🔗 References

- **Original Issue**: COMPARISON_ANALYSIS.md
- **DER++ Paper**: Buzzega et al., "Dark Experience for General Continual Learning", NeurIPS 2020
- **Mammoth Code**: `mammoth/models/derpp.py` (reference implementation)
- **Causal-DER Code**: `training/causal_der.py` (our implementation)

---

**Status**: 🟡 **IN PROGRESS**  
**Next Action**: Debug buffer storage to understand why only 10/500 samples are stored

---

## 📝 Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Crashes** | ✅ Fixed | No more NaN errors or AttributeErrors |
| **Training Loss** | ✅ Working | Loss decreases from 8.79 → 2.0 |
| **Buffer Storage** | ❌ Broken | Only 10/500 samples stored (should be ~500) |
| **Continual Learning** | ❌ Broken | 10% accuracy = random guessing |
| **Performance vs DER++** | ❌ Far below | 10% vs 56% Task-IL (-46 points) |

**Verdict**: The fixes allow the code to run, but the replay mechanism is still fundamentally broken. The buffer is not being filled properly, preventing continual learning.
