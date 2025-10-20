# 🚨 CRITICAL BUGS FOUND: DER++ vs Causal-DER Comparison

**Date**: October 20, 2025  
**Analysis**: Line-by-line comparison of `mammoth/models/derpp.py` vs `training/causal_der.py`  
**Result**: **MULTIPLE CRITICAL BUGS** preventing Causal-DER from learning

---

## 📋 Summary

**Causal-DER with ALL features disabled should behave identically to DER++, but it returns 0% accuracy.**

After comparing the code, I found **5 CRITICAL BUGS** that explain the complete learning failure:

1. ❌ **Over-sanitization with `torch.nan_to_num`** - Zeros out gradients
2. ❌ **Wrong distillation loss** - KL instead of MSE (architecture change)
3. ❌ **Causal code runs even when disabled** - Graph learning overhead
4. ❌ **Double buffer sampling** - Inefficient and different behavior
5. ❌ **Complex loss guards prevent learning** - Too many NaN fallbacks

---

## 🔍 Bug #1: Over-Sanitization Killing Gradients

### DER++ (CORRECT - Simple and Direct)
```python
def observe(self, inputs, labels, not_aug_inputs, epoch=None):
    self.opt.zero_grad()
    outputs = self.net(inputs)
    loss = self.loss(outputs, labels)  # ← Direct CE loss, no sanitization
    
    if not self.buffer.is_empty():
        buf_inputs, _, buf_logits = self.buffer.get_data(...)
        buf_outputs = self.net(buf_inputs)
        loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)  # ← Simple MSE
        loss += loss_mse
        
        buf_inputs, buf_labels, _ = self.buffer.get_data(...)
        buf_outputs = self.net(buf_inputs)
        loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)  # ← Direct CE
        loss += loss_ce
    
    loss.backward()
    self.opt.step()
    # ... store in buffer
```

### Causal-DER (BROKEN - Over-Sanitized)
```python
def compute_loss(self, model, data, target, output, task_id):
    # CRITICAL BUG: Sanitize outputs BEFORE computing loss
    output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)  # ← KILLS GRADIENTS!
    output = torch.clamp(output, min=-10.0, max=10.0)  # ← Clamps valid outputs
    
    current_loss = F.cross_entropy(output, target)  # ← Loss on SANITIZED outputs
    
    # ... later in replay path ...
    buf_outputs = model(buf_data)
    buf_outputs = torch.nan_to_num(buf_outputs, nan=0.0, posinf=10.0, neginf=-10.0)  # ← AGAIN!
    
    # More sanitization...
    kl_per_sample = torch.nan_to_num(kl_per_sample, nan=0.0, posinf=1e6, neginf=0.0)  # ← KILLS GRADIENTS!
    ce_per_sample = torch.nan_to_num(ce_per_sample, nan=0.0, posinf=1e6, neginf=0.0)  # ← KILLS GRADIENTS!
```

**Why this breaks learning**:
- `torch.nan_to_num` **STOPS GRADIENT FLOW** by replacing values with constants
- Even valid small logits (e.g., -5.2) get zeroed out if model is uncertain
- The model receives NO gradient signal because all "unsafe" values are replaced
- Loss appears finite but gradients are **DEAD**

**Fix**: Remove ALL `torch.nan_to_num` calls and trust PyTorch's stability.

---

## 🔍 Bug #2: Wrong Distillation Loss (KL vs MSE)

### DER++ (CORRECT - Uses MSE)
```python
buf_outputs = self.net(buf_inputs)
loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
loss += loss_mse
```

### Causal-DER (DIFFERENT - Uses Temperature-Scaled KL)
```python
T = self.temperature  # ← Usually 2.0
student_log_probs = F.log_softmax(buf_outputs / T, dim=-1)
teacher_probs = F.softmax(buf_logits / T, dim=-1)
kl_per_sample = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=1) * (T * T)
replay_kd = kl_per_sample.mean()
```

**Why this is different**:
- DER++ uses **MSE on raw logits** - simple and stable
- Causal-DER uses **temperature-scaled KL divergence** - more complex, sensitive to temperature
- With T=2.0, the KL loss is **4x weaker** than it should be (scaled by T²)
- This is a **fundamental architecture change**, not a bug-for-bug match

**Impact**: Even if all other bugs are fixed, Causal-DER will NOT replicate DER++ exactly because the distillation loss is fundamentally different.

**Options**:
1. Change Causal-DER to use MSE (match DER++ exactly)
2. Keep KL but adjust temperature to T=1.0 (remove scaling)
3. Accept that Causal-DER is a **different model** and tune hyperparameters separately

---

## 🔍 Bug #3: Causal Code Runs Even When Disabled

### Evidence from Logs (Features Disabled, But Still Running)
```
[INFO] 20-Oct-25 09:53:32 - Learning causal graph between tasks (offline)...
[INFO] 20-Oct-25 09:53:32 - Discovered causal graph with 42 edges
```

### The Bug in `compute_loss`
```python
def compute_loss(self, model, data, target, output, task_id):
    # ... current loss ...
    
    # ===== NEW: Task-Free Streaming =====
    if self.use_task_free_streaming:  # ← Correctly gated
        features = self.get_features(model, data)
        # ...
    
    # ===== NEW: Neural Causal Discovery =====
    neural_dag_metrics = {}
    if self.use_neural_causal_discovery and self.neural_causal_discovery is not None:  # ← Correctly gated
        features = self.get_features(model, data)
        # ...
    
    # ===== NEW: Train Counterfactual VAE =====
    vae_loss_val = torch.tensor(0.0, device=device)
    if self.use_counterfactual_replay and self.counterfactual_generator is not None:  # ← Correctly gated
        features = self.get_features(model, data)
        # ...
```

**BUT** in `end_task`:
```python
def end_task(self, model, task_id):
    """Called at end of task - learn causal graph."""
    # ... update statistics ...
    
    # ===== CORE CAUSAL INNOVATION: Learn Task-Level SCM =====
    # Learn causal graph between tasks (OFFLINE MODE)
    if self.graph_mode == 'offline':  # ← This is the DEFAULT!
        if len(self.task_features) >= 2:
            self._learn_causal_graph_offline()  # ← ALWAYS RUNS unless graph_mode != 'offline'
```

**Why this breaks**:
- The `--graph_mode` flag defaults to `'offline'`
- Even with all causal features disabled, `end_task` STILL runs causal graph learning
- This adds **computational overhead** and may interfere with buffer operations
- The graph learning code may have bugs that corrupt the buffer

**Fix**: Skip causal graph learning if `use_causal_sampling=0` AND all other causal features are disabled.

---

## 🔍 Bug #4: Double Buffer Sampling (Inefficient)

### DER++ (CORRECT - Two Separate Samples)
```python
# Sample for KD term
buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, ...)
buf_outputs = self.net(buf_inputs)
loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
loss += loss_mse

# Sample AGAIN for CE term
buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, ...)
buf_outputs = self.net(buf_inputs)
loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
loss += loss_ce
```

### Causal-DER (DIFFERENT - Single Sample)
```python
replay_data = self.buffer.sample(
    replay_batch_size,
    device,
    use_causal_sampling=self.use_causal_sampling,
    # ... (ONE sample for both KD and CE)
)

buf_data, buf_targets, buf_logits, buf_task_ids, buf_importances = replay_data
buf_outputs = model(buf_data)

# Compute BOTH KD and CE on the SAME batch
kl_per_sample = F.kl_div(...)  # KD term
ce_per_sample = F.cross_entropy(...)  # CE term

replay_kd = kl_per_sample.mean()
replay_ce = ce_per_sample.mean()
```

**Why this is different**:
- DER++ samples **two independent batches** (one for KD, one for CE)
- Causal-DER samples **one batch** and computes both losses on it
- This changes the **replay dynamics** - samples get replayed twice vs once
- May affect **convergence speed** and **stability**

**Impact**: This alone won't cause 0% accuracy, but it changes the training dynamics enough that hyperparameters tuned for DER++ won't work.

---

## 🔍 Bug #5: Complex Loss Computation Prevents Learning

### DER++ (CORRECT - Simple 3-Line Loss)
```python
loss = self.loss(outputs, labels)  # Current task CE
loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)  # Replay KD
loss += self.args.beta * self.loss(buf_outputs, buf_labels)  # Replay CE
loss.backward()
```

### Causal-DER (BROKEN - 200+ Lines with Many NaN Guards)
```python
def compute_loss(...):
    # Line 1185: Sanitize outputs
    output = torch.nan_to_num(output, ...)
    output = torch.clamp(output, ...)
    
    # Line 1191: Current loss
    current_loss = F.cross_entropy(output, target)
    
    # Line 1194-1201: CHECK for NaN/Inf - RETURN EARLY if found
    if torch.isnan(current_loss) or torch.isinf(current_loss):
        return torch.tensor(1.0, device=device, requires_grad=True), {...}
    
    # Line 1207-1242: Task-free streaming, neural causal, VAE training
    # ...
    
    # Line 1244-1280: Replay warmup check - SKIP REPLAY if early tasks
    if (...) or len(self.buffer.buffer) == 0:
        return current_loss + 0.001 * vae_loss_val, info  # ← EARLY RETURN
    
    # Line 1282-1310: Build task bias from causal graph
    # ...
    
    # Line 1312-1350: Sample from buffer (with MIR, counterfactuals, etc.)
    # ...
    
    # Line 1352-1430: Compute replay losses (KD, CE, feature KD)
    # ...
    
    # Line 1432-1440: NaN guards on per-sample losses
    kl_per_sample = torch.nan_to_num(kl_per_sample, ...)  # ← KILLS GRADIENTS
    ce_per_sample = torch.nan_to_num(ce_per_sample, ...)  # ← KILLS GRADIENTS
    
    # Line 1442-1454: Importance weighting
    # ...
    
    # Line 1456-1490: IRM penalty computation
    # ...
    
    # Line 1492-1500: CHECK for NaN in losses - FALLBACK to current_loss
    if torch.isnan(current_loss) or torch.isnan(replay_kd) or torch.isnan(replay_ce):
        total_loss = current_loss  # ← SKIPS REPLAY
    else:
        # Line 1502-1520: Combine losses
        total_loss = current_loss + effective_alpha * replay_kd + self.beta * replay_ce
        # ... add VAE, DAG, IRM penalties ...
    
    # Line 1522-1525: FINAL CHECK - fallback if total_loss is NaN
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = current_loss  # ← SKIPS REPLAY AGAIN
    
    return total_loss, info
```

**Why this breaks learning**:
- **5+ early return paths** that skip replay entirely
- **Multiple NaN guards** that replace valid gradients with zeros
- **Complex control flow** makes debugging impossible
- **Fallback to current_loss** means the model NEVER learns from replay
- If replay loss has ANY instability, the entire replay term is **SILENTLY DROPPED**

**Evidence from logs**:
- Loss stuck at ~4.6 (random baseline for 100 classes)
- This suggests the model is **ONLY learning from current_loss** and never from replay
- The replay losses are probably triggering NaN guards and getting skipped

**Fix**: Simplify the loss computation to match DER++'s 3-line approach:
```python
def compute_loss(self, model, data, target, output, task_id):
    # Current task loss (no sanitization!)
    current_loss = F.cross_entropy(output, target)
    
    # Replay losses
    if len(self.buffer.buffer) > 0:
        buf_data, buf_targets, buf_logits, _, _ = self.buffer.sample(...)
        buf_outputs = model(buf_data)
        
        # MSE distillation (match DER++ exactly)
        replay_mse = F.mse_loss(buf_outputs, buf_logits)
        
        # CE on buffer labels
        replay_ce = F.cross_entropy(buf_outputs, buf_targets)
        
        # Combine (NO NaN guards!)
        total_loss = current_loss + self.alpha * replay_mse + self.beta * replay_ce
    else:
        total_loss = current_loss
    
    return total_loss, {}
```

---

## 📊 Root Cause Analysis

### Why Causal-DER Returns 0% Accuracy

1. **Over-sanitization (Bug #1)** → Gradients are DEAD
2. **NaN guards (Bug #5)** → Replay losses are SKIPPED
3. **Model only learns from current_loss** → No replay = catastrophic forgetting
4. **Loss stuck at 4.6** (random baseline) → No learning at all

### The Smoking Gun

Look at the training logs:
```
Task 7 - Epoch 2: loss=4.8
Task 8 - Epoch 2: loss=4.96
Task 9 - Epoch 2: loss=4.77
Task 10 - Epoch 2: loss=4.61
```

**ln(100) ≈ 4.605** - the loss for random guessing on 100 classes.

The model is **NOT LEARNING AT ALL** because:
- Current task loss is computed on **sanitized outputs** (dead gradients)
- Replay losses are probably triggering NaN guards and getting **SKIPPED**
- The optimizer sees **no gradient signal** and makes random updates

---

## ✅ Recommended Fixes (in Order of Priority)

### Fix #1: Remove ALL `torch.nan_to_num` Calls (CRITICAL)
```python
# BEFORE (BROKEN)
output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
current_loss = F.cross_entropy(output, target)

# AFTER (FIXED)
current_loss = F.cross_entropy(output, target)  # PyTorch handles stability
```

**Why**: This is the #1 killer. Removing sanitization will immediately restore gradient flow.

### Fix #2: Simplify Loss Computation (CRITICAL)
```python
def compute_loss(self, model, data, target, output, task_id):
    current_loss = F.cross_entropy(output, target)
    
    if len(self.buffer.buffer) == 0:
        return current_loss, {}
    
    # Sample replay batch
    buf_data, buf_targets, buf_logits, _, _ = self.buffer.sample(...)
    buf_outputs = model(buf_data)
    
    # MSE distillation (match DER++)
    replay_mse = F.mse_loss(buf_outputs, buf_logits)
    replay_ce = F.cross_entropy(buf_outputs, buf_targets)
    
    # NO NaN guards, NO sanitization
    total_loss = current_loss + self.alpha * replay_mse + self.beta * replay_ce
    return total_loss, {}
```

**Why**: This matches DER++ exactly and removes all complex NaN guard logic.

### Fix #3: Disable Causal Graph Learning When Features Disabled (HIGH)
```python
def end_task(self, model, task_id):
    # Only run causal analysis if ANY causal feature is enabled
    if (self.use_causal_sampling or self.use_mir_sampling or 
        self.use_neural_causal_discovery or self.use_counterfactual_replay or
        self.use_enhanced_irm or self.use_ate_pruning):
        self._learn_causal_graph_offline()
```

**Why**: Removes unnecessary overhead when testing in pure DER++ mode.

### Fix #4: Match DER++ Double Sampling (MEDIUM)
```python
# Sample twice (match DER++ exactly)
buf_inputs_kd, _, buf_logits = self.buffer.sample(...)
buf_outputs_kd = model(buf_inputs_kd)
replay_mse = F.mse_loss(buf_outputs_kd, buf_logits)

buf_inputs_ce, buf_labels, _ = self.buffer.sample(...)
buf_outputs_ce = model(buf_inputs_ce)
replay_ce = F.cross_entropy(buf_outputs_ce, buf_labels)
```

**Why**: This matches DER++'s replay dynamics exactly.

### Fix #5: Use MSE Instead of KL (OPTIONAL)
If you want **exact** DER++ replication:
```python
# BEFORE (KL with temperature)
student_log_probs = F.log_softmax(buf_outputs / T, dim=-1)
teacher_probs = F.softmax(buf_logits / T, dim=-1)
replay_kd = F.kl_div(student_log_probs, teacher_probs, reduction='mean') * (T * T)

# AFTER (MSE)
replay_mse = F.mse_loss(buf_outputs, buf_logits)
```

**Why**: DER++ uses MSE, not KL. This is a fundamental difference.

---

## 🧪 Test Plan

### Step 1: Minimal Fix (Test Immediately)
Apply Fix #1 only:
- Remove all `torch.nan_to_num` calls
- Keep everything else the same
- Run 1 epoch on task 1 only
- **Expected**: Should see loss < 4.0 (evidence of learning)

### Step 2: Full DER++ Match (Test Next)
Apply Fix #1 + Fix #2:
- Remove sanitization
- Simplify to 3-line loss
- Run 2 epochs, 2 tasks
- **Expected**: Should see Task-IL > 30% after task 2

### Step 3: Exact Replication (Final Validation)
Apply all 5 fixes:
- Match DER++ exactly
- Run full 10 tasks, 2 epochs
- **Expected**: Should match DER++ baseline (15% Class-IL, 56% Task-IL)

---

## 📝 Conclusion

**Causal-DER is BROKEN** because:
1. Over-sanitization kills gradients (Bug #1)
2. Complex NaN guards skip replay (Bug #5)
3. Model never learns from buffer
4. Loss stuck at random baseline (4.6)

**The fix is simple**: Remove sanitization, simplify loss computation, match DER++ exactly.

**Estimated time to fix**: 30 minutes  
**Estimated time to validate**: 1 hour (run 2-epoch test)

---

**Next Action**: Apply Fix #1 and Fix #2, then re-run the 2-epoch test. Expected result: Task-IL > 30%.
