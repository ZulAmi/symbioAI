# 🎯 Clean Slate: DER++ to Causal-DER Roadmap

**Date:** October 20, 2025  
**Status:** ✅ Codebase cleaned, ready for rewrite  
**Strategy:** Build incrementally on proven DER++ baseline

---

## 📊 Current Status

### Cleaned Up ✅
- ❌ Removed 25+ experimental training modules
- ❌ Removed duplicate data/config directories  
- ❌ Removed failed experiment documentation
- ❌ Removed old test scripts and validation code
- ✅ Backed up everything to: `cleanup_backup_20251020_113904/`

### Preserved Core ✅
```
Symbio AI/
├── mammoth/                    # ✅ Official Mammoth framework (untouched)
│   ├── models/
│   │   ├── derpp.py           # ✅ Reference DER++ (56% Task-IL, WORKING)
│   │   └── causal_der.py      # ⚠️  Will update to use new engine
│   └── utils/
├── training/
│   ├── __init__.py            # ✅ Package init
│   └── causal_der.py          # ⚠️  Will rewrite as causal_der_v2.py
├── validation/                # ✅ Validation framework
├── requirements.txt           # ✅ Dependencies
├── README.md                  # ✅ Project docs
└── .vscode/                   # ✅ IDE settings
```

---

## 🚀 Implementation Plan

### Phase 1: Clean DER++ Baseline (Week 1)
**Goal:** Match Mammoth's DER++ performance exactly (56% Task-IL on seq-cifar100)

**Tasks:**
1. ✅ Clean up codebase (DONE)
2. 📝 Create `training/causal_der_v2.py`:
   - Copy exact loss from `mammoth/models/derpp.py` (3 lines)
   - Copy exact buffer implementation
   - Simple, clean, no sanitization
   - No causal features yet
3. 🧪 Test and verify:
   - Run: `python mammoth/utils/main.py --model causal-der ...`
   - Expected: ~56% Task-IL (matching DER++)
   - If matched: ✅ BASELINE ESTABLISHED

**Success Criteria:**
- [ ] causal_der_v2.py matches derpp.py performance (±2%)
- [ ] No NaN issues
- [ ] Buffer fills to 500/500
- [ ] Clean, readable code (<500 lines)

---

### Phase 2: Causal Importance Scoring (Week 2)
**Goal:** Add lightweight causal importance, measure impact

**What to Add:**
```python
# Compute importance during store()
importance = compute_simple_importance(
    prediction_confidence,  # How confident is the model?
    loss_value,            # How hard is this sample?
    task_novelty           # Is this a new pattern?
)
```

**Ablation Study:**
- Baseline: uniform importance = 1.0
- Test: causal importance weighting
- Measure: Does Task-IL improve? By how much?

**Expected Gain:** +1-2% Task-IL (modest improvement)

---

### Phase 3: Causal Sampling (Week 3)
**Goal:** Sample buffer based on causal importance

**What to Add:**
```python
# During buffer.sample()
if use_causal_sampling:
    probs = importances / importances.sum()
    indices = np.random.choice(len(buffer), size=batch_size, p=probs)
else:
    indices = np.random.choice(len(buffer), size=batch_size)  # uniform
```

**Ablation Study:**
- Test A: Uniform sampling (baseline)
- Test B: Importance-weighted sampling
- Test C: Temperature-scaled sampling (T=2.0)

**Expected Gain:** +2-3% Task-IL

---

### Phase 4: Causal Graph Learning (Week 4)
**Goal:** Discover task dependencies, bias replay toward helpful tasks

**What to Add:**
```python
# After each task
def learn_task_graph():
    # Measure: Does replaying Task i help with Task j?
    # Method: Simple correlation analysis
    # Output: NxN adjacency matrix (task dependencies)
    
# During sampling
def bias_toward_helpful_tasks(current_task, causal_graph):
    # Give higher weight to samples from causally-related tasks
    task_weights = causal_graph[current_task]
    return task_weights
```

**Ablation Study:**
- Test A: No task bias (baseline)
- Test B: Causal graph bias
- Measure: Does it reduce forgetting?

**Expected Gain:** +3-5% Task-IL

---

### Phase 5: Advanced Features (Week 5+)
**Only add if Phase 1-4 show consistent improvements**

Candidates:
- MIR sampling (Maximal Interfered Retrieval)
- Counterfactual augmentation
- Enhanced IRM (Invariant Risk Minimization)

**Each feature:**
1. Implement in isolation
2. Run ablation study
3. Keep only if improves performance
4. Document results for paper

---

## 📝 Paper Outline

### Title
"Causal-DER: Causal Reasoning for Dark Experience Replay in Continual Learning"

### Key Claims (Build Evidence Incrementally)
1. **Causal Importance** improves sample selection (Phase 2 results)
2. **Causal Sampling** reduces catastrophic forgetting (Phase 3 results)
3. **Task Graph Learning** discovers curriculum structure (Phase 4 results)
4. **Ablation Studies** show each component contributes (All phases)

### Experimental Setup
- Baseline: DER++ (56% Task-IL on seq-cifar100)
- Causal-DER: Incremental improvements documented
- Fair comparison: Same hyperparameters, same compute
- Multiple seeds: 3-5 runs per configuration

---

## 🔧 Technical Principles

### Code Quality
- ✅ Simple and readable (prefer clarity over cleverness)
- ✅ Well-documented (explain WHY, not just WHAT)
- ✅ Modular (each causal feature is a separate function)
- ✅ Testable (can disable any feature with a flag)

### Experimental Rigor
- ✅ One change at a time
- ✅ Proper ablation studies
- ✅ Fair baselines (same hyperparameters)
- ✅ Statistical significance (multiple seeds)

### Incremental Development
- ✅ Phase 1 must work before Phase 2
- ✅ Each phase: implement → test → measure → document
- ✅ Drop features that don't help
- ✅ Keep only what improves performance

---

## 📊 Success Metrics

### Phase 1 (Baseline)
- [x] Codebase cleaned
- [ ] DER++ baseline reproduced (56% Task-IL)
- [ ] Code is clean (<500 lines)
- [ ] No NaN issues

### Phase 2 (Causal Importance)
- [ ] Importance scoring implemented
- [ ] Ablation study completed
- [ ] Results documented
- [ ] Gain: +1-2% Task-IL (if any)

### Phase 3 (Causal Sampling)
- [ ] Sampling bias implemented
- [ ] Ablation study completed
- [ ] Results documented
- [ ] Gain: +2-3% Task-IL (cumulative)

### Phase 4 (Task Graph)
- [ ] Graph learning implemented
- [ ] Ablation study completed
- [ ] Results documented
- [ ] Gain: +3-5% Task-IL (cumulative)

### Final Target
- [ ] **Causal-DER: 60-65% Task-IL** (vs DER++ baseline 56%)
- [ ] **All ablations documented**
- [ ] **Paper draft ready**
- [ ] **Code ready for release**

---

## 🎯 Next Immediate Steps

1. **Create `training/causal_der_v2.py`**
   - Copy structure from `mammoth/models/derpp.py`
   - Implement clean DER++ engine
   - Add hooks for future causal features (commented out)

2. **Update `mammoth/models/causal_der.py`**
   - Import new CausalDEREngine from causal_der_v2
   - Keep same CLI arguments
   - Keep same observe() interface

3. **Test Baseline**
   - Run: `python mammoth/utils/main.py --model causal-der --dataset seq-cifar100 ...`
   - Expected: 56% Task-IL (± 2%)
   - If not matched: debug until it matches

4. **Document Baseline Results**
   - Create `BASELINE_RESULTS.md`
   - Record exact hyperparameters
   - Record performance across 3 seeds
   - This is our reference point

---

## 💡 Key Insights from Failed Attempt

**What Went Wrong:**
- ❌ Started with complex causal features before baseline worked
- ❌ Too much sanitization (torch.nan_to_num killed gradients)
- ❌ Causal code ran even when features "disabled"
- ❌ Lost track of what was causing issues
- ❌ Couldn't tell if bugs were in DER++ or causal parts

**What We Learned:**
- ✅ Always start with working baseline
- ✅ Add complexity incrementally
- ✅ One feature at a time
- ✅ Measure impact of each feature
- ✅ Keep code simple and readable

**Why This Approach Will Succeed:**
- ✅ We have a working DER++ reference (56%)
- ✅ We can copy exact implementation
- ✅ We can verify baseline before adding features
- ✅ We can measure exact contribution of each feature
- ✅ We can build publication-quality evidence

---

## 📚 References

**Mammoth DER++:**
- File: `mammoth/models/derpp.py` (60 lines, works perfectly)
- Performance: 56.06% Task-IL on seq-cifar100 (verified)
- Hyperparameters: alpha=0.1, beta=0.5, lr=0.03, momentum=0

**DER++ Paper:**
- Buzzega et al. (2020) "Dark Experience for General Continual Learning"
- Simple: CE(current) + α·MSE(replay_logits) + β·CE(replay_labels)
- No temperature scaling, no sanitization, just works

**Our Innovation:**
- Add causal reasoning on top of proven baseline
- Measure exact contribution of each component
- Build evidence for publication claims

---

**Status:** Ready to begin Phase 1 (Clean DER++ Baseline)  
**Confidence:** HIGH (we have working reference implementation)  
**Timeline:** 4-5 weeks to complete all phases  
**Goal:** Publishable results with clean ablation studies
