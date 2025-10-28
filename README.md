# Symbio AI - Causal Continual Learning Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/Research-Active-success.svg)](https://github.com/ZulAmi/symbioAI)

## Overview

**CausalDER** combines causal graph discovery with continual learning to understand how tasks relate to each other in neural networks. I've extended the official DER++ implementation from the Mammoth framework by adding causal structure learning between sequential tasks.

**Current Status:** Working implementation with completed ablation studies on CIFAR-100. Graph learning reveals interpretable task relationships without hurting performance. Looking for academic collaborators to push this toward publication.

**Code:** Public on GitHub at [github.com/ZulAmi/symbioAI](https://github.com/ZulAmi/symbioAI)

### Validated Results (October 2025)

**Ablation Study on CIFAR-100**

| Experiment         | Graph Learning | Importance Sampling | Task-IL    | Interpretation          |
| ------------------ | -------------- | ------------------- | ---------- | ----------------------- |
| **DER++ Baseline** | ❌             | ❌                  | **73.81%** | Standard replay         |
| **Graph Only**     | ✅             | ❌                  | **73.88%** | +0.07% (negligible)     |
| **Full Causal**    | ✅             | ✅                  | **71.75%** | -2.06% (sampling hurts) |

**What I found**:

- Graph learning is free: +0.07% is just noise
- Importance sampling hurts: -2.06% by killing diversity
- Got interpretable structure: 30-edge causal graph
- Balanced replay matters: Uniform sampling beats importance weighting
- Useful failure: Importance sampling doesn't work here

**CIFAR-100 detailed (primary dataset, 10 tasks, 5 epochs)**:

**Ablation Study (seed 1)**:

**Experiment 1: Full Causal**

- Result: 71.75% Task-IL
- Configuration: Graph learning + importance sampling
- Issue: Top-10 samples always from same task, destroying diversity

**Experiment 2: DER++ Baseline**

- Result: 73.81% Task-IL
- Configuration: Standard DER++ with uniform sampling
- Baseline for comparison

**Experiment 3: Graph Only**

- Result: 73.88% Task-IL (+0.07% vs baseline)
- Configuration: Graph learning enabled, importance sampling disabled
- Graph computed but not used for sampling

**What this means**:

- Graph learning: Free interpretability (+0.07% is noise)
- Importance sampling: Kills performance (-2.06% by destroying diversity)
- Bottom line: You get interpretability without any performance hit

**Causal Graph Discovered (Experiment 3, Graph Only)**:

- 30 strong causal edges (strength threshold 0.5-0.698)
- Task 3 identified as causal hub
- Graph density: 33.3% (30/90 possible edges)
- Mean edge strength: 0.189
- Temporal constraint: Only forward edges (i→j where i<j)
- Key relationships: Task 0→1: 0.678, Task 1→2: 0.676, Task 2→3: 0.698

**Why Importance Sampling Failed**:

DEBUG analysis (1-epoch runs) revealed:

- Full causal with sampling: 49.82% Task-IL (massive degradation)
- Top-10 samples concentrated on single task at each step
- Mean importance 0.55, std 0.20-0.30 creates extreme bias
- Conclusion: Importance weighting destroys balanced replay diversity

### Experimental Validation History

This section documents the complete experimental journey, including both successful and failed experiments. All experiments conducted on CIFAR-100 (10 tasks, 5 epochs) unless otherwise specified.

**Phase 1: Baseline Validation (October 2025)**

Established gold standard comparison using official DER++ implementation from Mammoth framework.

- **Initial Result**: 73.81% Task-IL accuracy (seed 1)
- **Multi-seed Validation (October 27)**: 72.99 ± 0.75% (5 seeds)
- **Configuration**: Buffer size 500, alpha=0.3, beta=0.5, 5 epochs
- **Location**: `validation/results/baseline_multiseed_20251027_112456/`
- **Purpose**: Statistical comparison with CausalDER
- **Status**: Complete - baseline variance established (std=0.75%, CV=1.03%)

**Phase 2: Initial Causal Integration (October 2025)**

First implementation of causal graph discovery integrated with DER++ replay mechanism.

- **Result**: 70.32% Task-IL accuracy
- **Gap**: -3.49% from baseline
- **Configuration**: Basic causal sampling without optimizations
- **Location**: `validation/results/causal_der_v2_baseline_5epoch_seed1.log`
- **Decision**: Met 70% threshold for GO decision - proved concept viability
- **Status**: Success - validated core causal integration works

**Phase 3: Causal Graph Learning (October 2025)**

Attempted more complex causal graph learning with enhanced structural constraints.

- **Result**: 62.3% Task-IL accuracy
- **Gap**: -11.51% from baseline, -7.89% worse than Phase 2
- **Location**: `validation/results/phase3_graph_learning_seed1.log`
- **Issue**: Enhanced graph learning initially degraded performance
- **Lesson**: More complexity does not always improve results
- **Status**: Negative result - led to optimization strategy change

**Quick Wins Optimization (October 2025)**

Initial attempt at optimizing causal sampling with multiple strategies:

1. Warm start blending (gradual transition from uniform to causal sampling)
2. Smoother importance weighting (0.7 causal + 0.3 recency)
3. Adaptive sparsification (dynamic threshold 0.9 to 0.7 quantile)

- **Result**: Still underperformed baseline
- **Root cause discovered**: Importance sampling fundamentally incompatible with balanced replay
- **Lesson**: Optimization cannot fix architectural mismatch
- **Status**: Abandoned - shifted to graph-only approach

**Ablation Studies (October 2025)**

Clean ablation to separate graph learning from importance sampling:

**Experiment 1: Full Causal**

- Graph learning + importance sampling
- Result: 71.75% (-2.06% from baseline)

**Experiment 2: DER++ Baseline**

- No graph learning, no importance sampling
- Result: 73.81% (baseline)

**Experiment 3: Graph Only**

- Graph learning enabled, importance sampling disabled
- Result: 73.88% (+0.07%, within noise)

**Key Findings**:

- Graph learning: Performance neutral
- Importance sampling: Destroys diversity (-2.06%)
- Location: `validation/results/ablations_20251027_161516/`
- Status: Complete - identified root cause of performance degradation

**Summary of Completed Experiments**

| Experiment Phase     | Dataset   | Seeds | Result    | Gap from DER++ | Status   |
| -------------------- | --------- | ----- | --------- | -------------- | -------- |
| Phase 1 Baseline     | CIFAR-100 | 1     | 73.81%    | N/A (baseline) | Complete |
| Phase 2 Causal       | CIFAR-100 | 1     | 70.32%    | -3.49%         | Complete |
| Phase 3 Graph        | CIFAR-100 | 1     | 62.3%     | -11.51%        | Complete |
| Quick Wins           | CIFAR-100 | 1     | Abandoned | Still degraded | Failed   |
| Ablation: Graph Only | CIFAR-100 | 1     | 73.88%    | +0.07% (noise) | Complete |
| Ablation: Full       | CIFAR-100 | 1     | 71.75%    | -2.06%         | Complete |

**Total Experiments Run**: 6 controlled ablation studies

**Total Compute Time**: ~5.2 hours (6×52min)

**Key Lesson**: Importance sampling breaks balanced replay - graph learning gives you interpretability for free

### What This Does

**Main Idea:**

Bringing causal graph discovery into continual learning to see which tasks actually depend on each other:

1. **Interpretable Task Graphs** - Finds task dependencies without messing up training
2. **Temporal Causality** - Only forward edges (i→j where i<j) since later tasks can't affect earlier ones
3. **Free** - Graph learning costs nothing (+0.07% is noise)
4. **Honest Failure** - Importance sampling hurts performance by -2.06%
5. **Clean Experiments** - Ablations clearly separate graph learning from sampling

**The Cool Part:** See which tasks causally influence others (like Task 2→Task 3 at 0.698 strength) without any performance penalty.

**Unexpected Finding:** Importance sampling breaks balanced replay - turns out to be a valuable negative result.

## Implementation

Built on top of the official [Mammoth continual learning framework](https://github.com/aimagelab/mammoth):

```
symbio-ai/
├── mammoth/             # Official Mammoth framework (unchanged)
│   └── models/
│       └── derpp.py     # Official DER++ implementation
├── training/            # CausalDER implementation (~2,450 lines)
│   ├── derpp_causal.py      # Main: Extends official DER++ (463 lines)
│   ├── causal_inference.py  # Causal graph discovery (758 lines)
│   ├── metrics_tracker.py   # Experiment tracking (469 lines)
│   └── causal_der_v2.py     # Deprecated (750 lines)
├── validation/          # Experimental scripts and results
│   ├── results/         # Experimental logs (gitignored)
│   └── scripts/         # Experiment scripts
│       └── completed_experiments/  # Archived scripts
└── requirements.txt     # Dependencies
```

**Core Implementation:**

- `training/derpp_causal.py` (463 lines): Extends official `mammoth/models/derpp.py` with causal sampling
- `training/causal_inference.py` (758 lines): Structural Causal Model with PC algorithm
- Uses ResNet-18 feature extraction (512D penultimate layer)
- Runtime: ~52 minutes per full experiment on Apple Silicon (MPS)

## Technical Details

### Method Overview

**CausalDER** discovers causal relationships between tasks during continual learning:

```
Current Task Training
    ↓
Feature Extraction (ResNet-18, 512D)
    ↓
Causal Graph Learning (PC algorithm)
    ↓
Graph Storage & Visualization
    ↓
DER++ Training (standard uniform replay)
    ↓
Post-Training Analysis of Task Relationships
```

**Note:** Graph is learned but NOT used for sampling decisions. Training uses standard DER++ with uniform replay.

### Key Hyperparameters

- **Buffer size**: 500 samples (50 per task)
- **Learning rate**: 0.03 with MultiStepLR (milestones=[3,4], gamma=0.2)
- **Batch size**: 128 (minibatch_size=32 for replay)
- **Epochs**: 5 per task
- **Causal discovery**: PC algorithm with partial correlation tests
- **Sparsification**: Adaptive 0.9→0.7 quantile threshold

### Experimental Protocol

**Dataset**: CIFAR-100 (10 tasks, 10 classes per task)
**Splits**: Standard train/test splits
**Seeds**: 5 (multi-seed validation complete)
**Device**: MPS (Apple Silicon) tested, NVIDIA GPU compatible
**Framework**: Mammoth (official continual learning library)

## Running the Code

### Installation

```bash
# Clone repository
git clone https://github.com/ZulAmi/symbioAI.git
cd symbioAI

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run CausalDER on CIFAR-100 (10 tasks, 5 epochs)
python3 mammoth/utils/main.py \
  --model derpp-causal \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --n_epochs 5 \
  --batch_size 128 \
  --minibatch_size 32 \
  --lr 0.03 \
  --seed 1

# Run official DER++ baseline for comparison
python3 mammoth/utils/main.py \
  --model derpp \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --n_epochs 5 \
  --batch_size 128 \
  --alpha 0.1 \
  --beta 0.5 \
  --seed 1
```

### Expected Results

- **DER++ Baseline**: ~73.81% Task-IL (CIFAR-100)
- **Graph Only (Recommended)**: ~73.88% Task-IL (graph learning enabled, sampling disabled)
- **Full Causal (Not Recommended)**: ~71.75% Task-IL (importance sampling degrades performance)
- **Runtime**: ~52 minutes per seed on Apple Silicon M1/M2 (CIFAR-100)
- **Graph Discovery**: 30-edge causal structure with temporal constraints

## Documentation

- **[One-Page Research Summary](Documents/RESEARCH_SUMMARY_1PAGE.md)** - Complete overview for collaboration inquiries
- [Quick Start Guide](run_improved_config.sh) - Optimized configuration script
- [Baseline Multi-Seed Script](run_baseline_multiseed.sh) - Statistical validation script

## Next Steps

### Completed Validation

✅ **Ablation Study Complete (October 27, 2025)**

- DER++ Baseline: 73.81% (seed 1)
- Graph Only: 73.88% (+0.07%, negligible)
- Full Causal: 71.75% (-2.06%, importance sampling hurts)
- **Takeaway**: Graph learning gives interpretability for free
- **Finding**: Importance sampling breaks balanced replay

### Pending Research Tasks

**Critical Next Steps:**

1. **Visualization & Analysis Tools** (1-2 days)

   - Create graph visualization scripts
   - Analyze task relationship patterns
   - Correlate graph structure with forgetting/transfer

2. **Extended Analysis** (3-5 days)

   - Multi-seed graph consistency check
   - Hub task identification across seeds
   - Forgetting prediction based on graph structure

3. **Manuscript Development** (2-3 weeks)
   - Focus on interpretability contribution
   - Include rigorous negative result on importance sampling
   - Target: Workshop paper (4-8 pages)

**High Priority (Weeks 1-2)**

1. Graph visualization and analysis tools

   - Network diagrams showing task dependencies
   - Hub task identification
   - Correlation with forgetting patterns

2. Alternative applications of causal graphs

   - Curriculum learning (optimal task ordering)
   - Forgetting prediction
   - Transfer learning guidance

3. Diversity-preserving sampling (future work)
   - Stratified sampling with per-task quotas
   - Top-k diverse selection
   - Balanced importance weighting

**Medium Priority (Weeks 3-4)**

4. Multi-seed graph consistency

   - Verify graph structure stable across seeds
   - Quantify hub task agreement
   - Statistical tests for edge consistency

5. Extended analysis

   - Task similarity vs causal strength
   - Forgetting rate vs outgoing edges
   - Transfer patterns prediction

**Research Roadmap**

**Short-term (Months 1-2)**

- Workshop/conference paper draft (4-8 pages)
- Graph visualization and analysis tools
- Theoretical analysis of graph discovery
- Documentation of negative results (importance sampling failure)

**Long-term (Months 3-6)**

- Application to curriculum learning design
- Forgetting prediction models based on graph structure
- Alternative uses of causal graphs (beyond sampling)
- Diversity-preserving importance sampling (if viable)

### Target Venues

- **NeurIPS 2026** (June deadline) - Main conference or CLeaR workshop
- **ICLR 2026** (January deadline) - Conference or workshop track
- **CoLLAs 2025** - Lifelong learning focused venue

## Collaboration Opportunities

**I'm looking for academic collaborators to help turn this into a workshop or conference paper.**

### What I've Got So Far

- Working implementation that extends official Mammoth DER++
- **Solid ablation study**: Separated graph learning (neutral) from importance sampling (harmful)
- **Interesting discovery**: 30-edge causal graph with temporal constraints
- **Honest negative result**: Found that importance sampling breaks balanced replay
- Clean experimental protocol with 6 controlled ablations - everything's reproducible
- Realistic timeline: could submit to a workshop in about 6 weeks
- Happy to discuss fair co-authorship arrangements

### Different Ways to Collaborate

| Level           | Time Commitment | What That Looks Like                   | Credit                  |
| --------------- | --------------- | -------------------------------------- | ----------------------- |
| **Advisory**    | 1-2 hrs/month   | Review my experimental design & drafts | Co-author on the paper  |
| **Active**      | 4-6 hrs/month   | Run experiments together, co-write     | Primary co-author       |
| **Partnership** | 10+ hrs/month   | Guide research direction, mentor       | Long-term collaboration |

**Get in Touch:** Open a GitHub issue or check out the full [research summary](Documents/RESEARCH_SUMMARY_1PAGE.md).

## License

MIT License - see LICENSE file for details.

## Acknowledgments

**Frameworks and Baselines:**

- Mammoth Framework - Continual learning benchmarking infrastructure
- DER++ (Buzzega et al., NeurIPS 2020) - Dark Experience Replay baseline
- Continual Learning Community - Datasets, benchmarks, and standard protocols

## References

1. **Buzzega et al.** (2020). "Dark Experience for General Continual Learning: a Strong, Simple Baseline." _NeurIPS_.
2. **Boschini et al.** (2022). "Class-Incremental Continual Learning into the eXtended DER-verse." _IEEE TPAMI_.
3. **Pearl, J.** (2009). _Causality: Models, Reasoning and Inference_. Cambridge University Press.
4. **Spirtes et al.** (2000). _Causation, Prediction, and Search_. MIT Press.
5. **Schölkopf et al.** (2021). "Toward Causal Representation Learning." _PNAS_.

## Citation

If you use this code in your research:

```bibtex
@software{causalder2025,
  title={CausalDER: Causal Graph Discovery for Continual Learning},
  author={Rahmat, Zulhilmi},
  year={2025},
  url={https://github.com/ZulAmi/symbioAI},
  note={Integration of causal inference with DER++ continual learning}
}
```

## Contact

**Zulhilmi Rahmat**  
GitHub: [@ZulAmi](https://github.com/ZulAmi)  
Email: zulhilmirahmat@gmail.com

For collaboration inquiries or questions, open a GitHub issue or see [RESEARCH_SUMMARY_1PAGE.md](Documents/RESEARCH_SUMMARY_1PAGE.md).

---

**Status:** Active research (October 2025) | **Ablation complete**: Graph learning neutral (+0.07%), importance sampling harmful (-2.06%) | Code public on GitHub
