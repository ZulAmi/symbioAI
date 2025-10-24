# Symbio AI - Causal Continual Learning Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/Research-Active-success.svg)](https://github.com/ZulAmi/symbioAI)

## Overview

**CausalDER** integrates causal graph discovery with continual learning to address catastrophic forgetting in neural networks. This research project extends the official DER++ implementation (Mammoth framework) with causal relationship modeling between sequential tasks.

**Research Status:** Working implementation with validated results (72.11% Task-IL on CIFAR-100). Seeking academic collaboration for publication.

### Validated Results (October 2025)

**CIFAR-100 Sequential (10 tasks, 5 epochs per task)**

| Method               | Task-IL    | Gap to Baseline | Key Features             |
| -------------------- | ---------- | --------------- | ------------------------ |
| Official DER++       | 73.81%     | -               | Memory replay baseline   |
| **CausalDER (Ours)** | **72.11%** | **-1.70%**      | + Causal graph discovery |

**Causal Graph Discovered:**

- 30 strong causal edges (strength 0.5-0.69)
- Task 3 identified as causal hub (influences 4+ downstream tasks)
- Graph density: 33.3% (30/90 possible edges)

### Research Contributions

**Novel Method:**

First integration of causal graph discovery into memory-based continual learning:

1. **Causal-Weighted Replay** - Samples prioritized by discovered task dependencies
2. **Adaptive Sparsification** - Dynamic threshold (0.9→0.7 quantile) as tasks progress
3. **Warm Start Blending** - Gradual transition from uniform to causal sampling
4. **Minimal Trade-off** - Only 1.70% performance cost for explainable task relationships

**Key Innovation:** Discovers which tasks causally depend on others (e.g., Task 3→Task 4 with 0.686 strength), enabling interpretable replay strategies.

## Implementation

Built on top of the official [Mammoth continual learning framework](https://github.com/aimagelab/mammoth):

```
symbio-ai/
├── mammoth/             # Official Mammoth framework (unchanged)
│   └── models/
│       └── derpp.py     # Official DER++ implementation
├── training/            # CausalDER implementation (~2,400 lines)
│   ├── derpp_causal.py      # Main: Extends official DER++ (443 lines)
│   ├── causal_inference.py  # Causal graph discovery (758 lines)
│   ├── metrics_tracker.py   # Experiment tracking (469 lines)
│   └── causal_der_v2.py     # Deprecated (750 lines)
├── validation/          # Test scripts and results
│   └── results/         # Experimental logs
├── documents/           # Research documentation
│   └── RESEARCH_SUMMARY_1PAGE.md  # One-page summary for outreach
└── requirements.txt     # Dependencies
```

**Core Implementation:**

- `training/derpp_causal.py` (443 lines): Extends official `mammoth/models/derpp.py` with causal sampling
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
Causal-Weighted Replay Sampling
    importance = 0.7 × causal_strength + 0.3 × recency
    ↓
DER++ Loss (classification + distillation + replay)
```

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
**Seed**: 1 (multi-seed validation pending)
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

- **Official DER++**: ~73.81% Task-IL
- **CausalDER**: ~72.11% Task-IL (with 30-edge causal graph)
- **Runtime**: ~52 minutes on Apple Silicon M1/M2

## Documentation

- **[One-Page Research Summary](documents/RESEARCH_SUMMARY_1PAGE.md)** - Complete overview for collaboration inquiries
- [Continual Learning Quick Start](docs/continual_learning_quick_start.md)
- [Architecture Overview](docs/architecture.md)

## Next Steps

### Research Roadmap

**Immediate (Weeks 1-2):**

- Multi-seed validation (5 seeds) for statistical significance
- Additional datasets (MNIST, CIFAR-10, TinyImageNet)
- Ablation studies (warm start, importance blending, sparsification)

**Short-term (Weeks 3-4):**

- Workshop/conference paper draft (4-8 pages)
- Comparison with SOTA causal learning methods
- Theoretical analysis of causal discovery complexity

**Long-term (Months 3-6):**

- Application to real-world domains (robotics, vision, NLP)
- Extended causal inference methods (interventions, counterfactuals)
- Full NeurIPS/ICLR conference paper

### Target Venues

- **NeurIPS 2026** (June deadline) - Main conference or CLeaR workshop
- **ICLR 2026** (January deadline) - Conference or workshop track
- **CoLLAs 2025** - Lifelong learning focused venue

## Collaboration Opportunities

**Seeking academic collaboration for co-authorship on workshop/conference paper.**

### What We're Looking For

**Expertise Areas** (any of):

- Causal inference (improving graph discovery, theoretical guarantees)
- Continual learning (optimizing replay strategies, multi-dataset validation)
- Theory (sample complexity bounds, convergence analysis)

### What We Bring

- ✅ Working implementation extending official Mammoth DER++
- ✅ Validated results (72.11% Task-IL with 30-edge causal graph)
- ✅ Clear experimental protocol (reproducible)
- ✅ 6-week timeline to workshop submission
- ✅ Open to co-authorship with fair credit

### Collaboration Levels

| Level           | Time Commitment | Role                            | Benefits                |
| --------------- | --------------- | ------------------------------- | ----------------------- |
| **Advisory**    | 1-2 hrs/month   | Review design & drafts          | Co-authorship           |
| **Active**      | 4-6 hrs/month   | Joint experiments & writing     | Primary co-author       |
| **Partnership** | 10+ hrs/month   | Research direction & mentorship | Long-term collaboration |

**Contact:** Open a GitHub issue or see [RESEARCH_SUMMARY_1PAGE.md](documents/RESEARCH_SUMMARY_1PAGE.md) for full details.

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

For collaboration inquiries or questions, open a GitHub issue or see [RESEARCH_SUMMARY_1PAGE.md](documents/RESEARCH_SUMMARY_1PAGE.md).

---

**Status:** Active research (October 2025) | Validated 72.11% Task-IL on CIFAR-100 | Seeking collaboration
