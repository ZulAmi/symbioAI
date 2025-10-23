# Symbio AI - Causal Continual Learning Research Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/Research-Active-success.svg)](https://github.com/ZulAmi/symbioAI)

## Overview

Symbio AI is a comprehensive continual learning research platform pioneering causal reasoning approaches to overcome catastrophic forgetting in neural networks. The platform features 30+ specialized training modules spanning over 30,000 lines of production-quality code, implementing novel algorithms that combine causal inference with adaptive replay strategies.

**Research Status:** Active development with validated baselines and systematic parameter optimization in progress.

### Key Achievements

**Phase 1: Validated Baseline (October 2025)**

- 70.19% Task-IL accuracy on CIFAR-100 (10 tasks, 5 epochs)
- Clean DER++ replication with competitive performance
- Full integration with Mammoth benchmarking framework

**Phase 2: Importance-Weighted Replay (October 2025)**

- Novel importance sampling formula: `importance = loss × uncertainty²`
- Adaptive replay strategy: 70% importance-weighted, 30% random
- Theoretical framework for causal sample attribution

**Phase 3: Causal Graph Learning (October 2025)**

- Complete implementation of Structural Causal Models (SCM)
- Automatic task dependency discovery via causal inference
- Interventional reasoning with counterfactual generation
- Current status: Parameter optimization (initial result: 62.30% Task-IL)

### Research Contributions

**Novel Methods:**

1. **Causal Importance Estimation** - First framework applying causal inference to replay sample selection
2. **Task Dependency Graphs** - Automatic discovery of task relationships through SCM
3. **Interventional Learning** - Integration of counterfactual reasoning into continual learning
4. **Parameter Optimization** - Systematic exploration with scientific controls and multi-seed validation

**Current Focus:** Parameter sweep experiments to optimize Phase 3 performance, targeting 71%+ Task-IL for publication.

## Architecture

The platform is organized into modular components ensuring reproducibility and extensibility:

```
symbio-ai/
├── config/              # Centralized configuration management
├── data/                # Dataset storage and preprocessing
├── docs/                # Technical documentation
├── mammoth/             # Mammoth framework integration
├── training/            # Core training modules (30,000+ lines)
│   ├── causal_der_v2.py             # Phase 1-3 implementation
│   ├── causal_inference.py          # Structural Causal Models
│   ├── continual_learning.py        # Primary CL orchestrator
│   ├── advanced_continual_learning.py  # Advanced strategies
│   ├── der_plus_plus.py             # DER++ baseline
│   └── [25+ specialized modules]    # Research prototypes
├── validation/          # Benchmarking and validation
│   ├── results/                     # Experimental results
│   └── tier1_continual_learning/    # Production experiments
└── requirements.txt     # Dependencies
```

**Implementation Status:**

- Core continual learning: 1,539 lines (orchestrator) + 1,251 lines (advanced)
- Causal DER v2: 684 lines (validated baseline)
- Causal inference: 328 lines (SCM implementation)
- Specialized modules: 27 modules totaling 29,000+ lines

## Validated Research Results

### Continual Learning Benchmarks

**MNIST Sequential (10 tasks)**

- Task-Incremental Accuracy: 97.44%
- Average Forgetting: 3.04%
- Protocol: 50 epochs per task

**CIFAR-10 Sequential (5 tasks)**

- Task-Incremental Accuracy: 84.67%
- Average Forgetting: 12.83%
- Protocol: 50 epochs per task
- Performance: Competitive with SOTA methods

**CIFAR-100 Sequential (10 tasks)**

- Baseline (Phase 1): 70.19% Task-IL (5 epochs)
- Phase 3 (default params): 62.30% Task-IL
- Status: Parameter optimization in progress
- Target: 71%+ Task-IL for publication
- Protocol: 10 classes per task, scientific controls maintained

### Phase 3: Causal Graph Learning

**Implementation Components:**

- Structural Causal Model with Pearl's causal hierarchy
- Feature extraction and caching (200 samples per task)
- Graph sparsification (quantile-based thresholding)
- Interventional reasoning and counterfactual generation
- Automatic dependency discovery

**Parameter Optimization Framework:**

- Cache size sweep: [0, 50, 100, 200] samples
- Sparsification sweep: [0.3, 0.5, 0.7, 0.9, None] quantiles
- Scientific controls: All baseline parameters frozen
- Multi-seed validation: 5 seeds for statistical significance
- Automated analysis and decision logic

## Core Continual Learning Methods

Production-ready implementations of state-of-the-art algorithms:

**Anti-Forgetting Strategies:**

- **Elastic Weight Consolidation (EWC)** - Fisher Information Matrix-based parameter protection
- **Experience Replay** - Intelligent memory buffer with importance-weighted sampling
- **Progressive Neural Networks** - Task-specific architectural expansion with lateral connections
- **Task-Specific Adapters** - Parameter-efficient fine-tuning via LoRA-inspired layers
- **DER++ Baseline** - Faithful replication of Buzzega et al. (NeurIPS 2020)

**Causal Extensions:**

- **Importance-Weighted Replay** - Samples prioritized by loss and uncertainty
- **Structural Causal Models** - Task dependency discovery
- **Interventional Reasoning** - Counterfactual-based learning
- **Graph-Based Replay** - Task relationships guide sample selection

**Documentation:** See `docs/continual_learning_quick_start.md` for implementation guides.

## Research Modules

The platform includes 25+ specialized modules representing active research directions:

**Meta-Learning and Transfer (3,563 lines)**

- recursive_self_improvement.py (963 lines) - Meta-evolution of learning strategies
- cross_task_transfer.py (1,190 lines) - Automatic transfer relationship discovery
- one_shot_meta_learning.py (1,410 lines) - Few-shot adaptation mechanisms

**Reasoning and Diagnosis (5,758 lines)**

- metacognitive_monitoring.py (1,567 lines) - Self-awareness and confidence estimation
- causal_self_diagnosis.py (2,515 lines) - Causal failure diagnosis
- automated_theorem_proving.py (1,344 lines) - Formal verification
- neural_symbolic_architecture.py (2,332 lines) - Hybrid reasoning systems

**Architecture and Optimization (4,494 lines)**

- dynamic_architecture_evolution.py (1,267 lines) - Adaptive network structures
- quantization_aware_evolution.py (1,106 lines) - Compression-aware evolution
- sparse_mixture_adapters.py (1,219 lines) - Massive adapter libraries
- memory_enhanced_moe.py (902 lines) - Memory-augmented MoE

**Multi-Modal and Agents (3,566 lines)**

- unified_multimodal_foundation.py (1,061 lines) - Cross-modal learning
- embodied_ai_simulation.py (1,224 lines) - Physical environment interaction
- multi_agent_collaboration.py (1,281 lines) - Agent coordination protocols

**Additional Systems (6,371 lines)**

- active_learning_curiosity.py (1,090 lines)
- compositional_concept_learning.py (1,358 lines)
- multi_scale_temporal_reasoning.py (951 lines)
- speculative_execution_verification.py (1,154 lines)
- advanced_evolution.py (1,244 lines)
- evolution.py (946 lines)
- distill.py (892 lines)
- manager.py, auto_surgery.py (736 lines)

**Note:** Research modules are experimental prototypes undergoing validation. Performance claims require benchmark verification.

## Documentation

- [Continual Learning Quick Start](docs/continual_learning_quick_start.md)
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Phase 3 Implementation Verification](PHASE3_IMPLEMENTATION_VERIFICATION.md)
- [Parameter Optimization Guide](PARAMETER_EXPERIMENTS_README.md)

## Contributing

This is an active research project. Contributions are welcome in the following areas:

- Continual learning algorithm improvements
- Validation of experimental modules
- Bug fixes and documentation
- Benchmark comparisons

**Process:**

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with clear description

## Intellectual Property

This repository demonstrates research platform capabilities while protecting core algorithmic innovations.

**Public Components:**

- Framework integration code
- Benchmark infrastructure
- Documentation and methodology
- Experimental protocols

**Proprietary Components:**

- Novel causal learning algorithms
- Advanced optimization techniques
- Proprietary training procedures

For collaboration inquiries regarding full implementations, please open a GitHub issue.

## Collaboration Opportunities

Open to partnerships in:

**Research Institutions**

- Joint validation studies
- Benchmark development
- Co-authorship on publications

**Industry Partners**

- Applied continual learning research
- Production deployment
- Custom algorithm development

**Academic Collaborations**

- Peer-reviewed publication co-authorship
- Graduate student projects
- Shared computational resources

**Resource Providers**

- Access to computational infrastructure
- Large-scale experiment support
- Dataset contributions

Contact: Open a GitHub issue or use repository discussions.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

**Frameworks and Baselines:**

- Mammoth Framework - Continual learning benchmarking infrastructure
- DER++ (Buzzega et al., NeurIPS 2020) - Dark Experience Replay baseline
- Continual Learning Community - Datasets, benchmarks, and standard protocols

## References

**Primary Baseline:**

Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). "Dark Experience for General Continual Learning: a Strong, Simple Baseline." Advances in Neural Information Processing Systems, 33.

**Mammoth Framework:**

Boschini, M., Bonicelli, L., Buzzega, P., Porrello, A., & Calderara, S. (2022). "Class-Incremental Continual Learning into the eXtended DER-verse." IEEE Transactions on Pattern Analysis and Machine Intelligence.

**Causal Inference:**

Pearl, J. (2009). Causality: Models, Reasoning and Inference. Cambridge University Press.

## Citation

If you use this platform or methodology in your research:

```bibtex
@software{symbioai2025,
  title={Symbio AI: Causal Continual Learning Research Platform},
  author={Rahmat, Zulhilmi},
  year={2025},
  url={https://github.com/ZulAmi/symbioAI},
  note={Research platform for continual learning with causal reasoning}
}
```

## Author

**Zulhilmi Rahmat**

AI/ML Research Engineer specializing in:

- Continual Learning
- Causal Inference
- Meta-Learning
- Neural-Symbolic Systems

GitHub: [@ZulAmi](https://github.com/ZulAmi)

For professional inquiries, collaboration proposals, or technical questions, use GitHub Issues or Discussions.

---

**Repository Status:** Active research with validated baselines and ongoing parameter optimization experiments.
